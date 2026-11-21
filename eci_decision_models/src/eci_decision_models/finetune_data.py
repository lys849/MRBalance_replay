"""Build the EventStoryLine v0.9 intra-sentence pool and Laya fine-tune folds.

The module walks the raw XML corpus, reconstructs sentences and event text,
labels every unordered intra-sentence event pair as ``cause`` / ``caused_by`` /
``none`` from the official ``full_corpus/v0.9`` evaluation export, and writes
deterministic artifacts plus Gao/iLIF-style fold-0 Laya JSONL files.

Label resolution follows the specification exactly:

1. A direct ``event_mentions_extended`` row for the ordered token pair wins.
2. Otherwise a ``coref_chain`` row is used when the two event token tuples sit
   on opposite sides of one coreference row.
3. Conflicting rows make the pair ambiguous and remove it from the pool.

Relations are canonicalised so that ``event1`` is always the earlier mention:
``PRECONDITION`` becomes ``cause`` and ``FALLING_ACTION`` becomes
``caused_by``.  The private helpers in :mod:`eci_decision_models.gold` are
reused so the orientation and token-tuple rules stay identical to the gold
reconstruction code.

The published reference counts are 7,805 intra-sentence pairs and 1,770 causal
pairs.  The builder refuses to write fold files when either count falls outside
the documented tolerance; it never rewrites the published target to match its
own output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from eci_decision_models.gold import (
    _pair_key,
    _parse_coref_rows,
    _parse_official_index,
    _relation_in_order,
)
from eci_decision_models.laya import build_laya_request

NAMESPACE: Final = "esl-intra:v1:"
EXCLUDED_TAG_FRAGMENTS: Final = (
    "ASPECTUAL",
    "CAUSATIVE",
    "PERCEPTION",
    "REPORTING",
)

TARGET_TOTAL_PAIRS: Final = 7805
TARGET_CAUSAL_PAIRS: Final = 1770
TOTAL_TOLERANCE: Final = 0.03
CAUSAL_TOLERANCE: Final = 0.05

DEV_TOPICS: Final = frozenset({"37", "41"})
SPLIT_SEED: Final = 209
N_FOLDS: Final = 5

STATE_KEYS: Final = frozenset({"sentence", "event1", "event2"})
CHOICE_KEYS: Final = frozenset({"cause", "caused_by", "none"})
EXPECTED_VALUES: Final = frozenset({"cause", "caused_by", "none"})

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_XML_ROOT: Final = PROJECT_ROOT / "data/EventStoryLine/annotated_data/v0.9"
DEFAULT_EVAL_ROOT: Final = PROJECT_ROOT / "data/EventStoryLine/evaluation_format/full_corpus/v0.9"
DEFAULT_ARTIFACTS_DIR: Final = PROJECT_ROOT / "artifacts/intra_pairs/v1"
DEFAULT_EXPERIMENTS_DIR: Final = PROJECT_ROOT / "experiments/pilot-fold0/data"


@dataclass(frozen=True)
class Token:
    """One XML token with its sentence-local ordering metadata."""

    t_id: int
    sentence: int
    number: int
    text: str


@dataclass(frozen=True)
class EventMention:
    """A non-excluded event mention anchored to a single sentence."""

    m_id: str
    tag: str
    tokens: tuple[int, ...]
    sentence: int
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    """Tokens, reconstructed sentence text and in-sentence event mentions."""

    doc_name: str
    topic: str
    tokens: dict[int, Token]
    sentence_text: dict[int, str]
    events: tuple[EventMention, ...]
    excluded_events: int
    skipped_no_anchor: int
    skipped_multi_sentence: int


@dataclass(frozen=True)
class PairRecord:
    """One labelled intra-sentence event pair."""

    record_id: str
    doc_name: str
    topic: str
    sent_in_doc_id: int
    sentence: str
    event1: str
    event2: str
    event1_tokens: tuple[int, ...]
    event2_tokens: tuple[int, ...]
    label: str
    provenance: str
    relation: str | None

    def payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "record_id": self.record_id,
            "doc_name": self.doc_name,
            "topic": self.topic,
            "sent_in_doc_id": self.sent_in_doc_id,
            "sentence": self.sentence,
            "event1": self.event1,
            "event2": self.event2,
            "event1_tokens": list(self.event1_tokens),
            "event2_tokens": list(self.event2_tokens),
            "label": self.label,
            "provenance": self.provenance,
        }
        if self.relation is not None:
            payload["relation"] = self.relation
        return payload


@dataclass(frozen=True)
class PoolResult:
    """A complete labelled pool plus its summary counters."""

    pairs: tuple[PairRecord, ...]
    doc_names: tuple[str, ...]
    summary: Mapping[str, object]


@dataclass(frozen=True)
class SplitResult:
    """Gao/iLIF fold-0 document split derived from the CV topics."""

    train_docs: tuple[str, ...]
    test_docs: tuple[str, ...]
    dev_docs: tuple[str, ...]
    cv_docs: tuple[str, ...]
    chunks: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class GaoSplit:
    """One Gao/iLIF cross-validation fold derived from the shared chunking."""

    fold: int
    train_docs: tuple[str, ...]
    test_docs: tuple[str, ...]
    dev_docs: tuple[str, ...]
    cv_docs: tuple[str, ...]
    chunks: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class DatasetResult:
    """Pair records partitioned by a document split."""

    train: tuple[PairRecord, ...]
    test: tuple[PairRecord, ...]
    dev: tuple[PairRecord, ...]


@dataclass(frozen=True)
class FoldPlan:
    """A fully materialised Gao fold, including its balanced training rows."""

    split: GaoSplit
    train_pairs: tuple[PairRecord, ...]
    test_pairs: tuple[PairRecord, ...]
    sample_seed: int


class Fold0BalanceMismatchError(RuntimeError):
    """Raised when generated fold-0 rows differ from the existing balanced file."""

    def __init__(self, reference: Path, detail: str) -> None:
        super().__init__(f"fold-0 balanced rows do not match {reference}: {detail}")
        self.reference = reference
        self.detail = detail


class CountMismatchError(RuntimeError):
    """Raised when the built pool misses the published reference counts."""

    def __init__(
        self,
        summary: Mapping[str, object],
        check: Mapping[str, object],
    ) -> None:
        super().__init__("intra-sentence pool does not match the published counts")
        self.summary = summary
        self.check = check


def _local_name(tag: object) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _is_event_tag(name: str) -> bool:
    return name.startswith("ACTION_") or name.startswith("NEG_ACTION_")


def _is_excluded_tag(name: str) -> bool:
    return any(fragment in name for fragment in EXCLUDED_TAG_FRAGMENTS)


def parse_document(payload: bytes, doc_name: str) -> ParsedDocument:
    """Parse one XML document into tokens, sentences and event mentions."""

    root = ET.fromstring(payload)
    topic = doc_name.split("/", 1)[0]

    tokens: dict[int, Token] = {}
    for element in root.iter():
        if _local_name(element.tag) != "token":
            continue
        t_id = int(element.get("t_id", ""))
        tokens[t_id] = Token(
            t_id=t_id,
            sentence=int(element.get("sentence", "")),
            number=int(element.get("number", "")),
            text=element.text or "",
        )

    by_sentence: dict[int, list[Token]] = defaultdict(list)
    for token in tokens.values():
        by_sentence[token.sentence].append(token)
    sentence_text = {
        sentence: " ".join(token.text for token in sorted(items, key=lambda item: item.number))
        for sentence, items in by_sentence.items()
    }

    events: list[EventMention] = []
    seen: set[str] = set()
    excluded_events = 0
    skipped_no_anchor = 0
    skipped_multi_sentence = 0

    for element in root.iter():
        name = _local_name(element.tag)
        if not _is_event_tag(name):
            continue
        m_id = element.get("m_id")
        if m_id is None or m_id in seen:
            continue
        seen.add(m_id)
        if _is_excluded_tag(name):
            excluded_events += 1
            continue

        anchors: list[int] = []
        for child in element:
            if _local_name(child.tag) != "token_anchor":
                continue
            t_id_text = child.get("t_id")
            if t_id_text is None:
                continue
            anchors.append(int(t_id_text))
        if not anchors:
            skipped_no_anchor += 1
            continue

        anchor_tokens = tuple(sorted(set(anchors)))
        sentences = {tokens[t_id].sentence for t_id in anchor_tokens if t_id in tokens}
        if len(sentences) != 1:
            skipped_multi_sentence += 1
            continue
        sentence = next(iter(sentences))
        text = " ".join(tokens[t_id].text for t_id in anchor_tokens)
        events.append(
            EventMention(
                m_id=m_id,
                tag=name,
                tokens=anchor_tokens,
                sentence=sentence,
                text=text,
            )
        )

    return ParsedDocument(
        doc_name=doc_name,
        topic=topic,
        tokens=tokens,
        sentence_text=sentence_text,
        events=tuple(events),
        excluded_events=excluded_events,
        skipped_no_anchor=skipped_no_anchor,
        skipped_multi_sentence=skipped_multi_sentence,
    )


def _read_optional_bytes(path: Path) -> bytes | None:
    if not path.exists():
        return None
    return path.read_bytes()


def _label_for_relation(relation: str) -> str | None:
    if relation == "PRECONDITION":
        return "cause"
    if relation == "FALLING_ACTION":
        return "caused_by"
    return None


def _coref_labels(
    first: tuple[int, ...],
    second: tuple[int, ...],
    coref_rows: Sequence[object],
) -> set[str]:
    labels: set[str] = set()
    for row in coref_rows:
        source_members = row.source_members  # type: ignore[attr-defined]
        target_members = row.target_members  # type: ignore[attr-defined]
        relation = row.relation  # type: ignore[attr-defined]
        if first in source_members and second in target_members:
            canonical = _relation_in_order(relation, first, second)
            label = _label_for_relation(canonical)
            if label is not None:
                labels.add(label)
        if first in target_members and second in source_members:
            canonical = _relation_in_order(relation, second, first)
            label = _label_for_relation(canonical)
            if label is not None:
                labels.add(label)
    return labels


def _label_pair(
    first: tuple[int, ...],
    second: tuple[int, ...],
    direct_index: object | None,
    coref_rows: Sequence[object],
) -> tuple[str | None, str | None, bool]:
    """Return ``(label, provenance, ambiguous)`` for one canonical pair."""

    key = _pair_key(first, second)
    if direct_index is not None:
        ambiguous = direct_index.ambiguous  # type: ignore[attr-defined]
        if key in ambiguous:
            return None, None, True
        rows = direct_index.rows  # type: ignore[attr-defined]
        row = rows.get(key)
        if row is not None:
            canonical = _relation_in_order(row.relation, row.source_tokens, row.target_tokens)
            label = _label_for_relation(canonical)
            if label is None:
                return "none", "none", False
            return label, "direct", False

    labels = _coref_labels(first, second, coref_rows)
    if len(labels) > 1:
        return None, None, True
    if labels:
        return next(iter(labels)), "coref", False
    return "none", "none", False


def _record_id(
    doc_name: str,
    sent_in_doc_id: int,
    event1_tokens: tuple[int, ...],
    event2_tokens: tuple[int, ...],
) -> str:
    content = {
        "doc_name": doc_name,
        "sent_in_doc_id": sent_in_doc_id,
        "event1_tokens": list(event1_tokens),
        "event2_tokens": list(event2_tokens),
    }
    canonical = json.dumps(
        content,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return NAMESPACE + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _new_topic_stats() -> dict[str, int]:
    return {"pairs": 0, "causal": 0, "none": 0, "ambiguous": 0}


def build_pool(xml_root: str | Path, eval_root: str | Path) -> PoolResult:
    """Build and label every intra-sentence event pair in the XML corpus."""

    xml_root = Path(xml_root)
    eval_root = Path(eval_root)
    doc_paths = sorted(
        xml_root.rglob("*.xml"),
        key=lambda path: path.relative_to(xml_root).as_posix(),
    )

    pairs: list[PairRecord] = []
    doc_names: list[str] = []
    per_topic: dict[str, dict[str, int]] = {}
    direct_count = 0
    coref_count = 0
    none_count = 0
    causal_count = 0
    ambiguous_count = 0
    enumerated_pairs = 0
    excluded_event_count = 0
    skipped_no_anchor = 0
    skipped_multi_sentence = 0

    for path in doc_paths:
        relative = path.relative_to(xml_root).as_posix()
        doc_names.append(relative)
        document = parse_document(path.read_bytes(), relative)
        topic_stats = per_topic.setdefault(document.topic, _new_topic_stats())
        excluded_event_count += document.excluded_events
        skipped_no_anchor += document.skipped_no_anchor
        skipped_multi_sentence += document.skipped_multi_sentence

        stem = path.name.removesuffix(".xml").removesuffix(".xml")
        direct_payload = _read_optional_bytes(
            eval_root / "event_mentions_extended" / document.topic / f"{stem}.tab"
        )
        coref_payload = _read_optional_bytes(
            eval_root / "coref_chain" / document.topic / f"{stem}.tab"
        )
        direct_index = (
            _parse_official_index(direct_payload, f"{document.topic}/{stem}.tab")
            if direct_payload is not None
            else None
        )
        coref_rows = (
            _parse_coref_rows(coref_payload, f"{document.topic}/{stem}.tab")
            if coref_payload is not None
            else ()
        )

        grouped: dict[int, list[EventMention]] = defaultdict(list)
        for event in document.events:
            grouped[event.sentence].append(event)

        for sentence in sorted(grouped):
            sentence_events = sorted(
                grouped[sentence],
                key=lambda item: (item.tokens[0], item.m_id),
            )
            for left in range(len(sentence_events)):
                for right in range(left + 1, len(sentence_events)):
                    event1 = sentence_events[left]
                    event2 = sentence_events[right]
                    if event1.tokens[0] > event2.tokens[0]:
                        event1, event2 = event2, event1

                    label, provenance, ambiguous = _label_pair(
                        event1.tokens,
                        event2.tokens,
                        direct_index,
                        coref_rows,
                    )
                    enumerated_pairs += 1
                    if ambiguous:
                        ambiguous_count += 1
                        topic_stats["ambiguous"] += 1
                        continue

                    assert label is not None
                    assert provenance is not None
                    if label == "none":
                        none_count += 1
                        topic_stats["none"] += 1
                    else:
                        causal_count += 1
                        topic_stats["causal"] += 1
                        if provenance == "direct":
                            direct_count += 1
                        else:
                            coref_count += 1

                    pairs.append(
                        PairRecord(
                            record_id=_record_id(
                                document.doc_name,
                                sentence,
                                event1.tokens,
                                event2.tokens,
                            ),
                            doc_name=document.doc_name,
                            topic=document.topic,
                            sent_in_doc_id=sentence,
                            sentence=document.sentence_text[sentence],
                            event1=event1.text,
                            event2=event2.text,
                            event1_tokens=event1.tokens,
                            event2_tokens=event2.tokens,
                            label=label,
                            provenance=provenance,
                            relation=_relation_for_label(label),
                        )
                    )
                    topic_stats["pairs"] += 1

    summary = MappingProxyType(
        {
            "total_pairs": len(pairs),
            "enumerated_pairs": enumerated_pairs,
            "causal_count": causal_count,
            "none_count": none_count,
            "direct_count": direct_count,
            "coref_count": coref_count,
            "ambiguous_count": ambiguous_count,
            "excluded_event_count": excluded_event_count,
            "docs_processed": len(doc_names),
            "skipped_no_anchor_count": skipped_no_anchor,
            "skipped_multi_sentence_count": skipped_multi_sentence,
            "per_topic_counts": MappingProxyType(
                {topic: MappingProxyType(dict(stats)) for topic, stats in sorted(per_topic.items())}
            ),
        }
    )
    return PoolResult(
        pairs=tuple(pairs),
        doc_names=tuple(doc_names),
        summary=summary,
    )


def _relation_for_label(label: str) -> str | None:
    if label == "cause":
        return "PRECONDITION"
    if label == "caused_by":
        return "FALLING_ACTION"
    return None


def published_count_check(
    summary: Mapping[str, object],
) -> Mapping[str, object]:
    """Compare the built pool against the published reference counts."""

    total = int(summary["total_pairs"])  # type: ignore[call-overload]
    causal = int(summary["causal_count"])  # type: ignore[call-overload]
    total_low = TARGET_TOTAL_PAIRS * (1.0 - TOTAL_TOLERANCE)
    total_high = TARGET_TOTAL_PAIRS * (1.0 + TOTAL_TOLERANCE)
    causal_low = TARGET_CAUSAL_PAIRS * (1.0 - CAUSAL_TOLERANCE)
    causal_high = TARGET_CAUSAL_PAIRS * (1.0 + CAUSAL_TOLERANCE)
    return MappingProxyType(
        {
            "target_total_pairs": TARGET_TOTAL_PAIRS,
            "total_tolerance": TOTAL_TOLERANCE,
            "total_bounds": [total_low, total_high],
            "total_passes": total_low <= total <= total_high,
            "target_causal_pairs": TARGET_CAUSAL_PAIRS,
            "causal_tolerance": CAUSAL_TOLERANCE,
            "causal_bounds": [causal_low, causal_high],
            "causal_passes": causal_low <= causal <= causal_high,
        }
    )


def _gao_partition(
    doc_names: Sequence[str],
    *,
    dev_topics: frozenset[str],
    seed: int,
    n_folds: int,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Return ``(dev_docs, cv_docs, chunks)`` for the frozen Gao split."""

    dev = sorted(name for name in doc_names if _topic_of(name) in dev_topics)
    cv = sorted(name for name in doc_names if _topic_of(name) not in dev_topics)
    random.Random(seed).shuffle(cv)
    chunks = _contiguous_chunks(cv, n_folds)
    return tuple(dev), tuple(cv), chunks


def assign_gao_fold(
    doc_names: Sequence[str],
    fold: int = 0,
    *,
    dev_topics: frozenset[str] = DEV_TOPICS,
    seed: int = SPLIT_SEED,
    n_folds: int = N_FOLDS,
) -> GaoSplit:
    """Assign one Gao/iLIF fold using the shared document-level chunking.

    The chunking (and therefore fold 0) is unchanged from the original split:

    1. Documents whose topic is in ``dev_topics`` (37 and 41) form ``dev``.
    2. Every other topic forms the cross-validation pool ``cv``.
    3. ``cv`` document names are sorted lexicographically.
    4. ``random.Random(209).shuffle`` is applied to the sorted list.
    5. The shuffled list is cut into five *contiguous* chunks, distributing any
       remainder to the earlier chunks.
    6. Fold ``k`` uses chunk ``k`` as ``test`` and the other four chunks, in
       chunk order, as ``train``.
    """

    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in [0, {n_folds}), got {fold}")
    dev, cv, chunks = _gao_partition(
        doc_names,
        dev_topics=dev_topics,
        seed=seed,
        n_folds=n_folds,
    )
    train = tuple(name for index, chunk in enumerate(chunks) if index != fold for name in chunk)
    return GaoSplit(
        fold=fold,
        train_docs=train,
        test_docs=chunks[fold],
        dev_docs=dev,
        cv_docs=cv,
        chunks=chunks,
    )


def split_documents(
    doc_names: Sequence[str],
    *,
    dev_topics: frozenset[str] = DEV_TOPICS,
    seed: int = SPLIT_SEED,
    n_folds: int = N_FOLDS,
) -> SplitResult:
    """Reproduce the Gao/iLIF fold-0 split exactly (legacy wrapper)."""

    split = assign_gao_fold(
        doc_names,
        0,
        dev_topics=dev_topics,
        seed=seed,
        n_folds=n_folds,
    )
    return SplitResult(
        train_docs=split.train_docs,
        test_docs=split.test_docs,
        dev_docs=split.dev_docs,
        cv_docs=split.cv_docs,
        chunks=split.chunks,
    )


def _topic_of(doc_name: str) -> str:
    return doc_name.split("/", 1)[0]


def _contiguous_chunks(items: Sequence[str], n_folds: int) -> tuple[tuple[str, ...], ...]:
    size, remainder = divmod(len(items), n_folds)
    chunks: list[tuple[str, ...]] = []
    start = 0
    for index in range(n_folds):
        end = start + size + (1 if index < remainder else 0)
        chunks.append(tuple(items[start:end]))
        start = end
    return tuple(chunks)


def build_datasets(pool: PoolResult, split: SplitResult | GaoSplit) -> DatasetResult:
    """Partition pair records into train/test and dev sets for a split."""

    def select(doc_names: Sequence[str]) -> tuple[PairRecord, ...]:
        wanted = set(doc_names)
        return tuple(pair for pair in pool.pairs if pair.doc_name in wanted)

    return DatasetResult(
        train=select(split.train_docs),
        test=select(split.test_docs),
        dev=select(split.dev_docs),
    )


def balanced_train_pairs(
    pool: PoolResult,
    split: GaoSplit,
    *,
    seed: int,
) -> tuple[PairRecord, ...]:
    """Return fold training rows balanced at 1 causal : 2 none.

    The deterministic procedure is frozen to reproduce the historical fold-0
    ``train-balanced.jsonl`` exactly:

    1. Collect all causal pairs and all none pairs from the train documents in
       pool order (no separate sorting).
    2. ``rng = random.Random(seed)``.
    3. ``rng.shuffle(none_pairs)`` in place.
    4. Keep the first ``2 * len(causal_pairs)`` none pairs.
    5. Concatenate ``causal_pairs + sampled_none``.
    6. ``rng.shuffle(combined)`` with the same RNG stream.

    Fold 0 uses ``seed=209``. Folds 1-4 use ``seed=209 + fold``.
    """

    train_docs = set(split.train_docs)
    causal = tuple(
        pair for pair in pool.pairs if pair.doc_name in train_docs and pair.label != "none"
    )
    none = [pair for pair in pool.pairs if pair.doc_name in train_docs and pair.label == "none"]
    needed = 2 * len(causal)
    if len(none) < needed:
        raise ValueError(
            f"fold {split.fold} has {len(none)} none pairs but needs {needed} "
            f"to balance {len(causal)} causal pairs at 1:2",
        )
    rng = random.Random(seed)
    rng.shuffle(none)
    combined = list(causal) + none[:needed]
    rng.shuffle(combined)
    return tuple(combined)


def _select_pairs(pool: PoolResult, doc_names: Sequence[str]) -> tuple[PairRecord, ...]:
    wanted = set(doc_names)
    return tuple(pair for pair in pool.pairs if pair.doc_name in wanted)


def plan_gao_folds(pool: PoolResult) -> tuple[FoldPlan, ...]:
    """Materialise all five folds with seed ``209 + fold`` for none sampling."""

    plans: list[FoldPlan] = []
    for fold in range(N_FOLDS):
        split = assign_gao_fold(pool.doc_names, fold)
        sample_seed = SPLIT_SEED + fold
        plans.append(
            FoldPlan(
                split=split,
                train_pairs=balanced_train_pairs(pool, split, seed=sample_seed),
                test_pairs=_select_pairs(pool, split.test_docs),
                sample_seed=sample_seed,
            )
        )
    return tuple(plans)


def laya_row(pair: PairRecord) -> dict[str, object]:
    """Build one Laya fine-tune row for a pair."""

    request = build_laya_request(pair.sentence, pair.event1, pair.event2)
    row: dict[str, object] = {
        "record_id": pair.record_id,
        "state": request.state,
        "questions": request.questions,
        "expected": {"eci_relation": pair.label},
    }
    validate_laya_row(row)
    return row


def validate_laya_row(row: dict[str, object]) -> None:
    """Enforce the exact state/questions schema required by Laya training."""

    required = {"state", "questions", "expected"}
    allowed = required | {"record_id"}
    if not required <= set(row) or not set(row) <= allowed:
        raise ValueError(f"unexpected Laya row keys: {sorted(row)}")
    record_id = row.get("record_id")
    if record_id is not None and not isinstance(record_id, str):
        raise ValueError("Laya record_id must be a string")
    state = row["state"]
    if not isinstance(state, dict) or set(state) != STATE_KEYS:
        raise ValueError("Laya state must have exactly sentence/event1/event2")
    questions = row["questions"]
    if not isinstance(questions, dict) or set(questions) != {"eci_relation"}:
        raise ValueError("Laya questions must contain exactly eci_relation")
    question = questions["eci_relation"]
    if not isinstance(question, dict):
        raise ValueError("eci_relation question must be a mapping")
    if question.get("type") != "choice":
        raise ValueError("eci_relation question must be a choice question")
    criteria = question.get("criteria")
    if not isinstance(criteria, dict) or set(criteria) != CHOICE_KEYS:
        raise ValueError("eci_relation criteria must contain cause/caused_by/none")
    expected = row["expected"]
    if not isinstance(expected, dict) or set(expected) != {"eci_relation"}:
        raise ValueError("expected must be a mapping keyed by eci_relation")
    if expected["eci_relation"] not in EXPECTED_VALUES:
        raise ValueError(f"unexpected expected label: {expected['eci_relation']!r}")


def _json_line(payload: object) -> str:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.write_text("".join(_json_line(row) for row in rows), encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary_payload(summary: Mapping[str, object]) -> dict[str, object]:
    """Return a fully JSON-serializable copy of the summary counters."""

    payload = dict(summary)
    per_topic = payload.get("per_topic_counts")
    if isinstance(per_topic, Mapping):
        payload["per_topic_counts"] = {
            str(topic): dict(stats) for topic, stats in per_topic.items()
        }
    return payload


def write_outputs(
    pool: PoolResult,
    split: SplitResult,
    datasets: DatasetResult,
    check: Mapping[str, object],
    artifacts_dir: str | Path,
    experiments_dir: str | Path,
) -> dict[str, Path]:
    """Write pair artifacts and fold-0 Laya JSONL files deterministically."""

    artifacts_path = Path(artifacts_dir)
    experiments_path = Path(experiments_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    experiments_path.mkdir(parents=True, exist_ok=True)

    pairs_path = artifacts_path / "pairs.jsonl"
    _write_jsonl(pairs_path, [pair.payload() for pair in pool.pairs])

    train_path = experiments_path / "train.jsonl"
    test_path = experiments_path / "test.jsonl"
    dev_path = experiments_path / "dev.jsonl"
    test_pairs_path = experiments_path / "test_pairs.jsonl"
    dev_pairs_path = experiments_path / "dev_pairs.jsonl"
    _write_jsonl(train_path, [laya_row(pair) for pair in datasets.train])
    _write_jsonl(test_path, [laya_row(pair) for pair in datasets.test])
    _write_jsonl(dev_path, [laya_row(pair) for pair in datasets.dev])
    _write_jsonl(
        test_pairs_path,
        [{**pair.payload(), "split": "test"} for pair in datasets.test],
    )
    _write_jsonl(
        dev_pairs_path,
        [{**pair.payload(), "split": "dev"} for pair in datasets.dev],
    )

    summary_payload = {
        **_summary_payload(pool.summary),
        "published_target_check": dict(check),
    }
    summary_path = artifacts_path / "summary.json"
    _write_json(summary_path, summary_payload)

    outputs = {
        "pairs": pairs_path,
        "train": train_path,
        "test": test_path,
        "dev": dev_path,
        "test_pairs": test_pairs_path,
        "dev_pairs": dev_pairs_path,
        "summary": summary_path,
    }
    manifest_payload = {
        "schema_version": "1",
        "namespace": NAMESPACE,
        "documents_processed": len(pool.doc_names),
        "summary": _summary_payload(pool.summary),
        "published_target_check": dict(check),
        "fold0_split": {
            "algorithm": (
                "dev topics 37,41; sort CV doc names; "
                "random.Random(209).shuffle; five contiguous chunks; "
                "test=chunk0; train=chunks1-4"
            ),
            "seed": SPLIT_SEED,
            "n_folds": N_FOLDS,
            "dev_topics": sorted(DEV_TOPICS),
            "train_docs": list(split.train_docs),
            "test_docs": list(split.test_docs),
            "dev_docs": list(split.dev_docs),
        },
        "outputs": {
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in outputs.items()
        },
    }
    manifest_path = artifacts_path / "manifest.json"
    _write_json(manifest_path, manifest_payload)
    outputs["manifest"] = manifest_path
    return outputs


def verify_fold0_balance(plan: FoldPlan, reference: str | Path) -> None:
    """Raise when generated fold-0 rows differ from the existing balanced file."""

    reference_path = Path(reference)
    if not reference_path.exists():
        return
    expected = [
        json.loads(line)
        for line in reference_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    generated = [laya_row(pair) for pair in plan.train_pairs]
    if len(expected) != len(generated):
        raise Fold0BalanceMismatchError(
            reference_path,
            f"row count differs: existing={len(expected)} generated={len(generated)}",
        )
    for index, (expected_row, generated_row) in enumerate(zip(expected, generated, strict=True)):
        expected_id = expected_row.get("record_id")
        generated_id = generated_row.get("record_id")
        if expected_id != generated_id:
            raise Fold0BalanceMismatchError(
                reference_path,
                f"record_id mismatch at row {index}: "
                f"existing={expected_id!r} generated={generated_id!r}",
            )
    for index, (expected_row, generated_row) in enumerate(zip(expected, generated, strict=True)):
        if expected_row != generated_row:
            raise Fold0BalanceMismatchError(
                reference_path,
                f"row content mismatch at row {index} (record_id={generated_row['record_id']!r})",
            )


def _causal_count(pairs: Sequence[PairRecord]) -> int:
    return sum(1 for pair in pairs if pair.label != "none")


def write_fold_outputs(
    plan: FoldPlan,
    experiments_dir: str | Path,
) -> dict[str, object]:
    """Write one fold's balanced train, test, test metadata and manifest."""

    fold = plan.split.fold
    fold_dir = Path(experiments_dir) / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_path = fold_dir / "train.jsonl"
    test_path = fold_dir / "test.jsonl"
    test_pairs_path = fold_dir / "test_pairs.jsonl"
    manifest_path = fold_dir / "manifest.json"

    _write_jsonl(train_path, [laya_row(pair) for pair in plan.train_pairs])
    _write_jsonl(test_path, [laya_row(pair) for pair in plan.test_pairs])
    _write_jsonl(
        test_pairs_path,
        [{**pair.payload(), "split": "test"} for pair in plan.test_pairs],
    )

    train_causal = _causal_count(plan.train_pairs)
    test_causal = _causal_count(plan.test_pairs)
    counts = {
        "train_pairs": len(plan.train_pairs),
        "train_causal": train_causal,
        "train_none": len(plan.train_pairs) - train_causal,
        "test_pairs": len(plan.test_pairs),
        "test_causal": test_causal,
        "test_none": len(plan.test_pairs) - test_causal,
    }
    paths = {
        "train": str(train_path),
        "test": str(test_path),
        "test_pairs": str(test_pairs_path),
        "manifest": str(manifest_path),
    }
    manifest = {
        "fold": fold,
        "split_seed": SPLIT_SEED,
        "sample_seed": plan.sample_seed,
        "n_folds": N_FOLDS,
        "docs": {
            "train_count": len(plan.split.train_docs),
            "test_count": len(plan.split.test_docs),
            "dev_count": len(plan.split.dev_docs),
            "train": list(plan.split.train_docs),
            "test": list(plan.split.test_docs),
            "dev": list(plan.split.dev_docs),
        },
        "counts": counts,
        "paths": paths,
        "algorithm": (
            "sorted CV docs -> random.Random(209).shuffle -> five contiguous "
            "chunks; fold k test=chunk k; train=other chunks; balanced train = "
            "all causal train pairs + 2x same none pairs sampled with "
            "random.Random(209 + k) then shuffled"
        ),
    }
    _write_json(manifest_path, manifest)
    return {
        "fold": fold,
        "split_seed": SPLIT_SEED,
        "sample_seed": plan.sample_seed,
        "test_docs": list(plan.split.test_docs),
        "train_docs": list(plan.split.train_docs),
        "counts": counts,
        "paths": paths,
    }


def write_folds_index(
    entries: Sequence[Mapping[str, object]],
    experiments_dir: str | Path,
    *,
    cv_doc_count: int,
) -> Path:
    """Write ``folds.json`` summarising all five folds."""

    path = Path(experiments_dir) / "folds.json"
    payload = {
        "split_seed": SPLIT_SEED,
        "n_folds": N_FOLDS,
        "cv_doc_count": cv_doc_count,
        "folds": {str(entry["fold"]): dict(entry) for entry in entries},
    }
    _write_json(path, payload)
    return path


def run_build(
    xml_root: str | Path = DEFAULT_XML_ROOT,
    eval_root: str | Path = DEFAULT_EVAL_ROOT,
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    experiments_dir: str | Path = DEFAULT_EXPERIMENTS_DIR,
    *,
    enforce_counts: bool = True,
    fold0_reference: str | Path | None = None,
) -> dict[str, Path]:
    """Build the pool, enforce counts, verify fold 0, then write all folds."""

    pool = build_pool(xml_root, eval_root)
    check = published_count_check(pool.summary)
    if enforce_counts and not (check["total_passes"] and check["causal_passes"]):
        raise CountMismatchError(pool.summary, check)

    plans = plan_gao_folds(pool)
    reference = (
        Path(fold0_reference)
        if fold0_reference is not None
        else Path(experiments_dir) / "train-balanced.jsonl"
    )
    verify_fold0_balance(plans[0], reference)

    split = split_documents(pool.doc_names)
    datasets = build_datasets(pool, split)
    outputs = write_outputs(
        pool,
        split,
        datasets,
        check,
        artifacts_dir,
        experiments_dir,
    )

    entries = [write_fold_outputs(plan, experiments_dir) for plan in plans]
    outputs["folds"] = write_folds_index(
        entries,
        experiments_dir,
        cv_doc_count=len(plans[0].split.cv_docs),
    )
    for entry in entries:
        fold = int(entry["fold"])  # type: ignore[call-overload]
        paths = entry["paths"]
        assert isinstance(paths, Mapping)
        outputs[f"fold{fold}_train"] = Path(str(paths["train"]))
        outputs[f"fold{fold}_test"] = Path(str(paths["test"]))
        outputs[f"fold{fold}_test_pairs"] = Path(str(paths["test_pairs"]))
        outputs[f"fold{fold}_manifest"] = Path(str(paths["manifest"]))
    return outputs


def _print_breakdown(
    summary: Mapping[str, object],
    check: Mapping[str, object],
) -> None:
    print("published count check FAILED", file=sys.stderr)
    print(f"total_pairs={summary['total_pairs']} check={dict(check)}", file=sys.stderr)
    print(f"causal_count={summary['causal_count']}", file=sys.stderr)
    print(f"enumerated_pairs={summary['enumerated_pairs']}", file=sys.stderr)
    print(f"ambiguous_count={summary['ambiguous_count']}", file=sys.stderr)
    print(f"excluded_event_count={summary['excluded_event_count']}", file=sys.stderr)
    print(
        f"direct_count={summary['direct_count']} "
        f"coref_count={summary['coref_count']} none_count={summary['none_count']}",
        file=sys.stderr,
    )
    per_topic = summary["per_topic_counts"]
    for topic, stats in per_topic.items():  # type: ignore[attr-defined]
        print(f"topic={topic} {dict(stats)}", file=sys.stderr)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the EventStoryLine intra-sentence pool and all five Laya folds.",
    )
    parser.add_argument("--xml-root", default=str(DEFAULT_XML_ROOT))
    parser.add_argument("--eval-root", default=str(DEFAULT_EVAL_ROOT))
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--experiments-dir", default=str(DEFAULT_EXPERIMENTS_DIR))
    parser.add_argument(
        "--no-enforce-counts",
        action="store_true",
        help="write outputs even when published reference counts are missed",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        outputs = run_build(
            args.xml_root,
            args.eval_root,
            args.artifacts_dir,
            args.experiments_dir,
            enforce_counts=not args.no_enforce_counts,
        )
    except CountMismatchError as exc:
        _print_breakdown(exc.summary, exc.check)
        return 1
    except Fold0BalanceMismatchError as exc:
        print(f"fold-0 balanced mismatch: {exc.detail}", file=sys.stderr)
        return 1

    summary_path = outputs["summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in (
        "docs_processed",
        "enumerated_pairs",
        "total_pairs",
        "causal_count",
        "none_count",
        "direct_count",
        "coref_count",
        "ambiguous_count",
        "excluded_event_count",
        "skipped_no_anchor_count",
        "skipped_multi_sentence_count",
    ):
        print(f"{key}={summary[key]}")
    for name, path in outputs.items():
        print(f"{name}={path}")

    folds_payload = json.loads(outputs["folds"].read_text(encoding="utf-8"))
    for fold in sorted(folds_payload["folds"], key=int):
        entry = folds_payload["folds"][fold]
        counts = entry["counts"]
        print(
            f"fold{fold} "
            f"train_docs={len(entry['train_docs'])} "
            f"test_docs={len(entry['test_docs'])} "
            f"test_pairs={counts['test_pairs']} "
            f"test_causal={counts['test_causal']} "
            f"train_pairs={counts['train_pairs']} "
            f"train_causal={counts['train_causal']} "
            f"train_none={counts['train_none']} "
            f"sample_seed={entry['sample_seed']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
