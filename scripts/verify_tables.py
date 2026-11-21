"""Verify reported aggregate tables without model calls or source corpora."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


for path in sorted((ROOT / "experiments").glob("*/metrics.json")):
    row = read(path)["all"]
    precision, recall = row["precision"], row["recall"]
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    assert abs(f1 - row["f1"]) < 1e-12
    print(f"{path.parent.name:24s} N={row['n_samples']} F1={100 * f1:.2f}")

base = ROOT / "eci_decision_models/experiments/pilot-fold0"
for label, names in [
    ("base", ["eval-base-test"] + [f"eval-base-fold{k}-test" for k in range(1, 5)]),
    ("finetuned", ["eval-1to2-4ep-test"] + [f"eval-fold{k}-test" for k in range(1, 5)]),
]:
    values = []
    for name in names:
        classification = read(base / name / "metrics.json")["classification"]
        matrix = classification["confusion_matrix"]
        scores = []
        for index, choice in enumerate(["cause", "caused_by", "none"]):
            tp = matrix[choice][index]
            denominator = sum(matrix[choice]) + sum(row[index] for row in matrix.values())
            scores.append(2 * tp / denominator if denominator else 0)
        macro = statistics.mean(scores)
        assert abs(macro - classification["macro_f1"]) < 1e-12
        values.append(100 * macro)
    print(f"Laya {label}: folds={values}, "
          f"mean={statistics.mean(values):.2f}, SD={statistics.stdev(values):.2f}")
