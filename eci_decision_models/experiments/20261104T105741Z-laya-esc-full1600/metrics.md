# ECI Evaluation Report

- run_id: 20261104T105741Z-laya-esc-full1600
- model: convaiinnovations/laya-typed-decisions
- data path: ../reference/MRBalance/data/test/ESC_intra_data.json
- data sha256: feff6575298d5268341e603f33e806ba87e1e82d35c24ce0135ccb0a3f335e0c
- gold path: artifacts/gold_choice_reconstructed/v1/gold_choice_reconstructed.jsonl
- gold sha256: f4689e54fa176c2b05b82adecf2a1549775280e1df38f132498a0aa4f5e4662a
- gold version: gold_choice_reconstructed/v1

## Caveats

- Laya-only evaluation: only the Laya model was run; Jev was not evaluated.
- Zero-shot evaluation: no task-specific fine-tuning was applied.
- The canonical prompt and option order were fixed for every item; no prompt or ordering variants were sampled.
- No extra baselines were run; results describe the single model alone.
- The `none` label means ESC was left unlabeled, not that the absence of a causal relation was proven.
- The Phase 2 gate remains open; this pilot does not close it.
- These results are not a Jev/Laya comparison.

## Classification

| Metric | Value |
| --- | --- |
| N | 1600 |
| valid | 1600 |
| fail | 0 |
| coverage | 1.0000 |
| accuracy | 0.2444 |
| macro-F1 | 0.2090 |

## Confusion Matrix

| Gold \ Predicted | cause | caused_by | none | invalid prediction |
| --- | --- | --- | --- | --- |
| cause | 20 | 211 | 13 | 0 |
| caused_by | 23 | 213 | 20 | 0 |
| none | 112 | 830 | 158 | 0 |

## Per-class

| Class | Precision | Recall | F1 | Support | Predicted |
| --- | --- | --- | --- | --- | --- |
| cause | 0.1290 | 0.0820 | 0.1003 | 244 | 155 |
| caused_by | 0.1699 | 0.8320 | 0.2821 | 256 | 1254 |
| none | 0.8272 | 0.1436 | 0.2448 | 1100 | 191 |

## Derived Binary

| Metric | Value |
| --- | --- |
| accuracy | 0.3906 |
| precision | 0.3314 |
| recall | 0.9340 |
| F1 | 0.4893 |
| positive recall | 0.9340 |
| negative recall | 0.1436 |

## Probability

| Metric | Value |
| --- | --- |
| Brier | 0.8043 |
| NLL | 1.3650 |
| zero policy | error |
| NLL epsilon | N/A |
| ECE | 0.2311 |
| valid coverage | 1.0000 |
| valid | 1600 |
| N | 1600 |

| Lower | Upper | Count | Accuracy | Confidence | Contribution |
| --- | --- | --- | --- | --- | --- |
| 0.0000 | 0.1000 | 0 | N/A | N/A | 0.0000 |
| 0.1000 | 0.2000 | 0 | N/A | N/A | 0.0000 |
| 0.2000 | 0.3000 | 0 | N/A | N/A | 0.0000 |
| 0.3000 | 0.4000 | 255 | 0.3137 | 0.3785 | 0.0103 |
| 0.4000 | 0.5000 | 817 | 0.2056 | 0.4505 | 0.1250 |
| 0.5000 | 0.6000 | 419 | 0.2387 | 0.5399 | 0.0789 |
| 0.6000 | 0.7000 | 97 | 0.3711 | 0.6306 | 0.0157 |
| 0.7000 | 0.8000 | 12 | 0.5833 | 0.7310 | 0.0011 |
| 0.8000 | 0.9000 | 0 | N/A | N/A | 0.0000 |
| 0.9000 | 1.0000 | 0 | N/A | N/A | 0.0000 |

## Direction Errors

| Direction error | Count |
| --- | --- |
| gold cause, predicted caused_by | 211 |
| gold caused_by, predicted cause | 23 |
| gold causal, predicted none | 33 |
| gold none, predicted causal | 942 |
| invalid prediction on causal gold | 0 |
| invalid prediction on none gold | 0 |

## Status

### Status counts
| Label | Count |
| --- | --- |
| ok | 1600 |

### Control status counts
| Label | Count |
| --- | --- |
| N/A | N/A |
