# Fine-tuned Laya evaluation: fold1-test

## Coverage

| metric | value |
| --- | --- |
| N | 1408 |
| valid | 1408 |
| fail | 0 |
| coverage | 1.0000 |
| matched | 1408 |
| missing_predictions | 0 |
| extra_predictions | 0 |

## Three-class

| metric | value |
| --- | --- |
| accuracy | 0.7337 |
| macro_f1 | 0.5533 |

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| cause | 0.3191 | 0.3543 | 0.3358 | 127 |
| caused_by | 0.4370 | 0.5441 | 0.4847 | 204 |
| none | 0.8657 | 0.8143 | 0.8392 | 1077 |

## Binary existence

| metric | value |
| --- | --- |
| precision | 0.4937 |
| recall | 0.5891 |
| f1 | 0.5372 |
| accuracy | 0.7614 |

## Strict direction (iLIF-style)

| metric | value |
| --- | --- |
| tp | 156 |
| fp | 200 |
| fn | 175 |
| tn | 877 |
| precision | 0.4382 |
| recall | 0.4713 |
| f1 | 0.4541 |
| n | 1408 |

## Status

| status | count |
| --- | --- |
| ok | 1408 |
