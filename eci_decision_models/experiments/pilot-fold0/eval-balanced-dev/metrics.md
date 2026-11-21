# Fine-tuned Laya evaluation: pilot-balanced-dev

## Coverage

| metric | value |
| --- | --- |
| N | 810 |
| valid | 810 |
| fail | 0 |
| coverage | 1.0000 |
| matched | 810 |
| missing_predictions | 0 |
| extra_predictions | 0 |

## Three-class

| metric | value |
| --- | --- |
| accuracy | 0.6889 |
| macro_f1 | 0.2984 |

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| cause | 0.0441 | 0.0280 | 0.0343 | 107 |
| caused_by | 0.0690 | 0.0317 | 0.0435 | 63 |
| none | 0.7756 | 0.8641 | 0.8174 | 640 |

## Binary existence

| metric | value |
| --- | --- |
| precision | 0.1031 |
| recall | 0.0588 |
| f1 | 0.0749 |
| accuracy | 0.6951 |

## Strict direction (iLIF-style)

| metric | value |
| --- | --- |
| tp | 5 |
| fp | 87 |
| fn | 165 |
| tn | 553 |
| precision | 0.0543 |
| recall | 0.0294 |
| f1 | 0.0382 |
| n | 810 |

## Status

| status | count |
| --- | --- |
| ok | 810 |
