# Fine-tuned Laya evaluation: pilot-1to2-2ep-dev

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
| accuracy | 0.6864 |
| macro_f1 | 0.5349 |

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| cause | 0.4106 | 0.5794 | 0.4806 | 107 |
| caused_by | 0.2500 | 0.5079 | 0.3351 | 63 |
| none | 0.8701 | 0.7219 | 0.7891 | 640 |

## Binary existence

| metric | value |
| --- | --- |
| precision | 0.3620 |
| recall | 0.5941 |
| f1 | 0.4499 |
| accuracy | 0.6951 |

## Strict direction (iLIF-style)

| metric | value |
| --- | --- |
| tp | 94 |
| fp | 178 |
| fn | 76 |
| tn | 462 |
| precision | 0.3456 |
| recall | 0.5529 |
| f1 | 0.4253 |
| n | 810 |

## Status

| status | count |
| --- | --- |
| ok | 810 |
