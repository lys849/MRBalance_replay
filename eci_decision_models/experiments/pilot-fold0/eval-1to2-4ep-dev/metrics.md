# Fine-tuned Laya evaluation: pilot-1to2-4ep-dev

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
| accuracy | 0.7185 |
| macro_f1 | 0.5636 |

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| cause | 0.4453 | 0.5327 | 0.4851 | 107 |
| caused_by | 0.2975 | 0.5714 | 0.3913 | 63 |
| none | 0.8717 | 0.7641 | 0.8143 | 640 |

## Binary existence

| metric | value |
| --- | --- |
| precision | 0.3936 |
| recall | 0.5765 |
| f1 | 0.4678 |
| accuracy | 0.7247 |

## Strict direction (iLIF-style)

| metric | value |
| --- | --- |
| tp | 93 |
| fp | 151 |
| fn | 77 |
| tn | 489 |
| precision | 0.3811 |
| recall | 0.5471 |
| f1 | 0.4493 |
| n | 810 |

## Status

| status | count |
| --- | --- |
| ok | 810 |
