# Fine-tuned Laya evaluation: base-fold3-test

## Coverage

| metric | value |
| --- | --- |
| N | 1145 |
| valid | 1145 |
| fail | 0 |
| coverage | 1.0000 |
| matched | 1145 |
| missing_predictions | 0 |
| extra_predictions | 0 |

## Three-class

| metric | value |
| --- | --- |
| accuracy | 0.2533 |
| macro_f1 | 0.2296 |

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| cause | 0.1562 | 0.1802 | 0.1674 | 111 |
| caused_by | 0.1285 | 0.8030 | 0.2215 | 132 |
| none | 0.8542 | 0.1818 | 0.2998 | 902 |

## Binary existence

| metric | value |
| --- | --- |
| precision | 0.2256 |
| recall | 0.8848 |
| f1 | 0.3595 |
| accuracy | 0.3310 |

## Strict direction (iLIF-style)

| metric | value |
| --- | --- |
| tp | 126 |
| fp | 738 |
| fn | 117 |
| tn | 164 |
| precision | 0.1458 |
| recall | 0.5185 |
| f1 | 0.2276 |
| n | 1145 |

## Status

| status | count |
| --- | --- |
| ok | 1145 |
