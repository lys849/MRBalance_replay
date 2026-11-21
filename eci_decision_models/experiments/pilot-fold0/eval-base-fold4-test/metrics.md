# Fine-tuned Laya evaluation: base-fold4-test

## Coverage

| metric | value |
| --- | --- |
| N | 1485 |
| valid | 1485 |
| fail | 0 |
| coverage | 1.0000 |
| matched | 1485 |
| missing_predictions | 0 |
| extra_predictions | 0 |

## Three-class

| metric | value |
| --- | --- |
| accuracy | 0.2411 |
| macro_f1 | 0.2094 |

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| cause | 0.1176 | 0.1311 | 0.1240 | 122 |
| caused_by | 0.1539 | 0.8818 | 0.2621 | 203 |
| none | 0.8763 | 0.1405 | 0.2422 | 1160 |

## Binary existence

| metric | value |
| --- | --- |
| precision | 0.2325 |
| recall | 0.9292 |
| f1 | 0.3719 |
| accuracy | 0.3131 |

## Strict direction (iLIF-style)

| metric | value |
| --- | --- |
| tp | 195 |
| fp | 997 |
| fn | 130 |
| tn | 163 |
| precision | 0.1636 |
| recall | 0.6000 |
| f1 | 0.2571 |
| n | 1485 |

## Status

| status | count |
| --- | --- |
| ok | 1485 |
