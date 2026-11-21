# Fine-tuned Laya evaluation: fold4-test

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
| accuracy | 0.7650 |
| macro_f1 | 0.5848 |

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| cause | 0.3649 | 0.4426 | 0.4000 | 122 |
| caused_by | 0.4440 | 0.5468 | 0.4901 | 203 |
| none | 0.8933 | 0.8371 | 0.8643 | 1160 |

## Binary existence

| metric | value |
| --- | --- |
| precision | 0.5251 |
| recall | 0.6431 |
| f1 | 0.5781 |
| accuracy | 0.7946 |

## Strict direction (iLIF-style)

| metric | value |
| --- | --- |
| tp | 165 |
| fp | 189 |
| fn | 160 |
| tn | 971 |
| precision | 0.4661 |
| recall | 0.5077 |
| f1 | 0.4860 |
| n | 1485 |

## Status

| status | count |
| --- | --- |
| ok | 1485 |
