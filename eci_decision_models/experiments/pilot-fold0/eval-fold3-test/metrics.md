# Fine-tuned Laya evaluation: fold3-test

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
| accuracy | 0.7782 |
| macro_f1 | 0.6041 |

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| cause | 0.4750 | 0.5135 | 0.4935 | 111 |
| caused_by | 0.4140 | 0.4924 | 0.4498 | 132 |
| none | 0.8859 | 0.8525 | 0.8689 | 902 |

## Binary existence

| metric | value |
| --- | --- |
| precision | 0.5199 |
| recall | 0.5926 |
| f1 | 0.5538 |
| accuracy | 0.7974 |

## Strict direction (iLIF-style)

| metric | value |
| --- | --- |
| tp | 122 |
| fp | 133 |
| fn | 121 |
| tn | 769 |
| precision | 0.4784 |
| recall | 0.5021 |
| f1 | 0.4900 |
| n | 1145 |

## Status

| status | count |
| --- | --- |
| ok | 1145 |
