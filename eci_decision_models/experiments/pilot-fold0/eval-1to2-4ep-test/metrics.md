# Fine-tuned Laya evaluation: pilot-1to2-4ep-test

## Coverage

| metric | value |
| --- | --- |
| N | 1645 |
| valid | 1645 |
| fail | 0 |
| coverage | 1.0000 |
| matched | 1645 |
| missing_predictions | 0 |
| extra_predictions | 0 |

## Three-class

| metric | value |
| --- | --- |
| accuracy | 0.7635 |
| macro_f1 | 0.5840 |

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| cause | 0.4515 | 0.4844 | 0.4673 | 192 |
| caused_by | 0.3907 | 0.4492 | 0.4179 | 187 |
| none | 0.8815 | 0.8523 | 0.8667 | 1266 |

## Binary existence

| metric | value |
| --- | --- |
| precision | 0.5558 |
| recall | 0.6174 |
| f1 | 0.5850 |
| accuracy | 0.7982 |

## Strict direction (iLIF-style)

| metric | value |
| --- | --- |
| tp | 177 |
| fp | 187 |
| fn | 202 |
| tn | 1079 |
| precision | 0.4863 |
| recall | 0.4670 |
| f1 | 0.4764 |
| n | 1645 |

## Status

| status | count |
| --- | --- |
| ok | 1645 |
