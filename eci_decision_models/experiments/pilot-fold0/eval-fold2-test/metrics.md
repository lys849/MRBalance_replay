# Fine-tuned Laya evaluation: fold2-test

## Coverage

| metric | value |
| --- | --- |
| N | 1312 |
| valid | 1312 |
| fail | 0 |
| coverage | 1.0000 |
| matched | 1312 |
| missing_predictions | 0 |
| extra_predictions | 0 |

## Three-class

| metric | value |
| --- | --- |
| accuracy | 0.7896 |
| macro_f1 | 0.6193 |

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| cause | 0.4964 | 0.4859 | 0.4911 | 142 |
| caused_by | 0.4750 | 0.5067 | 0.4903 | 150 |
| none | 0.8796 | 0.8735 | 0.8765 | 1020 |

## Binary existence

| metric | value |
| --- | --- |
| precision | 0.5686 |
| recall | 0.5822 |
| f1 | 0.5753 |
| accuracy | 0.8087 |

## Strict direction (iLIF-style)

| metric | value |
| --- | --- |
| tp | 145 |
| fp | 129 |
| fn | 147 |
| tn | 891 |
| precision | 0.5292 |
| recall | 0.4966 |
| f1 | 0.5124 |
| n | 1312 |

## Status

| status | count |
| --- | --- |
| ok | 1312 |
