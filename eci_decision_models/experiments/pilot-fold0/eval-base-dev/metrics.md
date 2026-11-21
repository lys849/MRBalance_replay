# Fine-tuned Laya evaluation: pilot-base-dev

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
| accuracy | 0.1852 |
| macro_f1 | 0.1854 |

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| cause | 0.1603 | 0.2336 | 0.1901 | 107 |
| caused_by | 0.0903 | 0.8254 | 0.1628 | 63 |
| none | 0.9359 | 0.1141 | 0.2033 | 640 |

## Binary existence

| metric | value |
| --- | --- |
| precision | 0.2254 |
| recall | 0.9706 |
| f1 | 0.3659 |
| accuracy | 0.2938 |

## Strict direction (iLIF-style)

| metric | value |
| --- | --- |
| tp | 77 |
| fp | 567 |
| fn | 93 |
| tn | 73 |
| precision | 0.1196 |
| recall | 0.4529 |
| f1 | 0.1892 |
| n | 810 |

## Status

| status | count |
| --- | --- |
| ok | 810 |
