# Fine-tuned Laya evaluation: base-fold1-test

## Coverage

| metric | value |
| --- | --- |
| N | 1408 |
| valid | 1408 |
| fail | 0 |
| coverage | 1.0000 |
| matched | 1408 |
| missing_predictions | 0 |
| extra_predictions | 0 |

## Three-class

| metric | value |
| --- | --- |
| accuracy | 0.2713 |
| macro_f1 | 0.2217 |

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| cause | 0.1163 | 0.0787 | 0.0939 | 127 |
| caused_by | 0.1585 | 0.8431 | 0.2669 | 204 |
| none | 0.8439 | 0.1857 | 0.3044 | 1077 |

## Binary existence

| metric | value |
| --- | --- |
| precision | 0.2511 |
| recall | 0.8882 |
| f1 | 0.3915 |
| accuracy | 0.3509 |

## Strict direction (iLIF-style)

| metric | value |
| --- | --- |
| tp | 182 |
| fp | 877 |
| fn | 149 |
| tn | 200 |
| precision | 0.1719 |
| recall | 0.5498 |
| f1 | 0.2619 |
| n | 1408 |

## Status

| status | count |
| --- | --- |
| ok | 1408 |
