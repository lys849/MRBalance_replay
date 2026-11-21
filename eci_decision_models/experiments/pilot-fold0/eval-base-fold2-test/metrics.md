# Fine-tuned Laya evaluation: base-fold2-test

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
| accuracy | 0.2683 |
| macro_f1 | 0.2228 |

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| cause | 0.1207 | 0.0986 | 0.1085 | 142 |
| caused_by | 0.1218 | 0.7600 | 0.2099 | 150 |
| none | 0.8615 | 0.2196 | 0.3500 | 1020 |

## Binary existence

| metric | value |
| --- | --- |
| precision | 0.2433 |
| recall | 0.8767 |
| f1 | 0.3810 |
| accuracy | 0.3659 |

## Strict direction (iLIF-style)

| metric | value |
| --- | --- |
| tp | 128 |
| fp | 796 |
| fn | 164 |
| tn | 224 |
| precision | 0.1385 |
| recall | 0.4384 |
| f1 | 0.2105 |
| n | 1312 |

## Status

| status | count |
| --- | --- |
| ok | 1312 |
