# Fine-tuned Laya evaluation: pilot-full2b-dev

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
| accuracy | 0.5556 |
| macro_f1 | 0.4668 |

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| cause | 0.3596 | 0.6822 | 0.4710 | 107 |
| caused_by | 0.1743 | 0.6032 | 0.2705 | 63 |
| none | 0.8715 | 0.5297 | 0.6589 | 640 |

## Binary existence

| metric | value |
| --- | --- |
| precision | 0.2850 |
| recall | 0.7059 |
| f1 | 0.4061 |
| accuracy | 0.5667 |

## Strict direction (iLIF-style)

| metric | value |
| --- | --- |
| tp | 111 |
| fp | 301 |
| fn | 59 |
| tn | 339 |
| precision | 0.2694 |
| recall | 0.6529 |
| f1 | 0.3814 |
| n | 810 |

## Status

| status | count |
| --- | --- |
| ok | 810 |
