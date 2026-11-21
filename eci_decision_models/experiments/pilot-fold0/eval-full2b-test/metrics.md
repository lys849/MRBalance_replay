# Fine-tuned Laya evaluation: pilot-full2b-test

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
| accuracy | 0.6912 |
| macro_f1 | 0.5259 |

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| cause | 0.3750 | 0.5312 | 0.4397 | 192 |
| caused_by | 0.2646 | 0.4118 | 0.3222 | 187 |
| none | 0.8854 | 0.7567 | 0.8160 | 1266 |

## Binary existence

| metric | value |
| --- | --- |
| precision | 0.4529 |
| recall | 0.6728 |
| f1 | 0.5414 |
| accuracy | 0.7374 |

## Strict direction (iLIF-style)

| metric | value |
| --- | --- |
| tp | 179 |
| fp | 308 |
| fn | 200 |
| tn | 958 |
| precision | 0.3676 |
| recall | 0.4723 |
| f1 | 0.4134 |
| n | 1645 |

## Status

| status | count |
| --- | --- |
| ok | 1645 |
