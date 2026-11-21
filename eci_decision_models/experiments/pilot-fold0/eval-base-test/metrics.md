# Fine-tuned Laya evaluation: pilot-base-test

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
| accuracy | 0.2249 |
| macro_f1 | 0.1970 |

| class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| cause | 0.1139 | 0.1198 | 0.1168 | 192 |
| caused_by | 0.1118 | 0.7059 | 0.1930 | 187 |
| none | 0.8206 | 0.1698 | 0.2814 | 1266 |

## Binary existence

| metric | value |
| --- | --- |
| precision | 0.2401 |
| recall | 0.8760 |
| f1 | 0.3768 |
| accuracy | 0.3325 |

## Strict direction (iLIF-style)

| metric | value |
| --- | --- |
| tp | 155 |
| fp | 1051 |
| fn | 224 |
| tn | 215 |
| precision | 0.1285 |
| recall | 0.4090 |
| f1 | 0.1956 |
| n | 1645 |

## Status

| status | count |
| --- | --- |
| ok | 1645 |
