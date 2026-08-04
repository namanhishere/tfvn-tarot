# Judge Calibration Report

- Schema: `judge/taxonomy.json` v1.0.0
- Calibration date: 2026-08-04
- Seed: 42  |  n per axis: 100  |  CI: 95% Wilson
- Judges: primary=`deepseek-chat` (provisioned), secondary=`claude-sonnet-4` (provisioned=False)

| Axis | Judge | detected/n | rate | 95% CI | chance | passes |
|---|---|---|---|---|---|---|
| tone | primary | 48/50 | 0.960 | [0.865, 0.989] | 0.167 | yes |
| tone | secondary | 48/50 | 0.960 | [0.865, 0.989] | 0.167 | yes |
| orientation | primary | 28/51 | 0.549 | [0.414, 0.677] | 0.500 | no |
| orientation | secondary | 23/51 | 0.451 | [0.323, 0.586] | 0.500 | no |
| translationese | primary | 31/47 | 0.660 | [0.517, 0.778] | 0.500 | yes |
| translationese | secondary | 38/47 | 0.809 | [0.675, 0.896] | 0.500 | yes |
| faithfulness | primary | 39/48 | 0.812 | [0.681, 0.898] | 0.125 | yes |
| faithfulness | secondary | 39/48 | 0.812 | [0.681, 0.898] | 0.125 | yes |

## Inter-judge disagreement rate

- tone: 0/50 degraded samples (0.0%)
- orientation: 5/51 degraded samples (9.8%)
- translationese: 7/47 degraded samples (14.9%)
- faithfulness: 0/48 degraded samples (0.0%)

## Layer-4 gating

Axes gating C3 layer 4: tone, translationese, faithfulness

no-gate axes (excluded from layer 4): orientation

## Caveat (Metis B4)

Secondary judge is not provisioned in this environment. Until it is, layer 4 is a self-consistency filter and all claims derived from it carry that caveat.
