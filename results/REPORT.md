# Evaluation Report — phishing-model-3class

- Test emails: **3529**
- Accuracy: **0.9921**
- Macro F1: **0.9924** (precision 0.9928, recall 0.9920)

## Per-class

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| ham | 0.990 | 0.993 | 0.992 | 1647 |
| phishing | 0.989 | 0.983 | 0.986 | 982 |
| ai_phish | 1.000 | 1.000 | 1.000 | 900 |

## Artifacts
- `confusion_matrix.png`
- `per_class_metrics.png`
- (layer_comparison skipped — add a `rules_pred` column to test.csv to include Layer 1 in the comparison)