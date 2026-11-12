# RainNet V1 — Archived checkpoint validation audit (12 November 2026)

The five historical RainNet candidates were evaluated *without retraining*
on 118 archived Dataset 4 validation images (400 x 400). Input arrays in the
HDF5 files are float32 on the 0-255 scale and **must be divided by 255.0**.
All predictions used the same probability threshold of 0.5, with per-image
Dice and IoU averaged over the validation cases.

| Dropout | Mean validation DSC | DSC standard deviation | Mean validation IoU |
|---|---:|---:|---:|
| 0.1 | 0.815602 | 0.110948 | 0.702073 |
| 0.3 | 0.764014 | 0.169970 | 0.644159 |
| **0.5** | **0.816416** | **0.112225** | **0.703124** |
| 0.7 | 0.740851 | 0.182715 | 0.617223 |
| 0.9 | 0.804545 | 0.133173 | 0.691041 |

**Selection:** RainNet dropout **0.5**, the highest observed mean validation
Dice among the five checkpoints, consistent with the dissertation's selected
design. The gap to 0.1 is only 0.000814 DSC and is **not evidence of
statistically significant superiority**.

Reproduce on the local, archived files:
`py -3.8 scripts/audit_model_selection.py`

Local detailed results: `reports/validation_model_selection.json`
(kept off GitHub together with historical models and satellite imagery).

## Scientific limitations

- This is retrospective assessment on a development validation split,
  **not an independent external benchmark**.
- Splitting across time and adjacent spatial tiles has not been audited
  for correlation or potential leakage.
- The JMA HRP image is a qualitative potential-rainfall reference, **not**
  rain-gauge measurement or future rainfall observation.
- JMA visual-magenta thresholding from JPEG is approximate; aligned
  view-space Dice does not establish meteorological forecast accuracy.
- No paired confidence interval or significance test is claimed.
- The GUI is a research prototype, not a public flood-warning tool.
