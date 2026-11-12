# Reproducibility and limitations

1. Train-time labels originated from visual JMA products. Agreement with the *current* JMA visual HRP rendering is not independent validation.
2. Use synchronized B13 and HRP for the same JMA region, image filename token and current-day HTTP server timestamp.
3. Do not use numerical Dice/IoU unless both images have identical pixel dimensions and preserved image-space co-registration.
4. The screen-space reference mask is extracted by approximate RGB magenta threshold, clipped to a central margin; it is not the underlying JMA HRP vector/grid product.
5. RainNet uses original 400×400 RGB tiles, float32 /255 normalization. Overlap tiles are averaged in the new real-time GUI, which differs slightly from original stitched-patch behavior.
6. DSC and IoU from the GUI quantify visual agreement, not verified rainfall events, millimeters per hour, spatial forecasting lead time, or warning skill.
7. JMA HRP uses multiple infrared channels and detects cloud-top proxy properties. Avoid interpreting the result as a direct precipitation estimate.
8. Since the dissertation reported a best setting with filter sequence [4,8,16,32,64] and dropout 0.5, the app defaults to that archived checkpoint. This selection is based on the dissertation and does not claim an independent 2026 model optimization.
9. Multi-agent is an explicit workflow separation with independent validation and reporting modules; it is not an autonomous swarm or a language model providing unverified weather predictions.
10. The live link is subject to JMA server status. Never substitute yesterday's JPEG for current imagery without prominently identifying it as historical.
