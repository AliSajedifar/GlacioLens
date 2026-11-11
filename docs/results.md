# GlacioLens results and comparison scope

## Accepted final endpoint

**GlacioLens-GLB-v1** is the accepted P56 seed234 EMA12 model, selected by full-validation two-class mIoU after a 24-epoch run. Its TEST1913 result is **87.8373% two-class mIoU and 76.7644% lake IoU**. TEST followed the selection lock; GLBC followed TEST. Scores are **percentages** here; [machine-readable metrics](../results/metrics.json) contain fractions and exact confusion counts.

| Cohort | Images | Two-class mIoU (%) | Lake IoU (%) |
|---|---:|---:|---:|
| Selection VAL | 1,912 | 88.590018723 | 78.259971203 |
| TEST | 1,913 | 87.837262847 | 76.764361706 |
| Dependent GLBC stress cohort | 1,105 | 83.749321226 | 69.055282163 |

Historical P53/P54 reached TEST 88.490940476% class mIoU /77.991265809% lake IoU, and GLBC 85.443749428% /72.263574264%. Those higher development scores are not substituted for the accepted final endpoint. P56 changes were −0.653677629 /−1.226904103 percentage points on TEST and −1.694428202 /−3.208292101 on GLBC. P53-native-r1 remains a separate EMA13 package.

## What the metrics mean

Sum TP, FP, FN and TN over **all valid pixels** before taking ratios:

```text
lake IoU       = TP / (TP + FP + FN)
background IoU = TN / (TN + FP + FN)
two-class mIoU = (lake IoU + background IoU) / 2
```

`losses_metrics.pooled_metrics` returns fractions, and JSON `null` for a zero denominator. Both class unions must exist for two-class mIoU. The CPU `recount` utility uses that same function; its [synthetic example](quickstart.md#recount-the-synthetic-example) illustrates why averaging per-image IoUs gives a different answer. The published benchmark row is not the average of image-level IoUs.

Pixel IoU, object-matching F1 and tiny-object coverage answer different questions. The retained analysis defined objects by 8-connectivity; “complete” excludes image-edge and 8-neighbor invalid contact. Tiny complete GT components have 16–63 pixels and small ones 64–255. Coverage >=50% means **intersection /GT component area >=0.5**; it is not an object-IoU>=0.5 match. The separate matching analysis used one-to-one maximum-cardinality matching, then summed IoU, at thresholds 0.25 and 0.50. These definitions describe retained analysis, not extra outputs from the public pixel-count CLI.

## External scalar reference

The primary [Cryo-Bench arXiv v5 Table VIII-B, GLB column](https://arxiv.org/html/2603.01576v5) reports 83.97% test mIoU for selected fine-tuned TerraMind. Relative to that scalar, P56's point score differs by **+3.8673 percentage points**, or **+4.6055% relative**.

This is a complete-system comparison: P56 includes additional physical DEM and a custom complete training recipe. Attributable selected reference weights and exact execution overrides are unavailable in the retained evidence. It is not an isolated backbone gain, a matched-budget reproduction or a paired comparison against that selected model. Released frozen-reference systems with trained GLB decoders have separately retained masks; their comparisons are different evidence and do not stand in for the paper's selected fine-tuned checkpoint.

## Scope and tradeoffs

Only one fresh final seed was run. Historical TEST exposure and retained TRAIN/VAL/TEST aliases/shared acquisition groups limit a claim of independent generalization. Exact split membership was preserved; P56 did not use a new geographic partition. GLBC is a dependent stress cohort: retained exposure grouping contains 381 TRAIN-overlap tiles, 75 heldout-only-overlap tiles and 649 with no positive-area overlap. The latter label is not proof of geographic independence.

The original analysis used 2,000 paired resamples at tile and 1,201 filename-group levels. Internal paired comparisons include zero. This conditional uncertainty and a single point-score table do not establish multi-seed consistency or universal superiority.

Small-object gains coexist with additional TEST false-positive area and GLBC large/border true-positive losses. False detections, complete lake extent and tiny coverage remain open limitations. Fully valid native tile evidence does not qualify arbitrary nodata imagery, missing DEM, new sensor domains or production deployment. See [the model card](../MODEL_CARD.md) for identity and intended-use limits.
