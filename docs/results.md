# GlacioLens-GLB-v1: results and comparisons

**GlacioLens-GLB-v1 achieves 87.84% two-class mIoU on the documented GLB test split, exceeding all 15 GLB scores reported in [Cryo-Bench v5, Table VIII-B](https://arxiv.org/html/2603.01576v5#S5.T8a), including TerraMind's 83.97% (+3.87 percentage points).** This is a comparison of reported complete-system results; GlacioLens uses additional physical DEM and its own training recipe.

The accepted final model scores **87.83726284704913% two-class mIoU** and **76.76436170643849% lake IoU** on 1,913 test tiles. Scores below use percentages; differences use percentage points.

## Final evaluation populations

| Cohort | Images | Two-class mIoU (%) | Lake IoU (%) |
|---|---:|---:|---:|
| Selection VAL | 1,912 | 88.590018723 | 78.259971203 |
| TEST | 1,913 | 87.837262847 | 76.764361706 |
| Dependent GLBC stress cohort | 1,105 | 83.749321226 | 69.055282163 |

Scores pool confusion counts over valid pixels; they are not means of per-image IoUs. [Original aggregate metrics](../results/metrics.json) retain exact counts and fractions. The validation-selected checkpoint and completed training recipe are documented in [the model card](../MODEL_CARD.md) and [reproduction guide](reproduction.md).

## External scalar reference

The reference rows reproduce the paper's validation-selected learning-rate results. U-Net and ViT were trained from scratch; the other reference models were fine-tuned. The GlacioLens row is the project's measured final result.

| Model | Two-class mIoU (%) |
|---|---:|
| **GlacioLens-GLB-v1** | **87.84** |
| TerraMind | 83.97 |
| CROMA | 82.38 |
| SatlasNet | 82.01 |
| GFM-Swin | 81.38 |
| Scale-MAE | 80.69 |
| DOFA | 80.47 |
| U-Net | 80.46 |
| ViT | 79.66 |
| SpectralGPT | 78.57 |
| S12-MoCo | 78.49 |
| S12-Data2Vec | 77.97 |
| RemoteCLIP | 77.79 |
| Prithvi | 77.76 |
| S12-DINO | 77.66 |
| S12-MAE | 77.53 |

The difference from TerraMind is **+3.86726284704913 percentage points**, approximately **+4.61% relative**. The machine-readable comparison preserves the paper's original two-decimal precision and the project's full stored precision.

## Small lakes, boundaries and false detections

The retained comparison below uses the **same 1,913 test tiles and valid-pixel support** for all four systems. The TerraMind and CROMA rows are **released frozen-encoder references with trained GLB decoders**. They are distinct from the selected fine-tuned paper rows above.

| Metric | GlacioLens-GLB-v1 | Released TerraMind, frozen encoder | Released CROMA, frozen encoder | Released U-Net |
|---|---:|---:|---:|---:|
| Two-class mIoU (%) | **87.84** | 79.91 | 77.68 | 80.46 |
| Lake IoU (%) | **76.76** | 61.61 | 57.41 | 62.83 |
| Object F1 at IoU 0.50 (%) | **60.18** | 48.63 | 40.57 | 44.26 |
| Tiny targets with at least 50% coverage | **143 / 275** | 20 / 275 | 6 / 275 | 34 / 275 |
| Small targets with at least 50% coverage | **864 / 1,089** | 496 / 1,089 | 316 / 1,089 | 585 / 1,089 |
| Boundary F1 (%) | **74.23** | 64.22 | 54.41 | 63.10 |
| Zero-overlap false-object count, lower is better | 1,868 | **1,088** | 1,302 | 2,453 |
| Zero-overlap false-object area in pixels, lower is better | **492,974** | 572,160 | 773,749 | 1,099,965 |

These results demonstrate improved small-lake coverage, object matching and boundary quality against the evaluated released references. Against frozen-encoder TerraMind, tiny-target coverage increases from **20 to 143 of 275**, and boundary F1 rises from **64.22% to 74.23%**. The false-detection tradeoff is measurable: total area of entirely false objects decreases, while their count increases. Against the released U-Net, both measures decrease.

## What the metrics mean

- **Pixel scores:** sum TP, FP, FN and TN over valid pixels before computing lake IoU `TP / (TP + FP + FN)` and background IoU `TN / (TN + FP + FN)`; two-class mIoU is their arithmetic mean.
- **Objects:** eight-connected components. Object F1 uses one-to-one matching at IoU 0.50, maximizing match count before summed IoU; it includes all visible objects.
- **Tiny and small:** complete ground-truth objects of 16–63 and 64–255 pixels, respectively. Complete objects touch neither the image edge nor invalid support through an eight-neighbour connection. These denominators therefore exclude border-truncated objects.
- **Coverage:** predicted intersection divided by ground-truth object area. Reaching 50% coverage is different from obtaining an object IoU of 0.50.
- **Boundary F1:** boundaries are foreground pixels removed by a 3×3 erosion; matching uses a two-pixel Chebyshev tolerance and safe valid support obtained by a 7×7 erosion.
- **Entirely false objects:** predicted eight-connected components with zero overlap with any ground-truth lake. Their count and area are separate from unmatched-object FP counts used in object F1 and from total false-positive pixels.

## Reproduce the measured comparisons

The [saved-results package](../results/reproduction/README.md) supplies the actual four-model TEST predictions, reference labels, valid support, all 7,652 per-tile confusion-count rows, exact 1,201-group mapping, metric definitions and bootstrap settings. It computes the object/boundary statistics from masks and reuses the existing public bootstrap implementation unchanged.

```sh
python -m pip install -e .
python -m pip install numpy==2.3.5 scipy==1.17.0
python tools/reproduce_results.py --output artifacts/reproduced_results
```

The original CPU reconstruction matched 498 retained numeric values with maximum absolute residual 0.0, including both 2,000-replicate bootstrap regimes. It used no model weights, satellite images or GPU. The command also checks its outputs against the original retained verification target; those expected values are not used to compute the metrics.

This reconstructs the four **measured TEST** rows and their intervals. The paper's fifteen selected-training scores remain externally reported scalars. Validation/GLBC endpoints, reference training runs and a portable full training launcher are separate scopes.

## Existing uncertainty estimates

The retained paired analysis resampled 1,201 filename-derived source groups 2,000 times. Its 95% percentile intervals for the **GlacioLens minus released reference** difference in two-class mIoU are:

| Released reference | Difference (percentage points) | 95% interval (percentage points) |
|---|---:|---:|
| TerraMind, frozen encoder | +7.93 | [+5.79, +10.25] |
| CROMA, frozen encoder | +10.16 | [+8.22, +12.28] |
| U-Net | +7.38 | [+5.55, +9.14] |

## Interpretation and scope

The headline is a descriptive comparison with the specified paper table, using additional physical elevation data and a custom training recipe. Selected fine-tuned reference predictions and exact execution overrides are unavailable, so the detailed object and boundary comparisons and paired intervals apply only to the separately evaluated released references. One final training seed, historical TEST exposure, and shared acquisitions/aliases limit independent-generalization claims; the additional GLBC cohort is a dependent stress evaluation. The intervals describe evaluation-composition uncertainty for fixed predictions, not training-seed variability. GlacioLens still misses some small lakes and produces false detections, and this GLB result does not establish superiority across all Cryo-Bench tasks or all published lake-segmentation methods.

**Sources and data:** [Cryo-Bench v5](https://arxiv.org/abs/2603.01576v5), [Table VIII-B](https://arxiv.org/html/2603.01576v5#S5.T8a), the [author's implementation](https://github.com/Sk-2103/Cryo-Bench), and the project's retained final TEST counts, component analysis and paired bootstrap results. The accompanying [comparison CSV](../results/benchmark_comparison.csv) separates published scalars from comparisons of retained predictions; blank cells mean unavailable or inapplicable, never zero.


## Historical development context

The accepted public model is P56 EMA12 / step 11,484, selected by full-validation two-class mIoU after 24 epochs. Historical P53/P54 development reached TEST 88.490940476% two-class mIoU / 77.991265809% lake IoU and GLBC 85.443749428% / 72.263574264%. Those higher development results are not substituted for this release. The final model differs by -0.653677629 / -1.226904103 percentage points on TEST and -1.694428202 / -3.208292101 on GLBC, respectively. P53-native-r1 is a separate package.

The retained paired intervals against the internal B, C0, N and P53 systems include zero. The same analysis also reports large/border extent losses against some internal references. These facts do not negate the separately specified external comparisons, but they prevent presenting this model as an unqualified improvement over every development configuration.

## Scope and tradeoffs

The fixed GLB population includes retained TRAIN/VAL/TEST aliases and shared acquisition groups. GLBC exposure grouping contains 381 TRAIN-overlap observations, 75 heldout-only overlap observations and 649 with no positive-area overlap; the latter is not proof of geographic independence. Fully valid native-tile evidence does not qualify arbitrary nodata imagery, missing DEM, other input domains or production deployment. See the [model card](../MODEL_CARD.md) for input and software-qualification scope.

## Error analysis

The front-page gallery deliberately illustrates successful cases. A retained difficult case, `NZ_S2B_59GMM_20200814_0_L2A_150`, has lake IoU 30.02% with TP 1,725, FP 6 and FN 4,016; most of its error is missed lake extent. Its prediction and target are included in the public saved-results package. The [earlier evaluation gallery](https://github.com/AliSajedifar/GlacioLens/blob/1bb1b6eb192bf491b2a9b09aa2c56c9a4efe6204/docs/figures/glaciolens_qualitative.png) and [immutable case provenance](https://github.com/AliSajedifar/GlacioLens/blob/1bb1b6eb192bf491b2a9b09aa2c56c9a4efe6204/docs/figures/glaciolens_qualitative.json) preserve its visual evidence. Example selection does not change the complete-cohort metrics or the false-detection/extent tradeoffs above.
