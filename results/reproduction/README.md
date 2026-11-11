# Reproduce the GLB test comparisons on CPU

This package rebuilds the measured **GlacioLens-GLB-v1, released frozen-encoder TerraMind, released frozen-encoder CROMA and released U-Net** comparisons on the same 1,913 recorded GLB TEST tiles. It includes the actual saved binary predictions, reference labels and valid support. No model weights, satellite imagery, model execution or GPU is needed.

The selected fine-tuned Cryo-Bench paper scores are separately cited scalars in [the comparison CSV](../benchmark_comparison.csv). Their unavailable predictions are not invented here. This package reproduces the project's measured four-system TEST analysis, rather than the paper authors' training runs or the separate validation/GLBC endpoints.

## Run

From a public source checkout with Python and NumPy/SciPy installed:

```sh
python -m pip install -e .
python -m pip install scipy
python tools/reproduce_results.py --output artifacts/reproduced_results
```

The recorded reconstruction used NumPy 2.3.5 and SciPy 1.17.0. For that exact analysis environment, install `numpy==2.3.5 scipy==1.17.0` in a separate CPU environment. This is the results-analysis environment, not a replacement for the model's qualified CUDA environment.

The command verifies the distributed data identities and then reconstructs pixel confusion counts, connected-component matching, size-group coverage, boundary statistics, false-object counts/area and both existing bootstrap regimes. It compares its computed values with `expected_results.json`; a mismatch raises an error. The expectation file is a verification target, not a source of the computed metrics.

Outputs:

- `matched_reference_comparison.csv`: the four measured reference rows.
- `bootstrap_intervals.csv`: absolute and paired intervals with explicit units.
- `reconstructed_results.json`: full pooled statistics, size groups and intervals.
- `reconstructed_per_tile.csv`: all 7,652 measured model/tile count rows.
- `verification.json`: residuals, versions, execution time and zero-model/GPU accounting.

## What is distributed

| File | Purpose |
|---|---|
| `test_masks.npz` | Exact ordered IDs, GT, valid support and four saved prediction arrays |
| `per_tile_counts.csv` | TP/FP/FN/TN for every model/tile, in model-major then canonical tile order |
| `group_mapping.csv` | Exact tile order, filename-derived group ID and sorted group index |
| `metric_config.json` | Metric conventions, seed, resampling algorithm and source identity |
| `expected_results.json` | Original retained values for strict verification |
| `source_manifest.json` | Data file identities and provenance of source arrays and evaluator logic |

The NPZ uses Unicode IDs and packed `uint8[1913,256,32]` arrays. Decode the final axis with `np.unpackbits(..., bitorder="big")` to recover each 256×256 mask. `allow_pickle=False` is required. Predictions use the keys `GlacioLens-GLB-v1`, `TerraMind-frozen`, `CROMA-frozen` and `U-Net`. GT and support are stored once.

## Metrics and resampling

Pixel metrics pool integer TP/FP/FN/TN over original valid support before taking ratios. Two-class mIoU averages the lake and background IoUs; it does not average image-level IoUs.

Objects are eight-connected components. Matching maximizes the number of eligible matches before their summed IoU, at thresholds 0.25 and 0.50. Complete GT objects touch neither the image edge nor an eight-neighbour invalid pixel. Tiny and small groups contain 16–63 and 64–255 GT pixels. Coverage uses predicted intersection divided by GT area, so coverage at 50% is different from an object IoU match at 0.50.

Boundary pixels are foreground minus a 3×3 erosion. The original valid support is eroded with a 7×7 kernel; boundaries outside that safe support are removed before matching. A 5×5 dilation gives Chebyshev tolerance two pixels. Matched boundary counts and supports are pooled before precision, recall and F1. Entirely false objects are predicted components with zero GT overlap; their count and area are distinct from unmatched-object counts and total FP pixels.

The command imports the **existing public `glacial_lake.uncertainty` implementation unchanged**. Both bootstrap regimes independently initialize `default_rng(56043)` and use 2,000 equal-unit multinomial draws in batches of 100, common to every model in a replicate. Tile resampling uses 1,913 units. Group resampling uses 1,201 exact filename-derived groups, `id.rsplit('_', 1)[0]`, ordered with `np.unique`; group membership and index are explicit in the CSV. A selected group contributes all its member tiles, so the number of tiles in a group-bootstrap replicate can vary. Intervals use linear 2.5th/97.5th percentiles of float64 metric values or paired differences.

These intervals condition on the saved predictions and describe test-composition uncertainty. They do not measure seed variability, correct historical development exposure or establish independent geography. No paired interval is assigned to the paper's selected fine-tuned checkpoints.

## Data attribution

GLB reference labels and valid-support metadata derive from **Glacial-Lake-Bench**, whose Zenodo dataset record credits Saurabh Kaushik, Beth Tellman, Ian Howat, Umesh Haritashya and Lalit Maurya. The dataset paper credits the first four authors. The [dataset paper](https://doi.org/10.5194/essd-2026-474), Data Availability Statement, releases GLB/GLBC at [Zenodo record 17917359](https://zenodo.org/records/17917359) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Adaptations here are selection of the recorded TEST population and lossless binary packing; model predictions and derived counts were produced by the project using the named systems. Preserve the dataset attribution when redistributing the data-derived bundle. The evaluation code uses the repository's Apache-2.0 license; it does not relicense third-party data or reference models.

This is a compact saved-results package. Training imagery, private account paths, checkpoint files and historical workflow reports are not included.
