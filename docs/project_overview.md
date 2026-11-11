# GlacioLens project overview

GlacioLens is a TerraMind-based research model for glacial lake segmentation using multimodal satellite imagery and elevation data. Its first trained-model identity, **GlacioLens-GLB-v1**, achieved **87.8373% two-class mIoU and 76.7644% lake IoU on 1,913 recorded GLB TEST tiles**.

## Problem and approach

Glacial lakes may occupy only a small part of a satellite tile. Bright terrain can resemble water, and a plausible detection can still miss a substantial part of a shoreline. GlacioLens studies these problems on the original registered image grid, where lake extent and valid support remain explicit.

The model integrates TerraMind-Large and a UPerNet decoder with optical/SAR inputs and physical DEM. Elevation enters both the base encoder and a learned spatial residual. That residual combines local input detail with decoder context captured during the same forward pass. The training method combines dense segmentation, component-positive and large false-region objectives with a fixed source-balanced supplementary plan. These are evidenced integration and adaptation choices; the accepted result does not provide an isolated ablation claim for any one of them. See the [architecture and callable source map](method.md).

## What was built

- **An explicit multimodal input contract.** Original grids, frozen training statistics, physical elevation units and genuine usable support keep model inputs traceable. The guide distinguishes recorded stored SAR values from an unverified physical VV/VH interpretation.
- **A complete recorded experiment.** The final seed234 run trained all 319,492,514 parameters for 24 epochs and 22,968 updates. Full-validation two-class mIoU selected EMA12 before the final TEST and stress endpoints. Pair accumulation, epoch tails, supplementary views and EMA state are documented explicitly.
- **An inspectable inference interface.** A local checkpoint and target-free image/DEM manifest produce GeoTIFF masks and an identity/work receipt. Shared CPU preflight rejects malformed, unregistered or incomplete inputs before CLI model construction.
- **A usable source package.** Lightweight CPU commands, a deterministic saved-mask example, focused technical guides and an explicit export allowlist make the software understandable without private assets. Byte protections preserve the runtime's bound resources through Git delivery.

## Result and interpretation

GlacioLens-GLB-v1 records **87.8373% two-class mIoU /76.7644% lake IoU on TEST1913** and **83.7493% /69.0553% on the dependent GLBC1105 stress cohort**. The [results guide](results.md) and [aggregate metrics](../results/metrics.json) give denominators, full precision and metric definitions.

The TEST two-class mIoU is **3.8673 percentage points** above the published task-adapted TerraMind scalar of 83.97% in [Cryo-Bench v5, Table VIII-B](https://arxiv.org/html/2603.01576v5). This is an unpaired complete-system comparison with different inputs/training: GlacioLens adds physical DEM and its own recipe. The selected reference checkpoint and execution overrides were not reproduced. The difference does not establish a backbone-specific gain or matched-budget superiority.

The accepted P56 result is lower than the historical P53/P54 development result. Small-object gains coexist with additional TEST false-positive area and GLBC large/border losses. One final seed, historical TEST exposure, shared acquisitions/aliases and GLBC dependence limit geographic and robustness claims. [The model card](../MODEL_CARD.md) keeps those limitations alongside the release identity.

## Availability and next steps

Start with the [CPU quickstart](quickstart.md) to inspect metadata and evaluate synthetic saved masks. Users who already have the exact checkpoint and valid inputs can follow the [native-inference guide](inference.md). The original P61 CLI qualification covered three prescribed L4 images in batches [2,1], with zero differing pixels, unchanged state and exact geometry. Later CPU-verified maintenance and this source snapshot do not extend that qualification to arbitrary devices, inputs or batching.

The [source repository](https://github.com/AliSajedifar/GlacioLens) is published under Apache-2.0, maintained by Ali Sajedifar. Selected P56 weights are [public on Hugging Face](https://huggingface.co/MaxiWb/glaciolens-glb-v1) under Apache-2.0, with their original byte size and SHA256 verified. A portable full training launcher is not supplied. [Release status](release_status.md) separates current capabilities, licensing and model delivery. For deeper technical review, follow the [identity](../results/model_identity.json), [recorded recipe](../configs/p56_recipe.json), [input preparation](reproduction.md) and [source provenance](../release/source_provenance.json).
