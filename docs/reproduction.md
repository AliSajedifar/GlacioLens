# GlacioLens inputs and recorded reproduction

The [quickstart](quickstart.md) runs CPU metadata inspection and saved-mask evaluation. This page explains the GlacioLens-GLB-v1 input contract and completed P56 scientific recipe. For prediction with an existing checkpoint, follow the canonical [native-inference route](inference.md); its inputs do not require training labels or sidecar indexes. It is an implementation guide, not a claim that a portable full training launcher ships here.

## Assets and availability

| Asset | Required for | What this source release supplies |
|---|---|---|
| Registered GLB optical/SAR imagery | Training, evaluation and prediction | Contract and generic paths; no imagery. Obtain authorized upstream data separately. |
| Pixel labels and original valid support | Training and evaluation | Label/support semantics; no labels or per-image support. Labels are not inference inputs. |
| Physical DEM and usable support | Model input in every cohort | Preparation contract and normalization; no elevation arrays. External GLO-30 source and equivalent prepared sidecars required. |
| Exact TRAIN/VAL/TEST membership/order and GLBC identity | Reproduction and comparison | Counts and identity metadata; sealed per-image manifests/aliases remain privately retained. An upstream download alone does not establish exact membership. |
| Fixed supplementary pool/plan | Recorded training | Public `build_pools`/`build_plan` logic; materialized source/anchor plan remains private and needs exact input membership/support to reproduce. |
| TerraMind pretrained initializer | Fresh training | Audited loading implementation, recorded pinning and upstream references; no initializer weights. Obtain independently under applicable terms. |
| Selected P56 EMA12 weights | Selected-model inference | [Hash, byte size and tensor identity](../results/model_identity.json); current authorized private access verified in P61; no established public download. |
| Frozen normalization and aggregate metrics | Interpretation and implementation | [Normalization](../configs/normalization.json), [recipe](../configs/p56_recipe.json), [results](../results/metrics.json) bundled. |

A hash establishes identity, not access or redistribution rights. [local_assets.example.json](../configs/local_assets.example.json) is a routing worksheet; no runner consumes that entire file. No owner authentication or hidden download is required by the public CPU tools.

## Input contract

### Native images and targets

The stored raster has **11 bands ×256×256**. `read_registered_image` selects zero-based slots **0,1,2,3,4,5,9,10**, equivalent to Rasterio's one-based bands **1,2,3,4,5,6,10,11**. It reads/casts the eight selected bands to float32 `[8,256,256]`. The six optical slots feed `S2L2A` `[6,256,256]`; the two stored SAR slots feed `S1GRD` `[2,256,256]`. The recipe's upstream initialization band names do not prove that stored SAR slots are physically VV/VH. Do not change value scaling or infer band meaning solely from position.

For paired training/evaluation, `read_tile(root, row)` requires image count=11, target count=1, both 256×256, equal CRS and approximately equal affine transforms. Rows have `id`, `image`, `target`; paths resolve against the explicit `root`. The DEM dataset additionally checks retained `valid_pixels` and `positive_pixels` per row. Row order is supplied by the caller. The native loader checks equality of CRS; preparation must also establish that the geospatial metadata is meaningful, not merely equally absent.

Image support intersects the selected-band raster masks and finite values. Paired support also intersects target masks and finite labels. Observed valid target values must be in {0,1} or {0,255}; the latter maps 255 to lake. Invalid targets are set to zero, with a separate boolean support mask. Stored dtype can vary, but the loader casts selected inputs/targets to float32 and requires these dimensions/semantics. It does not turn arbitrary raw sensor values into the measured stored domain.

Use the exact TRAIN-derived means/stds in [configs/normalization.json](../configs/normalization.json): mean and std arrays are cast to float32, broadcast over the eight channels, then `(x - mean) / std` is evaluated and invalid pixels set to zero as float32. Statistics were fitted over 1,002,700,800 TRAIN pixels, not VAL/TEST. The public data classes require the statistics explicitly; do not refit evaluation statistics or silently fall back to another split.

### Physical DEM preparation

The retained preparation used **Copernicus GLO-30 DSM**, metres relative to **EGM2008**, without vertical-datum conversion. It is a surface model at approximately 30 m; resampling onto the native GLB grid adds no elevation observations. The historical `SourceCache.chip` recipe reprojected physical metres bilinearly to each original raster's CRS/affine/grid, reprojected source validity with nearest-neighbor, and accepted only valid finite results. Its implementation remains in the exact retained executed-source snapshot; that historical acquisition script is not a public command in this source candidate.

The shared acquisition/grid steps are:

1. Establish each image's CRS, affine transform, bounds, band order and stored value domain before any model component is used.
2. Obtain authorized GLO-30 elevations covering the same footprint. Verify metres, scale=1, offset=0, source nodata and vertical reference. Preserve source/grid provenance.
3. Reproject elevations bilinearly and source validity with nearest-neighbor onto the **same 256×256 target grid**, respecting nodata. Check coverage and finite values against the original grid; matching dimensions alone do not prove registration.

#### Training/evaluation sidecars

Continue the historical training/evaluation recipe by producing `sidecar_index.json` with `id`, `member`, `grid_hash`. Under an explicit `dem_root`, each `chips/<member>` NPZ holds float32 `metres`, boolean `valid` (both 256×256) and a scalar `grid_hash`. Bind hashes to checked geometry; do not invent matching hash strings without that check. Validate original target/support counts and exact ID correspondence. Use `NativeDEMDataset(..., arm="B", dem_root=...)` only as a component after those prerequisites are established; no full pipeline command is supplied.

`NativeDEMDataset` defines usable DEM as `valid AND finite(metres)`. It initializes a float32 zero array, computes `(metres[usable].astype(float64) - 651.663) / 928.168`, casts those normalized values to float32, and returns DEM `[1,256,256]`; unusable positions stay zero. The conversion order is intentional. This produces nine model input channels across the three modalities. In supplementary zooms, the usable mask is resized nearest-neighbor and unusable DEM is reset to zero after interpolation.

The measured native GLB cohorts use fully valid native image/target tiles. Implemented zero-fill and DEM imputation are not evidence that arbitrary nodata rasters, incomplete elevations, other sensors, image sizes or geographic distributions are qualified. The generic `image`/`dem`/`usability` routing manifest in [the quickstart](quickstart.md#inspect-metadata) is distinct from these native-loader and sidecar schemas; it validates syntax/existence only.

### Target-free inference files

For [native prediction](inference.md#input-manifest), obtain authorized [GLB imagery](https://zenodo.org/records/17917359) and [Copernicus GLO-30 DSM](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM) under their separate terms. Keep the retained native GLB stored-value domain and 11-band 256×256 grid; do not infer new SAR calibration or convert arbitrary sensor products from band names. Apply the shared grid/reprojection steps above to physical metres with the retained EGM2008 reference, bilinear elevation resampling and nearest-neighbor source-validity resampling. The historical preparation implementation remains retained privately; this guide specifies the handoff, not a new geospatial acquisition command.

Write two **single-band GeoTIFFs** for each native image: physical DEM metres and usability. Both must have exactly the image's dimensions, meaningful CRS and affine transform. Preserve the DEM as physical values; the predictor applies its frozen float64-then-float32 normalization. Derive usability from genuine elevation source coverage/masks and finite reprojected data, and preserve real raster validity masks/nodata. Never manufacture all-one usability to conceal incomplete coverage. Prediction currently accepts only completely valid image/DEM/usability support and usability=1 at every pixel; reject incomplete tiles rather than impute them into this scope.

List those three files in each strict `id,image,dem,usability` manifest row. No labels, target counts, training NPZs, `sidecar_index.json` or `NativeDEMDataset` are required for prediction. The predictor performs no mosaicking, reprojection or physical calibration. A tiny synthetic CPU raster fixture can test this file contract but is neither meaningful model input nor a P61 parity case.

## Recorded training and selection

This is the **completed P56 recipe**, not a runnable launch instruction. Counts: TRAIN 15,300; VAL 1,912; TEST 1,913; dependent GLBC stress cohort 1,105. The final fresh seed is 234, with 24 epochs ×957 updates =22,968 optimizer updates.

| Mechanism | Actual recorded behavior |
|---|---|
| Native order/augmentation | `random.Random(234+epoch)` shuffle; seed234 stateless D4 rotations/flips; loader generator `234000+epoch` |
| Normal update | 16 native views in eight sequential physical pairs; each native pair objective weighted 2/16. Two auxiliary pairs (four views), each weighted 0.25/2. |
| Epoch tail | Four native views in two pairs, weighted 2/4; one auxiliary pair weighted 0.25. Tail auxiliary role pair alternates by epoch. |
| Per epoch | 956 normal updates plus one tail: 15,300 native and 3,826 supplementary views. “Batch size 16” describes native accumulation, not all work in an update. |
| Full run | 367,200 native +91,824 auxiliary training views; each auxiliary role appears 22,956 times. The complete run including validation/endpoints recorded 504,106 model views /252,054 calls. |
| Optimizer | Fused AdamW, betas (0.9,0.999), epsilon 1e-8, no gradient clipping; full parameter coverage. |
| Groups | Encoder peak LR 1e-5; new modules 1e-4. Weight decay 0.05, except biases, tensors of dimension <=1, normalization parameters and encoder-declared no-decay names use zero. |
| Schedule | Applied before each update: linear warmup through update 1,914, then cosine to 0.01× peak at update 22,968. |
| Objective | Native dense +0.20 positive-region loss +hard-negative ramp; auxiliary dense mean weighted 0.25. See [method](method.md#objectives-and-sampling) and [exact excerpt](training_update.md). |
| EMA | Initialize shadow after successful update 1,914. Subsequently average floating parameters with decay 0.9995; copy nonparameter buffers. Update once per successful optimizer step. |
| Selection | Full VAL at EMA epochs 3–24; strict increase in pixel-pooled two-class mIoU, earliest exact ties. EMA12 at step 11,484 was locked after the full run, before TEST then GLBC. |

Recorded outputs have different meanings: selected weights are the EMA12 inference state; final full-resume state contains model/optimizer/EMA/RNG/progress for coherent continuation and is not the selected endpoint. Epoch logs describe 24 completed epochs; identity/selection records bind the choice; saved masks and per-image counts support pooled evaluation. The privately retained original source and evidence remain immutable. The public projection has documented path adapters and no owner storage orchestration.

An implementer still needs exact authorized splits/order and aliases, prepared original-grid inputs/DEM, initial weights and their provenance, the fixed supplementary plan, a qualified batching/precision harness, tensor/gradient/initialization checks, coherent optimizer/EMA/RNG recovery and a validation/selection/export protocol. The public recipe inspector prints JSON; it does not supply those orchestration pieces. Do not substitute the distinct P53 package or fit another seed to present the accepted result.

## Recorded scientific environment

[requirements-science-recorded.txt](../requirements-science-recorded.txt) and the recipe record torch 2.14.1+cu130, torchvision 0.29.1+cu130, TerraTorch 1.2.13 at commit `f967ef77b00fe9097157a29d1dad4fda93aba4f9`, Lightning 2.6.6, NumPy 2.4.6, Rasterio 1.5.2, SciPy 1.16.3 and timm 1.0.31. Three upstream source fingerprints in the recipe further bind the encoder implementation.

The original supervisor used the CUDA 13.0 wheel index below. The [official PyTorch previous-version instructions](https://pytorch.org/get-started/previous-versions/) confirm that index and torch/torchvision pairing. **This Torch step is used by the P61 bootstrap; by itself it is not the complete scientific setup:**

```console
python -m pip install torch==2.14.1+cu130 torchvision==0.29.1+cu130 --index-url https://download.pytorch.org/whl/cu130
```

The original [inference bootstrap](inference.md#colab-installation) established the remaining pins on Colab NVIDIA L4 / Python 3.13.16 in P61. P63 tightens its dependency-check policy and CPU-checks orchestration; the revised whole package was not rerun on L4. A lightweight CPU install alone does not establish this scientific environment. A complete training pipeline remains a separate deliverable.

## Selected model use

P56 EMA12/step11,484, seed234, has checkpoint size **1,280,672,641 bytes**, SHA256 `19bcdaa9f925da61ecc46aab755b265960c632aacdec582bd3050a6b84bd4230`; named-tensor SHA256 is `902ef093300c15d95a7a27fd3771be27c25b7aeae0a0e9ed47b3486025f7345c`. Current authorized private access and exact received bytes were verified in P61; no public checkpoint URL is established. The original endpoint used FP32 parameters, BF16 forwards and lake threshold **logit >=0**, on the exact input/support contract. Pair batching and true odd-tail handling can affect BF16 outputs; arbitrary single-image behavior is not qualified by these records.

[P56-native-r1](inference.md) now supplies the separate target-free CLI/API with strict state/config/source binding and portable local input/output routing. P61 qualified the original pair at TEST indices 0/1 and true singleton at 1912 under the exact scientific stack: three views, two calls, zero mask differences, unchanged state and exact GeoTIFF geometry. No initializer is needed for this selected-state route. This is interface qualification, not new training or a full benchmark. P53 EMA13 remains a distinct package and evidence lineage.
