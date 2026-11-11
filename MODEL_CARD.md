# Model card: GlacioLens-GLB-v1

## Intended use and identity

**GlacioLens** is a research model family for binary glacial lake segmentation on registered GLB optical/SAR tiles with physical elevation. Its descriptive title is **GlacioLens: TerraMind-Based Multimodal Glacial Lake Segmentation**.

| Identity | Meaning |
|---|---|
| GlacioLens-GLB-v1 | Accepted **P56 EMA12 / step 11,484**, seed234; 24 training epochs /22,968 optimizer updates |
| P56-native-r1 | Existing native-inference runtime provenance for that selected state |
| `glacial-lake` / `glacial_lake` | Unchanged distribution and CLI name /Python import name |
| P53-native-r1 | Separate historical EMA13 development package; not GlacioLens-GLB-v1 |

TerraMind-Large, UPerNet and a learned spatial residual contain **319,492,514** trainable parameters, including **57,249** in the spatial path. TerraMind and UPerNet are upstream architectures; the project integrates and adapts them. Full VAL1912 evaluation at EMA epochs 3–24 selected the state by strict pooled two-class mIoU improvement, with earliest exact ties. Selection was locked before TEST1913, then GLBC1105. [The recipe](configs/p56_recipe.json) and [identity metadata](results/model_identity.json) bind the completed run.

Selected checkpoint size: **1,280,672,641 bytes**. File SHA256: `19bcdaa9f925da61ecc46aab755b265960c632aacdec582bd3050a6b84bd4230`. Named-state SHA256: `902ef093300c15d95a7a27fd3771be27c25b7aeae0a0e9ed47b3486025f7345c`. This is the selected inference state, distinct from the final optimizer-resume file. P61 verified authorized private access and exact received bytes; no public download is established.

## Inputs and prediction scope

Native images are 11-band 256×256 registered rasters. Zero-based slots **0,1,2,3,4,5,9,10** supply six optical and two stored SAR channels; the SAR slots are not physically verified as VV/VH. Frozen TRAIN statistics normalize the stored-value domain. Additional physical DEM comes from **Copernicus GLO-30 DSM, EGM2008**, bilinearly aligned to the image grid. Usable finite metres are normalized in float64 by `(metres - 651.663) / 928.168`, then cast to float32. The DEM enters both the base encoder and spatial branch.

Native inference requires completely valid image/DEM support and a genuine usability mask equal to 1 everywhere. It rejects incomplete coverage; documented training zero-fill behavior does not qualify missing-data inference. Labels are not prediction inputs. [Input preparation](docs/reproduction.md#target-free-inference-files) and the [inference guide](docs/inference.md) specify the file, grid and support requirements.

The recorded endpoint uses FP32 parameters, CUDA BF16 forwards and **logit >= 0** for lake. Consecutive manifest rows form physical pairs and the true odd final row runs alone. Different batch contexts, devices, sizes, sensors or incomplete inputs are outside the qualification.

## Measured result

| Population | Tiles / observations | Two-class mIoU (%) | Lake IoU (%) |
|---|---:|---:|---:|
| Selection VAL | 1,912 | 88.5900 | 78.2600 |
| TEST | 1,913 | 87.8373 | 76.7644 |
| Dependent GLBC stress cohort | 1,105 | 83.7493 | 69.0553 |

These metrics pool confusion counts over valid pixels before taking ratios. They are not per-image means or object-matching scores. [Results and definitions](docs/results.md) explain the distinction; [aggregate data](results/metrics.json) retain full precision.

## Limitations and tradeoffs

One fresh final seed provides conditional evidence. Historical TEST exposure, shared acquisitions/aliases and a dependent GLBC cohort limit independent geographic interpretation. The fixed supplementary plan was shared with development, and the original internal paired uncertainty intervals include zero.

P56 regressed against historical P53/P54 by **0.6537 percentage points** in TEST two-class mIoU and **1.6944 points** on GLBC. Small-object gains coexist with extra TEST false-positive area and GLBC large/border true-positive losses. False detections and complete lake extent remain limitations.

The [published scalar comparison](docs/results.md#external-scalar-reference) is **+3.8673 percentage points** above the paper's 83.97% task-adapted TerraMind score. Additional DEM and a different complete recipe prevent isolated architecture attribution. It is unpaired, with no matched selected-checkpoint reproduction. No universal superiority, production readiness or deployment-safety claim is established.

## Availability and software qualification

The source provides [CPU onboarding](docs/quickstart.md), research components, the recorded recipe and a [target-free local-input CLI/API](docs/inference.md). It includes no datasets, weights or portable full training launcher.

The original P61 installed CLI on NVIDIA L4 used TEST indices **0/1/1912**, batches **[2,1]**, three intentional views and two calls. All three masks matched the retained P56 masks exactly, with unchanged selected state and original GeoTIFF geometry. This bounded interface check is separate from the full P56 scientific evaluation. P63 setup/preflight/Git maintenance was CPU-verified; the revised complete package and current documentation snapshot have not been newly executed on L4. Historical runtime hashes remain separately scoped in [source provenance](release/source_provenance.json).

Owner-controlled source is published at [AliSajedifar/GlacioLens](https://github.com/AliSajedifar/GlacioLens) under [Apache-2.0](LICENSE), maintained by Ali Sajedifar. The selected P56 weights are separately approved for Apache-2.0 distribution; verified public delivery is pending. Data retain their own terms. [Release status](docs/release_status.md#owner-decisions) records current availability. See [citation guidance](docs/citation.md) for attribution.
