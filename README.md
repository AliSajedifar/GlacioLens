# GlacioLens

**Multimodal deep learning for glacial lake segmentation**

GlacioLens maps glacial lakes in satellite imagery, combining a fine-tuned **TerraMind-Large encoder**, **UPerNet decoder** and a custom spatial refinement pathway. It brings together optical imagery, stored SAR channels and physical elevation to recover small lakes and trace complex shorelines on the original image grid.

[Results](docs/results.md) · [Model weights](https://huggingface.co/MaxiWb/glaciolens-glb-v1) · [Run inference](docs/inference.md) · [Method](docs/method.md)

![Selected GlacioLens results: clear lake, small lake and missed lake extent, with satellite RGB, labels, predictions and error maps](docs/figures/glaciolens_qualitative.png)

**Qualitative segmentation results.** Three illustrative GLB TEST selections, not a statistical sample: `NZ_S2B_59GLM_20200814_1_L2A_1`, `AK_S2A_9VVC_20200909_0_L2A_231` and `NZ_S2B_59GMM_20200814_0_L2A_150`. Original saved GlacioLens-GLB-v1 predictions; native 256 × 256 context in every panel. RGB uses Red/Green/Blue (stored slots 2/1/0) with a per-band 2nd–98th percentile display stretch. The small-lake row shows the same boxed **64 × 64 crop** enlarged in all four panels. [Exact case provenance and counts](docs/figures/glaciolens_qualitative.json).

Imagery and reference labels: [Glacial-Lake-Bench, Kaushik et al. (2026), Zenodo 17917359](https://zenodo.org/records/17917359), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). RGB rendering, crop enlargement, predictions and error overlays by GlacioLens; [dataset and provider attribution](THIRD_PARTY_NOTICES.md#glb-and-glbc-data-attribution).

## What I built

- **Multimodal model integration.** Combined six optical channels, two stored SAR channels and aligned physical elevation in a TerraMind-based segmentation system.
- **Context-aware spatial refinement.** Implemented a lightweight, 57,249-parameter residual pathway that combines local image detail with live decoder features during the same forward pass.
- **Component-aware training and usable inference.** Integrated small-object and hard-negative objectives with fixed supplementary sampling, then released the selected weights and a GeoTIFF prediction CLI with input-alignment, preprocessing and checkpoint-integrity checks.

The [project overview](docs/project_overview.md) explains the engineering decisions and evidence. TerraMind and UPerNet are credited upstream architectures; these contributions concern their task-specific integration and adaptation.

## Results

**GlacioLens-GLB-v1 achieves 87.84% two-class mIoU on GLB TEST, exceeding all 15 GLB scores reported in [Cryo-Bench v5, Table VIII-B](https://arxiv.org/html/2603.01576v5#S5.T8a).** The highest reported comparator is TerraMind at **83.97%**, a difference of **+3.87 percentage points**.

This is a comparison of reported complete-system results: GlacioLens uses additional physical DEM and its own training recipe. The [comparison guide](docs/results.md#external-scalar-reference) identifies the reference regimes and evaluation scope.

| Evaluation population | Tiles / observations | Two-class mIoU | Lake IoU |
|---|---:|---:|---:|
| GLB validation | 1,912 | 88.59% | 78.26% |
| **GLB test** | **1,913** | **87.84%** | **76.76%** |
| Dependent GLBC stress cohort | 1,105 | 83.75% | 69.06% |

The GLB partition contains **15,300 training, 1,912 validation and 1,913 test tiles**, matching Cryo-Bench's split table. [Dataset provenance](docs/reproduction.md#dataset-version-and-split-sizes) explains the source versions and the dataset paper's different headline total.

### Small lakes and boundaries

On the same test tiles, GlacioLens improves over the **released frozen-encoder TerraMind reference** with its trained GLB decoder:

| Measure | Released frozen TerraMind | GlacioLens |
|---|---:|---:|
| Tiny targets reaching at least 50% ground-truth coverage | 20 / 275 | **143 / 275** |
| Object F1 at IoU 0.50 | 48.63% | **60.18%** |
| Boundary F1 | 64.22% | **74.23%** |

These are separately reproduced reference results, distinct from the paper's selected fine-tuned TerraMind row. The area of entirely false detections decreases, while their count increases. [Detailed results](docs/results.md#small-lakes-boundaries-and-false-detections) include all measured references, definitions, false-detection tradeoffs and existing uncertainty intervals.

## Use the project

```sh
git clone https://github.com/AliSajedifar/GlacioLens.git
cd GlacioLens
```

- **Run the tools on CPU:** the [quickstart](docs/quickstart.md) installs the package and evaluates a small synthetic mask example without weights or satellite data.
- **Predict real imagery:** download the [public model weights](https://huggingface.co/MaxiWb/glaciolens-glb-v1) using the [pinned instructions](docs/inference.md#checkpoint-download), then follow the [inference guide](docs/inference.md) to produce georeferenced masks from your local inputs.
- **Understand or reproduce the method:** verify the [exact split and sampling metadata](data/manifests/README.md), then read the [architecture](docs/method.md), [recorded training recipe](docs/reproduction.md) and [model card](MODEL_CARD.md).

Native prediction expects registered, fully valid **11-band 256 × 256 images**, aligned physical DEM in metres and a genuine usability mask. Labels are not required. The documented inference environment uses NVIDIA L4; detailed input and environment requirements are in the inference guide. The CLI is `glacial-lake` and the Python package is `glacial_lake`.

## Research scope

The results describe one final training seed on a historically exposed benchmark split with shared acquisitions and aliases; GLBC is a dependent stress cohort. Small-lake recovery comes with false-detection and lake-extent tradeoffs. [The model card](MODEL_CARD.md#limitations-and-tradeoffs) documents these limits. Inference is available; a portable end-to-end training launcher is not yet supplied.

## License, attribution and maintainer

Created and maintained by [Ali Sajedifar](https://github.com/AliSajedifar). Source and released model weights are available under **[Apache-2.0](LICENSE)**. Imagery, labels and elevation products retain their own terms. See [third-party notices](THIRD_PARTY_NOTICES.md) and [citation guidance](docs/citation.md).
