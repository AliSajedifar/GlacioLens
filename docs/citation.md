# Citation and acknowledgment guidance

Use the project title **GlacioLens: TerraMind-Based Multimodal Glacial Lake Segmentation** and identify the measured model as **GlacioLens-GLB-v1 (P56 EMA12 / step 11,484)**. Record the exact commit you used from the [source repository](https://github.com/AliSajedifar/GlacioLens). The display name does not rename the `glacial-lake` distribution, `glacial_lake` imports or historical scientific records.

The repository is maintained by **Ali Sajedifar** ([AliSajedifar](https://github.com/AliSajedifar)). Cite the project title, maintainer, repository URL and exact commit used, together with the model identity and upstream acknowledgments below. No formal paper citation, coauthor list, affiliation or DOI is claimed.

## Upstream work

- *Cryo-Bench: Benchmarking Geospatial Foundation Models for Cryosphere Mapping*, [arXiv:2603.01576v5](https://arxiv.org/html/2603.01576v5), specifically **Table VIII-B, GLB, selected fine-tuned TerraMind: 83.97%**. Its scalar is used only with [the comparison qualifications](results.md#external-scalar-reference).
- [Cryo-Bench source](https://github.com/Sk-2103/Cryo-Bench): inspected GLB loading/comparison implementations. Its upstream notice is described in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
- [TerraMind](https://github.com/IBM/terramind) and [TerraTorch](https://github.com/torchgeo/terratorch): upstream backbone, model factory and decoder implementation. The scientific TerraTorch source is pinned to commit `f967ef77b00fe9097157a29d1dad4fda93aba4f9`; exact relevant upstream file fingerprints are in the recipe.
- [Official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/): verified torch 2.14.1 /torchvision 0.29.1 CUDA 13.0 wheel-source guidance. This is installation metadata, not evidence of a newly tested scientific environment.

Use each upstream project's own citation metadata when preparing a paper or release. Distinguish upstream model architectures from this project's integration, physical-input preparation and training/evidence work. Do not imply upstream authors endorse this experiment. The Cryo-Bench MIT notice does not establish rights over this entire project or external weights/data.
