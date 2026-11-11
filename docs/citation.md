# Citation and acknowledgment guidance

If you use GlacioLens in your research, please cite the software and acknowledge relevant upstream work. GitHub's **Cite this repository** button on the [repository page](https://github.com/AliSajedifar/GlacioLens) generates APA and BibTeX from [CITATION.cff](../CITATION.cff). Use the citation style required by your venue.

## Software citation

Sajedifar, A. (2026). *GlacioLens: TerraMind-Based Multimodal Glacial Lake Segmentation* (Version 0.1.0) [Computer software]. https://github.com/AliSajedifar/GlacioLens

```bibtex
@software{sajedifar_glaciolens_2026,
  author  = {Sajedifar, Ali},
  title   = {{GlacioLens: TerraMind-Based Multimodal Glacial Lake Segmentation}},
  version = {0.1.0},
  year    = {2026},
  url     = {https://github.com/AliSajedifar/GlacioLens}
}
```

For traditional BibTeX styles that do not support `@software`, use:

```bibtex
@misc{sajedifar_glaciolens_2026,
  author       = {Sajedifar, Ali},
  title        = {{GlacioLens: TerraMind-Based Multimodal Glacial Lake Segmentation}},
  year         = {2026},
  howpublished = {Computer software, version 0.1.0},
  url          = {https://github.com/AliSajedifar/GlacioLens}
}
```

This is a software citation, not a claim of a peer-reviewed GlacioLens paper or DOI. Citation is a scholarly request; the [Apache-2.0 license](../LICENSE) and existing attribution terms remain unchanged.

## Record the revisions used

Record the exact source commit used in your experiment. Software version **0.1.0** identifies the `glacial-lake` distribution; **GlacioLens-GLB-v1** identifies the trained model, **P56 EMA12 / step 11,484**. The model name does not rename `glacial_lake` imports or historical scientific records.

If you use the trained model, also record the exact Hugging Face weight revision and checkpoint identity. The released weights are pinned to [bc65beeff3604d51e2242cb51536d5b51ff702f3](https://huggingface.co/MaxiWb/glaciolens-glb-v1/tree/bc65beeff3604d51e2242cb51536d5b51ff702f3), independently of later model-card edits. See the [model card](../MODEL_CARD.md) and [inference guide](inference.md) for the checkpoint SHA256 and qualified execution scope. Record the revision actually used rather than substituting the latest repository or card revision.

## Upstream work

- *Cryo-Bench: Benchmarking Geospatial Foundation Models for Cryosphere Mapping*, [arXiv:2603.01576v5](https://arxiv.org/html/2603.01576v5), specifically **Table VIII-B, GLB, selected fine-tuned TerraMind: 83.97%**. Its scalar is used only with [the comparison qualifications](results.md#external-scalar-reference).
- [Cryo-Bench source](https://github.com/Sk-2103/Cryo-Bench): inspected GLB loading/comparison implementations. Its upstream notice is described in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
- [TerraMind](https://github.com/IBM/terramind) and [TerraTorch](https://github.com/torchgeo/terratorch): upstream backbone, model factory and decoder implementation. The scientific TerraTorch source is pinned to commit `f967ef77b00fe9097157a29d1dad4fda93aba4f9`; exact relevant upstream file fingerprints are in the recipe.
- [Official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/): verified torch 2.14.1 /torchvision 0.29.1 CUDA 13.0 wheel-source guidance. This is installation metadata, not evidence of a newly tested scientific environment.

Use each upstream project's own citation metadata when preparing a paper or release. Distinguish upstream model architectures from this project's integration, physical-input preparation and training/evidence work. Do not imply upstream authors endorse this experiment. The Cryo-Bench MIT notice does not establish rights over this entire project or external weights/data.

## Data and elevation acknowledgments

Cite the relevant GLB/GLBC dataset and Copernicus GLO-30 elevation sources separately. Retain GLB CC BY 4.0 attribution and the applicable Copernicus adapted-product notices described in [third-party notices](../THIRD_PARTY_NOTICES.md#glb-and-glbc-data-attribution). The software citation does not replace dataset or upstream acknowledgments.
