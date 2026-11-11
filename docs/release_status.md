# GlacioLens release status

**Source is published at [AliSajedifar/GlacioLens](https://github.com/AliSajedifar/GlacioLens). [Selected P56 weights are public on Hugging Face](https://huggingface.co/MaxiWb/glaciolens-glb-v1) under Apache-2.0.** GlacioLens-GLB-v1 is the accepted trained-model label; the software distribution remains `glacial-lake 0.1.0`. The [model card](../MODEL_CARD.md#intended-use-and-identity) maps those names to the unchanged P56 checkpoint and runtime identities.

## Available capabilities

| Goal | Current capability |
|---|---|
| Try the package without private assets | [CPU quickstart](quickstart.md): normal local install, metadata/routing inspection and deterministic synthetic saved-mask evaluation. |
| Run selected-model prediction | [Native-inference CLI/API](inference.md) with an authorized exact local checkpoint and fully valid native inputs on the recorded L4 stack. [Verified original weights](inference.md#checkpoint-download) are public on Hugging Face under Apache-2.0. |
| Study or reproduce the method | Computational components, exact normalization, aggregate results, [ordered split and materialized sampling metadata](../data/manifests/README.md) and a [recorded recipe](reproduction.md); no portable full preparation/training launcher. |
| Obtain the source | [Public GitHub repository](https://github.com/AliSajedifar/GlacioLens), maintained by Ali Sajedifar, under Apache-2.0. The explicit export allowlist includes source, an attributed qualitative preview and sanitized reproduction metadata. |

The payload contains source, aggregate results, a three-case imagery/label/prediction display figure and sanitized split/source/sampling metadata. It also includes a compact saved TEST mask/count/group bundle and CPU evaluation tools for the four measured comparisons. It contains no weights, raw training raster archive or private orchestration. P53-native-r1 remains a separate historical EMA13 package. CPU checks and the original three-image L4 qualification have distinct scopes, recorded under [exercised environment](#exercised-environment).

## Owner decisions

The repository maintainer is **Ali Sajedifar** ([AliSajedifar](https://github.com/AliSajedifar)); the source destination and publication authorization are resolved. The maintainer also approved **Apache-2.0 for the exact selected P56 weights**. This is a separate weight-license decision, not an inference from the source or initializer license.

| Release area | Current status |
|---|---|
| Source publication | [AliSajedifar/GlacioLens](https://github.com/AliSajedifar/GlacioLens), public; use the exact commit associated with your checkout. |
| Selected checkpoint terms | Apache-2.0, explicitly approved for P56 EMA12 / step 11,484. |
| Model delivery | [MaxiWb/glaciolens-glb-v1](https://huggingface.co/MaxiWb/glaciolens-glb-v1), public and ungated. Original checkpoint verified at revision `bc65beeff3604d51e2242cb51536d5b51ff702f3` by byte size and comparable SHA256 metadata. |
| Real inference example | [Exact three-case inputs and retained expected masks](native_demo.md), delivered as a separate immutable HF archive with data notices. CPU packaging checks preserve original P61 qualification; no new model run. |
| Citation | Maintainer and repository are known; no DOI, affiliation, coauthors or legal copyright history is invented. |

Public checkpoint size/SHA256 and the live source/model links have been verified. Publication does not extend numerical qualification or establish training portability.

## Source payload

The intended repository root is **the contents of `dist/github_source/`**. The equivalent `dist/glacial_lake_detection_source.zip` has one `glacial_lake_detection/` directory; users enter that directory after extraction. Do not publish the whole working project or nest its private/generated directories in the source tree.

To regenerate the candidate, run `python tools/build_public_source.py` from the source root using an installed Python. [release/public_files.json](../release/public_files.json) is the authoritative file selection. The exporter checks actual members, hashes/CRC and private-content patterns, then materializes the same bytes in the tree. The external `dist/glacial_lake_detection_source.manifest.json` records the current archive/tree identity. Unexpected existing tree content is refused for inspection. [Source provenance](../release/source_provenance.json) keeps current presentation metadata separate from historical runtime qualification hashes.

## Licensing scope

| Scope | Verified position for this candidate |
|---|---|
| Owner-controlled code, docs and generated aggregate figure | Apache-2.0; [LICENSE](../LICENSE) is the unmodified official text. Package metadata declares that scope and includes the license and third-party notices in wheels. The maintainer is Ali Sajedifar; no additional copyright history is claimed. |
| Copied/adapted upstream implementations | No redistributed external implementation was identified in the reviewed public source. Project modules adapt interfaces/method recipes and import upstream implementations. Exact pinned exceptions, retained project-reference acknowledgments and comparison limits are in [third-party notices](../THIRD_PARTY_NOTICES.md) and [file provenance](../release/license_provenance.json). |
| Original TerraMind initializer | The [official TerraMind-1.0-large model card](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-large) declares Apache-2.0. This is a verified upstream model declaration, separate from the code repository and from the selected trained P56 state. No initializer is bundled. |
| Selected P56 weights | EMA12 /step11,484; file SHA256 `19bcdaa9f925da61ecc46aab755b265960c632aacdec582bd3050a6b84bd4230`. P61 verified current authorized private access and the received exact file bytes. The maintainer separately approved Apache-2.0 for this selected checkpoint. The unchanged original file is [publicly hosted](https://huggingface.co/MaxiWb/glaciolens-glb-v1) and verified at immutable revision `bc65beeff3604d51e2242cb51536d5b51ff702f3`. No conclusion is made that training automatically transfers a dataset's license to all weights. |
| GLB/GLBC imagery, labels and splits | The [GLB Zenodo deposit](https://zenodo.org/records/17917359) and its [record metadata](https://zenodo.org/api/records/17917359) declare **CC BY 4.0** and open access. The record describes GLB plus a challenge component called GLC; this project's GLBC name identifies the retained stress cohort. The three-case qualitative figure and sanitized dataset-derived split/source/sampling metadata are explicitly attributed CC BY 4.0 derivatives. Source-provider attribution remains applicable; raw satellite imagery and private storage details stay outside this payload. The explicitly distributed TEST reference masks and valid support are losslessly packed derivatives under the dataset attribution; generated predictions are identified separately. No noncommercial restriction is inferred. The CPU tutorial uses synthetic masks. |
| Copernicus GLO-30 and adapted sidecars | The [official Copernicus DEM guidance](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM) permits worldwide GLO-30 access under its free license and specifies source and adaptation notices. The separate [three-case HF example](native_demo.md) includes original qualified sidecars from public GLO-30 COG data, with exact input identities, genuine usability and the [prescribed adapted-data notice/product terms](../examples/native_demo_DATA_NOTICES.md). These raw inputs stay outside the GitHub source; other prepared training sidecars remain separate requirements. |

For adapted GLO-30 material, the prescribed attribution is: “produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved”. The aggregate score plot contains evaluation numbers. The separate qualitative preview includes satellite RGB, reference labels and saved predictions under the dataset attribution above. It contains no DEM raster.

## Exercised environment

The P59 CPU onboarding qualification used an extracted public-only source archive, a new environment without system site packages and a normal `pip install .` route with the explicit PyPI index and no cache. Windows 11, CPython **3.12.14**, NumPy **2.5.4**, pip **25.0.1**, package **glacial-lake 0.1.0** passed help, identity, recipe, routing and synthetic recount checks.

The synthetic result was TP=3, FP=2, FN=1, TN=6; lake IoU=0.5 and class mIoU=0.5833333333333333. A guarded check denied owner project-state access and completed with zero such reads and no model-framework import. Missing-asset, existing-output, dtype/rank/shape and undefined-metric cases behaved as documented.

Python >=3.11 remains declared support. P59 did not exercise Linux/macOS commands or Python versions other than 3.12.14. The numerical training stack was not installed. This qualification did not change or re-evaluate the scientific results.

P60 retains that P59 onboarding qualification. A packaging-only check on Windows /CPython 3.13.11 with setuptools 80.9.0 and wheel 0.45.1 built a wheel from public source and inspected its Apache license metadata and exact notice payloads. The synthetic count/metric path was also checked against the wheel with the existing NumPy 2.4.1 installation. This does not extend model or scientific-runtime qualification. That historical P60 check did not include hosted CI.

P61 installed the actual package with all eight recorded scientific pins and verified the three upstream source fingerprints on Colab NVIDIA L4. The CLI completed one invocation, batches [2,1], three views/two calls, with unchanged selected state and exact three-case retained-mask parity. Independent local Rasterio 1.5.2 decoding verified pixels and geometry after release. The installed wheel also passed all CPU routes without importing Torch. See [the inference guide](inference.md) for the precise scope, installation and the unrelated base-environment `pip check` exception.

P63 maintenance preserves all 35 Git-bound payloads, rejects unexpected dependency-check failures and performs shared CPU preflight before CLI construction. Its actual Git-derived wheel and focused CPU checks passed; the revised complete package has not been newly executed on L4. Current GlacioLens branding/documentation does not change runtime code, scientific settings or the selected state and makes no new L4 qualification claim. The unchanged CPU synthetic tutorial retains its earlier execution evidence.

## CPU result reproduction and public checks

The [results reconstruction](../results/reproduction/README.md) uses saved binary masks and the original metadata to regenerate the measured four-system TEST tables and uncertainty estimates. It does not load the model. A full CPU pass compared 498 retained numbers with maximum absolute residual 0.0 in NumPy 2.3.5 / SciPy 1.17.0. The original bootstrap source and all qualified inference resources remain unchanged.

The small [CPU workflow](../.github/workflows/cpu.yml) builds a wheel, installs it and runs the [12 public CPU contract tests](../tests/test_public_cpu.py) and [5 example-delivery tests](../tests/test_native_demo.py). The local wheel-installed suite passed in Python 3.12.14 with preinstalled NumPy 2.3.5 / SciPy 1.17.0; a GitHub-hosted execution is a separate workflow result. Neither local CPU checks nor publishing this code constitute new model or GPU qualification.
