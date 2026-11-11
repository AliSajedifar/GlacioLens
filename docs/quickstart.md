# GlacioLens CPU quickstart

Your first success is a real saved-mask recount with **TP=3, FP=2, FN=1, TN=6**, lake IoU **0.5** and two-class mIoU **0.5833333333**. The masks are synthetic: this checks installation, counting and metric interpretation, not GlacioLens prediction quality. No imagery, checkpoint, credentials or GPU is needed.

Get the [public repository](https://github.com/AliSajedifar/GlacioLens):

```sh
git clone https://github.com/AliSajedifar/GlacioLens.git
cd GlacioLens
```

All relative commands below run from that folder containing `pyproject.toml`. Alternatively, an exported source ZIP contains one `glacial_lake_detection/` directory; enter that extracted directory. A GitHub-generated archive uses its own enclosing directory. No public package-index distribution is claimed.

## Install the lightweight package

GlacioLens retains the distribution and console name `glacial-lake`, with Python import `glacial_lake`. Python >=3.11 is declared. NumPy >=2.2 is its sole runtime dependency for saved-mask operations; pip also resolves build tooling from `pyproject.toml`. Use an isolated environment, with internet or an authorized package cache supplying those packages. Do not request the optional `[science]` extra for this tutorial.

Windows PowerShell, with an installed Python on PATH:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\glacial-lake.exe --help
```

Linux/macOS shell:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/glacial-lake --help
```

These explicit executable paths need no shell activation or execution-policy changes. Help lists `inspect-recipe`, `inspect-identity`, `check-manifest`, `recount` and the lazily loaded `predict` command. There is no `train` command. Prediction requires the separate [L4 scientific setup and authorized assets](inference.md); it is not part of this CPU tutorial. For contributors who intend to edit Python code, `python -m pip install -e .` using the chosen environment is an editable alternative; the consumption route verified here is the normal local install above.

## Recount the synthetic example

Windows:

```powershell
.\.venv\Scripts\python.exe examples/make_synthetic_masks.py example_output/masks.npz
.\.venv\Scripts\glacial-lake.exe recount example_output/masks.npz
```

Linux/macOS:

```sh
.venv/bin/python examples/make_synthetic_masks.py example_output/masks.npz
.venv/bin/glacial-lake recount example_output/masks.npz
```

A successful recount prints JSON containing `"pooled": [3, 2, 1, 6]` and `"model_calls": 0`. The generator prints the path, synthetic-data label, dimensions and expected answers. It writes three uint8 arrays in one NPZ. It refuses to overwrite an existing file; choose another new output name when repeating the example. No training, prediction or benchmark data is involved.

Each sample is one row of eight pixels. `1` means lake in prediction/target and included in valid support; `0` means background or excluded support. Only the first six positions in each sample are counted:

| Sample | Target | Prediction | Valid | TP | FP | FN | TN |
|---|---|---|---|---:|---:|---:|---:|
| 1 | 11100010 | 11010011 | 11111100 | 2 | 1 | 1 | 2 |
| 2 | 10000010 | 11000001 | 11111100 | 1 | 1 | 0 | 4 |
| Pooled | | | | 3 | 2 | 1 | 6 |

The real CLI prints JSON with `samples: 2`, `columns: ["tp", "fp", "fn", "tn"]`, `pooled: [3, 2, 1, 6]`, `model_calls: 0`, and this `metrics` object:

```json
{
  "tp": 3, "fp": 2, "fn": 1, "tn": 6,
  "valid_pixels": 12,
  "lake_iou": 0.5,
  "lake_dice": 0.6666666666666666,
  "precision": 0.6,
  "recall": 0.75,
  "background_iou": 0.6666666666666666,
  "two_class_miou": 0.5833333333333333,
  "undefined_convention": "null; two-class mIoU requires both class unions",
  "aggregation": "pixel-pooled confusion counts"
}
```

These ratios are **fractions**, so two-class mIoU is approximately 58.33333333%. Lake IoU is 3/(3+2+1); background IoU is 6/(6+2+1); mIoU averages the two. Averaging the samples' individual class mIoUs gives 57.5%, which is a different aggregation. The output adds counts before computing metrics.

Generated NPZ files are local outputs, not public source members. `example_output/` is ignored. The fixture dimensions exercise the generic recount format; they are not native model-input dimensions.

## Inspect metadata

The retained `results/model_identity.json` is the P61 identity record with P63 maintenance metadata. Its `availability` and `public_weights_available` fields describe that historical snapshot and are preserved unchanged. Current public access is recorded in [release status](release_status.md#owner-decisions) and the [publication section](../release/source_provenance.json), with the [verified Hugging Face weights](https://huggingface.co/MaxiWb/glaciolens-glb-v1).

In PowerShell, use the installed console executable:

```powershell
.\.venv\Scripts\glacial-lake.exe inspect-identity --file results/model_identity.json
.\.venv\Scripts\glacial-lake.exe inspect-recipe --config configs/p56_recipe.json
.\.venv\Scripts\glacial-lake.exe check-manifest examples/input_manifest.json
```

On Linux/macOS replace `.\.venv\Scripts\glacial-lake.exe` with `.venv/bin/glacial-lake`; arguments and relative files are identical. Identity/recipe inspection prints the supplied JSON unchanged. It does not load a checkpoint, check its availability or start training. Routing inspection prints one row, `routing_schema_valid: true`, `files_checked: false`, `raster_geometry_checked: false`, and `model_calls: 0`.

The generic manifest is a nonempty JSON list. Each row has a unique nonempty string `id` and nonempty strings `image`, `dem`, `usability`. Paths use forward slashes, are relative, and contain no parent traversal, drive colon or backslash. The checker preserves the list order; it does not sort, pair model batches or validate band order. Unknown extra keys are currently ignored. This routing example is distinct from the training dataset's rows and DEM sidecar index described in [reproduction](reproduction.md#input-contract).

To check your own files exist beneath the current directory, use `glacial-lake check-manifest your_manifest.json --root . --check-files` through the same environment's console path. Without `--root`, existence paths resolve relative to the manifest's directory. Without `--check-files`, paths receive only syntax checks and `--root` has no existence-check effect. The bundled example deliberately points to nonexistent generic assets; its existence check should fail until you supply them. Even a passing existence check does not read a raster or establish matching CRS, shape, dtype, transform or physical units.

## Saved-mask format and errors

The recount NPZ must have `prediction`, `target`, `valid` arrays, all **uint8** with identical three-dimensional shape `[N, H, packed_width]`. Pack the last dimension with `numpy.packbits(..., axis=-1, bitorder="big")`. Each byte encodes eight columns. Target and prediction bits denote lake/background; valid bits select counted pixels. Any width-padding bits must have valid=0. Original scientific arrays used `[N,256,32]`, but this CPU utility accepts other sizes.

There is no ID array or automatic sample alignment in this generic format: the producer must put all three arrays in the same sample/pixel order. Extra NPZ keys are ignored. Packed bits are binary already; the utility does not threshold logits or convert 0/255 raster labels. Bytes represent eight bits, not one pixel each. Invalid-position prediction/target bits can be nonzero but never contribute to counts; the fixture demonstrates this.

Missing keys produce a `KeyError`; non-uint8 or non-3D arrays raise `ValueError`; mismatched shapes also raise `ValueError`. Pickled/object arrays are disallowed by NumPy's `allow_pickle=False`. The code cannot infer the intended width, bit order or semantic alignment. A zero denominator produces JSON `null`; two-class mIoU is `null` unless both class unions exist. An empty or all-invalid array has zero counts and undefined ratios. These are data checks, not a P56 prediction interface.

## Troubleshooting and verification scope

- If `python` is missing in PowerShell, use an installed Python 3.11+ launcher such as `py -3.12` for environment creation. Later commands still use the environment's explicit executable.
- If the console executable is missing, confirm the local installation completed in that same environment. `python -m glacial_lake --help` through its Python is an equivalent entry path.
- If NumPy cannot be installed because the package index/cache is inaccessible, fix that access or use an authorized compatible wheel. Do not install the scientific stack as a workaround.
- If the generator reports an existing output, use a different new filename. If the recount file is missing, run the generator first from the documented working directory.
- JSON metadata, examples and configs are repository files, not implicit package data. Keep the source copy and pass the explicit paths shown above after installation.

The exercised scope is CPU onboarding; no trained-model quality is measured by this example. The recorded scientific CUDA environment and any P56 training/inference integration are separate and were not installed or executed here. The public-only archive install passed on Windows 11 /Python 3.12.14 /NumPy 2.5.4; see [release status](release_status.md#exercised-environment) for tested versions and scope. Linux/macOS and other Python versions were not executed; declared Python support is not a cross-platform test matrix.
