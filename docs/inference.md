# GlacioLens native inference

**GlacioLens-GLB-v1** uses the existing `P56-native-r1` interface to restore the accepted **P56 EMA12 / step 11,484** state and write georeferenced lake masks from local image/DEM inputs. It is separate from the historical P53 EMA13 package. A full training launcher is not supplied.

This is the canonical prediction route for users who already have authorized assets:

1. Verify the [exact local checkpoint](#checkpoint-identity) and prepare fully valid native inputs. Weights are not publicly hosted.
2. Extract the obtained source archive and run the [recorded L4 setup](#colab-installation).
3. Create the [ordered, target-free manifest](#input-manifest) using your local image, DEM and usability files.
4. [Run one CLI job](#run-prediction) into a new output directory. The API is an alternative integration route.
5. Check the [completion receipt, package and download the results](#outputs-and-resource-release), then release the runtime.

For a first success without those assets, use the [synthetic CPU quickstart](quickstart.md).

## Availability and scope

P61 qualified the installed public CLI on Colab NVIDIA L4 against the retained P56 masks for original TEST indices **0, 1 and 1912**: **zero differing pixels in all three cases**, with batches **[2, 1]**, three intentional qualification views and two model calls. The selected named state was identical before and after. A separate local CPU check decoded the actual saved GeoTIFFs and verified exact original geometry and 0/1 pixels. This limited interface check adds no new benchmark accuracy or production qualification.

P63 maintenance preserves Git-delivered bytes and enforces strict dependency checks and CPU preflight. Those changes were CPU-checked; the revised complete package and current source snapshot have not been newly executed on L4. The original P61 executed-source identities remain separate in [source provenance](../release/source_provenance.json).

The supported contract is a fully valid native GLB tile: 11 bands, 256×256, the recorded stored-value domain, meaningful CRS/affine metadata, and an aligned physical DEM in metres. Other sensors, image sizes, missing DEM/nodata, arbitrary geography and devices are outside the qualification. The runtime rejects incomplete support. It requires NVIDIA L4 and the exact scientific versions below.

The public source contains no weights or data. Private access to the exact checkpoint and its received bytes was verified during the original P61 qualification; P63 did not re-access weights. Obtain authorized local assets separately; no public weight URL or package-index release is claimed. The maintainer has separately approved Apache-2.0 for the selected P56 weights. Images, labels and DEM products retain their own terms. See [asset notices and delivery status](release_status.md#licensing-scope).

## Checkpoint identity

Supply the original selected file unchanged:

- Bytes: **1,280,672,641**.
- File SHA256: `19bcdaa9f925da61ecc46aab755b265960c632aacdec582bd3050a6b84bd4230`.
- Named-state SHA256: `902ef093300c15d95a7a27fd3771be27c25b7aeae0a0e9ed47b3486025f7345c`.
- Complete graph: **319,492,514** parameters, including **57,249** spatial parameters.

The constructor verifies the file before deserialization, constructs the complete graph with `build_model(..., resume=True)`, clears the initializer path, disables pretrained downloads, and strictly loads the full selected state. It does not need the TerraMind initializer, labels, a cloud account, historical reports, an optimizer or the training dataset. GroupNorm conversion, live decoder context, upstream source fingerprints and named tensors are checked.

## Colab installation

The recorded P61 interpreter was **Python 3.13.16**. This is provenance, not qualification of every fresh Colab image; the lightweight CPU package still declares Python >=3.11. Start a fresh Python 3 notebook and select **Runtime → Change runtime type → L4 GPU**. Upload an already obtained `glacial_lake_detection_source.zip` through the notebook Files panel. Obtain the [published repository](https://github.com/AliSajedifar/GlacioLens) and run `python tools/build_public_source.py` from its root to create this archive; verify the generated external manifest checksum before extraction. GitHub-generated ZIPs have a different enclosing directory and are not direct substitutes for this cell. This cell refuses an existing extraction:

```python
from pathlib import Path, PurePosixPath
import zipfile

source_zip = Path("/content/glacial_lake_detection_source.zip")
source_dir = Path("/content/glacial_lake_detection")
with zipfile.ZipFile(source_zip) as archive:
    members = archive.infolist()
    names = [member.filename for member in members]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate source archive member")
    for name in names:
        parts = PurePosixPath(name).parts
        if (not parts or parts[0] != source_dir.name or len(parts) < 2
                or ".." in parts or "\\" in name or ":" in name):
            raise ValueError("Unexpected source archive path")
    source_dir.mkdir(exist_ok=False)
    for member in members:
        target = source_dir.joinpath(*PurePosixPath(member.filename).parts[1:])
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(archive.read(member))
```

Upload the authorized checkpoint and prepared input assets separately. The source package does not retrieve them from an owner's cloud account.

Run the included bootstrap in a fresh Python process:

```python
import subprocess, sys
subprocess.run([
    sys.executable, "/content/glacial_lake_detection/tools/bootstrap_inference.py",
    "--source", "/content/glacial_lake_detection",
    "--evidence", "/content/p56_setup"
], check=True)
```

It installs the CUDA 13.0 Torch pair from the official wheel index, resolves the remaining pinned stack, freezes the proposed dependency versions, installs the actual package, and checks imports/L4/BF16 support without constructing or running a model. Keep `p56_setup/environment.json`, `pip_plan.json` and `constraints.txt`. P63 accepts a clean `pip check`, or return code 1 with only `ipython 7.34.0 requires jedi, which is not installed.` and no unexpected stderr. That exact historical base exception is recorded as `ready_with_known_base_exception` only after the readiness probe succeeds; a clean successful setup is `ready`. Other nonzero results and probe failures stop setup with failed diagnostics. No automatic jedi repair occurs. Run bootstrap before importing the scientific libraries in the notebook; prediction below also uses a fresh process.

| Distribution | Required version |
|---|---|
| torch / torchvision | 2.14.1+cu130 / 0.29.1+cu130 |
| TerraTorch | 1.2.13, commit `f967ef77b00fe9097157a29d1dad4fda93aba4f9` |
| Lightning / timm | 2.6.6 / 1.0.31 |
| NumPy / Rasterio / SciPy | 2.4.6 / 1.5.2 / 1.16.3 |

The exact upstream pin and notices are in [third-party notices](../THIRD_PARTY_NOTICES.md). CPU inspection still uses the lightweight installation in the [quickstart](quickstart.md); `--help` and `predict --help` do not import Torch.

## Input manifest

Prepare the files through the [target-free inference preparation branch](reproduction.md#target-free-inference-files), keeping the native stored-value domain. The following names are user-supplied examples, not bundled assets. Keep relative paths portable with forward slashes; the `/content` parent simply locates them in this Colab example:

```text
/content/EMA12.ckpt
/content/input_manifest.json
/content/inputs/tile_a.tif
/content/inputs/tile_a_dem.tif
/content/inputs/tile_a_usable.tif
/content/inputs/tile_b.tif, tile_b_dem.tif, tile_b_usable.tif
/content/inputs/tile_c.tif, tile_c_dem.tif, tile_c_usable.tif
```

Create a JSON list with exactly these four fields per row:

```json
[
  {"id": "tile_a", "image": "inputs/tile_a.tif", "dem": "inputs/tile_a_dem.tif", "usability": "inputs/tile_a_usable.tif"},
  {"id": "tile_b", "image": "inputs/tile_b.tif", "dem": "inputs/tile_b_dem.tif", "usability": "inputs/tile_b_usable.tif"},
  {"id": "tile_c", "image": "inputs/tile_c.tif", "dem": "inputs/tile_c_dem.tif", "usability": "inputs/tile_c_usable.tif"}
]
```

Relative paths resolve against the manifest directory. Absolute local paths are also accepted. URLs and GDAL virtual paths are rejected. IDs must be unique even ignoring case, start with an ASCII letter/digit, contain only letters/digits/underscore/hyphen/dot, be at most 160 characters, and be portable file stems. Labels or extra fields are rejected by this prediction schema. The older generic `check-manifest` command provides only routing checks and has a looser schema; it does not qualify raster inputs.

Images must have exactly 11 bands and the recorded native grid. Selected zero-based slots are **0,1,2,3,4,5,9,10**. The first six selected bands form `S2L2A`, the final two `S1GRD`; frozen TRAIN statistics normalize stored values in float32. Band names alone do not establish physical SAR interpretation.

DEM and usability are single-band GeoTIFFs with exactly the image dimensions, CRS and affine. DEM values are physical metres, already on that grid, with finite valid support. Do not pass a normalized DEM. Derive usability from genuine source coverage/masks and finite data; never manufacture an all-one raster to conceal missing coverage. Usability must actually be 1 everywhere for this qualified scope; incomplete tiles are rejected. The runtime computes `(metres.astype(float64) - 651.663) / 928.168` and then casts to float32. See [target-free DEM handoff and vertical reference](reproduction.md#target-free-inference-files). No reprojection or scaling is performed by prediction. Labels, target counts, training NPZs, `sidecar_index.json` and `NativeDEMDataset` are not prediction requirements.

## Run prediction

With a manifest at `/content/input_manifest.json`, an unchanged checkpoint at `/content/EMA12.ckpt`, and a new output directory:

```python
import subprocess, sys
subprocess.run([
    sys.executable, "-m", "glacial_lake", "predict",
    "--checkpoint", "/content/EMA12.ckpt",
    "--manifest", "/content/input_manifest.json",
    "--output-dir", "/content/p56_outputs"
], check=True)
```

The equivalent installed command is:

```console
glacial-lake predict --checkpoint EMA12.ckpt --manifest input_manifest.json --output-dir p56_outputs
```

For application integration, use the API instead of running the CLI again:

```python
from glacial_lake.native_inference import Predictor
receipt = Predictor("EMA12.ckpt").predict_many("input_manifest.json", "p56_outputs")
```

Rows retain their supplied order. Consecutive rows form physical pairs; an odd final row runs alone. There is no sorting, padding, duplicate tail or batch-size option. Changing BF16 batch context can change thresholded pixels, so this policy does not establish arbitrary-batch or arbitrary-single-image equivalence.

The CLI first runs the shared CPU preparation: frozen resource checks, strict manifest/asset/raster validation and output-destination checks. It constructs the predictor once only after preparation succeeds. The API uses the same preparation for each job and supports reusing one `Predictor` across jobs with new output directories. Preparation retains metadata, not all normalized tensors. Execution rechecks manifest/config identities and rereads each physical batch, rejecting changed input records before forward.

Parameters remain FP32; evaluation uses `torch.inference_mode()` and CUDA BF16 autocast, converts logits to FP32, and labels lake at **logit >= 0**. No postprocessing, threshold tuning, training or EMA updates are performed. Relevant backend defaults are recorded without changing them.

## Outputs and resource release

The output directory must not exist. For the three-row example, successful output has this layout:

```text
p56_outputs/
  tile_a.tif
  tile_b.tif
  tile_c.tif
  receipt.json
```

It contains one `<id>.tif` per row and `receipt.json`. Each TIFF has one uint8 band, the original CRS/affine/dimensions, 0=background, 1=lake and 255 reserved for input nodata. The accepted full-support contract produces no nodata pixels. Each file is read back to verify exact pixels and spatial metadata.

The receipt records selected state before/after, package/config/source/environment hashes, input file and normalized-array hashes, manifest order, actual started/completed calls and views, batches, geometry, timestamps and output hashes. Early CPU preflight, configuration, checkpoint, environment/device or constructor errors can occur before any prediction receipt exists; the CLI reports them on stderr. Destination rechecks and atomic creation also precede the initial receipt. After the initial receipt is created, job exceptions are recorded with `status=failed`, the completed/started work and available partial outputs, provided the filesystem remains writable. Do not treat partial outputs as a complete run or overwrite them with a retry. Bootstrap's `p56_setup/environment.json` is a separate setup record; install failures before its dependency-check stage may have only subprocess diagnostics. Setup and per-call times are observations, not a latency benchmark.

| Failure | Stage and remedy |
|---|---|
| Missing/wrong checkpoint | Constructor, before prediction receipt: supply the exact original EMA12 file and verify its size/hash; do not substitute P53 or an initializer. |
| Unsupported versions/device | Bootstrap or constructor, before prediction receipt: use the recorded stack and L4; inspect setup diagnostics. A failed probe is not ready. |
| Malformed manifest, duplicate ID or missing asset | CPU preflight, before constructor/receipt: correct the strict four-field schema and local resolved paths. |
| Grid, finite-data or support mismatch | CPU preflight, before constructor/receipt: prepare correctly registered physical DEM/usability; reject incomplete tiles. |
| Existing output or parent that is a file | Preflight or execution-time guard, before receipt: choose a new writable directory under a directory parent. Never overwrite an earlier run. |
| Input changes after preparation | Batch reread, before that forward: inspect the failed partial receipt and preserve the inputs used; do not count unstarted work as completed. |

After a successful job, this cell checks completion, ordered work counts, selected state and output file hashes, then archives only the masks, receipt and three setup records. A partial job is rejected. It does not include checkpoints, raw inputs, environments or caches:

```python
from pathlib import Path
import hashlib, json, zipfile

output_dir = Path("/content/p56_outputs")
setup_dir = Path("/content/p56_setup")
manifest = json.loads(Path("/content/input_manifest.json").read_text())
receipt = json.loads((output_dir / "receipt.json").read_text())
order = [row["id"] for row in manifest]
groups = [order[start:start + 2] for start in range(0, len(order), 2)]
if not (order and receipt["status"] == "complete" and receipt["order"] == order
        and receipt["calls_started"] == receipt["calls_completed"] == len(groups)
        and receipt["views_started"] == receipt["views_completed"] == len(order)
        and [batch["ids"] for batch in receipt["batches"]] == groups
        and [batch["size"] for batch in receipt["batches"]] == [len(g) for g in groups]
        and receipt["named_state_before"] == receipt["named_state_after"]
        == "902ef093300c15d95a7a27fd3771be27c25b7aeae0a0e9ed47b3486025f7345c"
        and [item["id"] for item in receipt["outputs"]] == order):
    raise ValueError("Prediction job is incomplete or inconsistent; retain partial evidence")
payload = [(output_dir / "receipt.json", "outputs/receipt.json")]
for ident, item in zip(order, receipt["outputs"]):
    name = item["file"]
    if name != ident + ".tif" or Path(name).name != name:
        raise ValueError("Unexpected output filename")
    path = output_dir / name
    data = path.read_bytes()
    if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
        raise ValueError("Output differs from receipt")
    payload.append((path, "outputs/" + name))
environment = json.loads((setup_dir / "environment.json").read_text())
if environment["status"] not in {"ready", "ready_with_known_base_exception"}:
    raise ValueError("Setup was not ready")
for name in ("environment.json", "pip_plan.json", "constraints.txt"):
    path = setup_dir / name
    if not path.is_file():
        raise FileNotFoundError(path)
    payload.append((path, "setup/" + name))
result_zip = Path("/content/p56_result_bundle.zip")
with zipfile.ZipFile(result_zip, "x", compression=zipfile.ZIP_DEFLATED) as archive:
    for path, name in payload:
        archive.write(path, name)
print(result_zip, result_zip.stat().st_size, "bytes")
```

Download that checked bundle and confirm it is saved locally before release:

```python
from google.colab import files
files.download(str(result_zip))
```

Then disconnect/delete the task runtime through Colab's Runtime menu, or run the native release call:

```python
from google.colab import runtime
runtime.unassign()
```

Use Manage sessions to confirm that your task session is absent, without terminating unrelated sessions.

The private qualification harness prohibited network connections and access outside the clean inference assets. It recorded one blocked socket connection attempt (caller/destination were not captured), zero permitted connections, and no target/reference reads. No initializer or other model download occurred. The package sets upstream offline flags; the external qualification guard is not a public network sandbox.
