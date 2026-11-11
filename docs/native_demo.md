# Run the real three-case example

This example supplies the exact original inputs and expected outputs from the three-case P61 NVIDIA L4 qualification of **GlacioLens-GLB-v1 / P56 EMA12 / step 11,484**. Download the small archive from the existing model repository using the source-side helper. No training labels are needed.

The archive contains three original 11-band images, three aligned physical DEMs, three genuine usability rasters, an ordered manifest, three retained prediction masks and data notices. All nine input TIFF containers, normalized input arrays and grids match the original qualification exactly. Publication was checked on CPU; this source snapshot was not newly executed on L4.

## Obtain and verify the example

Run from the repository root:

```sh
python tools/native_demo.py download
```

Public archive: [native_demo.zip](https://huggingface.co/MaxiWb/glaciolens-glb-v1/resolve/f9d9eab60fba1c7dd023cfcfd1cb51a4da53dd2c/examples/native_demo.zip) at revision `f9d9eab60fba1c7dd023cfcfd1cb51a4da53dd2c`, **6,608,007 bytes**, SHA256 `d936c393f827665f04f0fc55c47cc7a0fb9ba7d7d707fa53f31bebebe8a49661`.

The [download manifest](../examples/native_demo.json) fixes the immutable Hugging Face revision, archive SHA256 and every member identity. The helper verifies the whole archive before extracting to `examples/native_demo`, rejects unsafe or unexpected members, and refuses any existing destination. Use `--directory` to choose a new destination instead of overwriting your data. The archive is separate from the GitHub source and contains no checkpoint.

For input checks in a lightweight CPU environment:

```sh
python -m pip install -e .
python -m pip install rasterio
python tools/native_demo.py preflight
```

Preflight checks file hashes, order, genuine full support, geometry and exact normalized-array hashes. It does not construct or execute a model. A CPU preflight pass does not qualify the CPU environment for inference.

## Obtain the unchanged checkpoint and use the recorded L4 setup

Follow the [immutable checkpoint download](inference.md#checkpoint-download) and verify SHA256 `19bcdaa9f925da61ecc46aab755b265960c632aacdec582bd3050a6b84bd4230` (1,280,672,641 bytes). The checkpoint revision remains `bc65beeff3604d51e2242cb51536d5b51ff702f3`. Keep the downloaded file at `user_assets/glaciolens-glb-v1.ckpt`.

Select an available NVIDIA L4 runtime and use the existing [recorded installation route](inference.md#colab-installation), including `tools/bootstrap_inference.py` and its readiness evidence. L4 availability or free access is not guaranteed. The lightweight CPU installation above does not replace the strict inference stack. Obtain the source, example and checkpoint in that environment, and enter the extracted source directory before the following commands.

Run the full ordered job once in a fresh process; the output directory must not already exist:

```sh
glacial-lake predict --checkpoint user_assets/glaciolens-glb-v1.ckpt --manifest examples/native_demo/inputs.json --output-dir example_output/native_demo
```

| Position | Original TEST index | Tile ID | Physical batch | Expected lake pixels |
|---|---:|---|---|---:|
| 0 | 0 | ACS_S2A_17WPV_20200720_1_L2A_34 | First pair | 2,380 |
| 1 | 1 | ACS_S2A_17WPV_20200722_1_L2A_86 | First pair | 382 |
| 2 | 1912 | WC_S2B_9UYT_20200829_0_L2A_223 | Singleton tail | 11,472 |

Preserve this order and physical batches **[2,1]**. Do not sort the manifest, pad the tail, or run one member alone and assume BF16 parity. The original qualification used three views and two calls with zero differing pixels against retained P56 predictions. It is historical execution evidence, separate from the new CPU delivery checks.

## Compare the decoded outputs

```sh
python tools/native_demo.py compare --output-dir example_output/native_demo
```

The CPU comparison verifies all three decoded masks, CRS/affine/dimensions, uint8 encoding, nodata and valid support. Different TIFF container bytes are reported separately and do not imply different scientific pixels. The prediction `receipt.json` separately records actual order, batches, selected state, environment and calls. Preserve it with the masks and setup evidence; follow the [completion and runtime-release instructions](inference.md#outputs-and-resource-release).

## Data terms

The [example notices](../examples/native_demo_DATA_NOTICES.md) accompany the archive. GLB imagery retains CC BY 4.0 attribution. Physical DEM and genuine usability are retained adaptations of Copernicus GLO-30 Public COG data with the provider's prescribed adapted-data notice and applicable product terms. These data are not relicensed Apache-only. Source and selected weights retain Apache-2.0. No elevation, support or prediction was invented or regenerated for this package.
