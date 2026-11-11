# Exact dataset and sampling metadata

These files preserve the populations used for GlacioLens-GLB-v1: TRAIN **15,300**, VAL **1,912**, TEST **1,913** and the separate GLBC stress cohort **1,105**. They identify release records, not independent geographic sites. TRAIN and VAL retain their original `Glacial_Lake_Bench/...` prefixes; TEST and GLBC retain their original bare IDs. List order is the recorded canonical order, not an exposure-table sort or an epoch's shuffled training order.

From the repository root, verify every metadata file, population count, ordered-ID digest, source mapping and decoded supplementary pool/plan with Python 3.11 or newer:

```sh
python tools/verify_data_manifest.py
```

This command uses only the standard library and performs no model work or downloads. It validates the bundled metadata without claiming that you have the raw dataset. To additionally check presence of every original mapped image/label and all available raw-file SHA256 values:

```sh
python tools/verify_data_manifest.py --data-root path/to/extracted/dataset_parent
```

The parent directory should contain `Glacial_Lake_Bench/`, `GLB/` and `Glacial-Lake-Challenge/`. Six restored TEST members originate from the pinned Cryo-Bench distribution and retain its `GLB/test/images/` and `GLB/test/masks/` paths. All other GLB mappings retain the initial Zenodo archive's `img_dir/` and `ann_dir/` paths. Blank raw-file hashes are unavailable, never an implied match. The optional check reports exactly how many raw files were byte-verified.

## Files and identities

| File | Meaning |
|---|---|
| `train_ids.json`, `val_ids.json`, `test_ids.json`, `glbc_ids.json` | Exact ordered canonical ID lists. |
| `release.json` | Public source revisions/archive identities, cohort counts, historical logical digests, newly exported file-byte hashes, coverage and count notes. |
| `source_members.csv` | One row per ordered record: split, zero-based index, original ID, archive-relative image/label paths and known raw-file SHA256 values. No array, grid or archive hash is mislabeled as a raw raster hash. |
| `split_aliases.csv` | Retained cross-split image-byte, label-byte and equal-native-grid relations where supported, plus the six release-restored TEST records. Relations are distinct annotations; no records are removed. This is not a complete spatial-overlap inventory, and GLBC remains dependent. |
| `auxiliary_pool.json`, `auxiliary_plan.json` | Losslessly indexed copies of the exact materialized supplementary pool and 24-epoch schedule: 91,824 views. |

Logical digests use `sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))`, with default `ensure_ascii=True` and no final newline. List order is retained. File-byte hashes cover the exported bytes, including their newline, and are deliberately separate. `release.json` does not contain a self-referential byte hash.

The pool table declares its column order. `train_ordered_index` references `train_ids.json`; each row retains the original role, realized cell and complete anchor object. The schedule preserves nested epochs → updates → microbatch groups → ordered records. Each record references its pool object and retains requested cell, flip, quarter-turn count and an optional crop origin. Epoch and update integers come from their original nesting positions. `load_sampling()` in the verifier reconstructs the original dictionaries and checks both historical digests:

- Pool: `b3b7f54e5975d8bdc8add750fa9225a3e033364f69f5c6b774baed66f4e77502`.
- Plan: `2e0bfe01a68804b7d54c3106cf70abfbfafa41e50742f02474b05a085fc75b7d`.

No new sampling plan was generated. Component boxes, areas and crop origins are label-derived recipe metadata, not raster masks. This export does not include raw satellite imagery, labels, DEM arrays, native epoch shuffles, optimizer states or a qualified portable training launcher. The original pool/plan builders remain in the recorded scientific source; the decoder reconstructs these particular accepted materializations.

## Sources, count note and attribution

The initial Zenodo GLB archive supplied 19,119 records; six restored TEST records produce 19,125. Cryo-Bench v5 Table II reports the same 15,300 / 1,912 / 1,913 split. The dataset-description preprint's 19,115 headline differs by ten; its cause is not established. The six-record restoration explains only 19,119 → 19,125. See the [dataset provenance guide](../../docs/reproduction.md#dataset-version-and-split-sizes).

Dataset-derived memberships, source mappings, annotations and label-derived sampling metadata retain **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)** attribution: [Glacial-Lake-Bench, Zenodo record 17917359](https://zenodo.org/records/17917359), Saurabh Kaushik, Beth Tellman, Ian Howat, Umesh Haritashya and Lalit Maurya. The [dataset paper](https://doi.org/10.5194/essd-2026-474) is by Kaushik, Tellman, Howat and Haritashya. Adaptations by GlacioLens: selected population projections, portable paths, relationship annotations and lossless indexing. Apache-2.0 covers the verification code and does not replace the data license. No DEM product is redistributed here.
