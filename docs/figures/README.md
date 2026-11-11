# Recreate the qualitative figure

The [showcase](glaciolens_qualitative.png) displays selected successful examples from the released GlacioLens-GLB-v1 model. It uses the saved predictions and ground truth in the [results reproduction bundle](../../results/reproduction/), not a new inference run.

The three rows show a large lake, the original clear-lake case and the original small-lake case. Their exact identities, counts, source-file hashes, native grids and display settings are recorded in [the figure configuration](glaciolens_qualitative.json). These are illustrative selections, not a representative statistical sample.

## CPU commands

Run from the repository root in a Python environment with the plotting dependencies:

```sh
python -m pip install numpy matplotlib pillow tifffile
python tools/fetch_qualitative_inputs.py --output-root example_data
python tools/render_qualitative.py --image-root example_data --masks results/reproduction/test_masks.npz --output example_outputs/glaciolens_qualitative.png --receipt example_outputs/figure_receipt.json
```

The fetcher retrieves only the three source TIFF members from the public GLB ZIP using bounded HTTP byte ranges, checks their hashes and reuses matching local files. It refuses a whole-archive response. Readers who already have the extracted dataset can skip the fetcher and point `--image-root` at the directory containing `Glacial_Lake_Bench/img_dir/test/`.

The renderer uses the immutable mask identities in the configuration and checks TP/FP/FN/TN counts before drawing. RGB uses Red/Green/Blue stored slots 2/1/0 and fixed per-band 2nd–98th percentile bounds. All panels preserve the native 256 × 256 extent. The small-lake inset uses the same 32 × 32 crop in every column; it contains every target/prediction pixel in that case and covers no full-panel positive/error pixel. Labels, predictions and errors use nearest-neighbor display.

No model, checkpoint, GPU, DEM or training runtime is needed to reproduce the figure. The configuration records the plotting versions used for the published PNG; other plotting versions may change text rasterization or PNG bytes while the validated scientific masks and counts remain identical.

## Attribution and the earlier difficult case

Imagery and reference labels: [Glacial-Lake-Bench, Kaushik et al. (2026), Zenodo17917359](https://zenodo.org/records/17917359), [CC BY4.0](https://creativecommons.org/licenses/by/4.0/). The original data record credits Saurabh Kaushik, Beth Tellman, Ian Howat, Umesh Haritashya and Lalit Maurya. Display adaptations are RGB selection, contrast stretching and matched crop enlargement; prediction/error overlays are by GlacioLens. See [third-party notices](../../THIRD_PARTY_NOTICES.md).

The previously displayed missed-extent case remains available through its [immutable figure](https://raw.githubusercontent.com/AliSajedifar/GlacioLens/1bb1b6eb192bf491b2a9b09aa2c56c9a4efe6204/docs/figures/glaciolens_qualitative.png) and [case provenance](https://github.com/AliSajedifar/GlacioLens/blob/1bb1b6eb192bf491b2a9b09aa2c56c9a4efe6204/docs/figures/glaciolens_qualitative.json). The current showcase changes presentation, not the quantitative evaluation or documented limitations.

## Social preview

The [social preview](glaciolens_social.png) uses the same verified large-lake image and released prediction, with pooled TEST metrics displayed separately. Its [provenance](glaciolens_social.json) records the source arrays, display settings, headline values and attribution.

After obtaining the imagery with the fetcher above, recreate it on CPU:

```sh
python tools/render_social_preview.py --image-root example_data --output example_outputs/glaciolens_social.png --receipt example_outputs/social_preview_receipt.json
```

The renderer checks source and mask identities and reads the headline scores from `results/metrics.csv`. It uses the existing plotting dependencies and Matplotlib's bundled DejaVu fonts. Package or font versions can change rendered bytes without changing the checked scientific data. The original prediction is not smoothed; nearest-neighbor scaling is only for display. Dataset attribution and the existing third-party notices apply to this adapted figure.

The PNG is 1280 × 640 and under 1 MB. To use it as the repository link preview, upload it through GitHub **Settings → Social preview → Edit → Upload an image**; committing the file alone does not change that setting. See [GitHub's instructions](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/customizing-your-repositorys-social-media-preview).
