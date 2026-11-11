"""Render the repository social preview from the verified GLB RGB and saved mask.

Uses the first selected showcase case, its frozen display stretch and exact
saved prediction. This is CPU-only presentation rendering, not inference.
Requires NumPy, Pillow, Matplotlib (bundled fonts) and tifffile.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-root', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=root / 'docs/figures/glaciolens_qualitative.json')
    parser.add_argument('--masks', type=Path, default=root / 'results/reproduction/test_masks.npz')
    parser.add_argument('--metrics', type=Path, default=root / 'results/metrics.csv')
    parser.add_argument('--output', type=Path, default=root / 'docs/figures/glaciolens_social.png')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()

    import numpy as np
    import PIL
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib
    import tifffile

    config = json.loads(args.config.read_text(encoding='utf-8'))
    with args.metrics.open(encoding='utf-8', newline='') as stream:
        test_rows = [row for row in csv.DictReader(stream) if row['cohort'] == 'TEST']
    require(len(test_rows) == 1, 'Expected one TEST metric row')
    test_row = test_rows[0]
    require(int(test_row['images']) == 1913, 'Unexpected TEST population')
    class_miou = float(test_row['two_class_miou_percent'])
    lake_iou = float(test_row['lake_iou_percent'])
    require(f'{class_miou:.2f}' == '87.84' and f'{lake_iou:.2f}' == '76.76',
            'Headline does not match the documented release metrics')
    require(config['schema_version'] == 2, 'Unsupported showcase configuration')
    case = config['cases'][0]
    require(case['id'] == 'GL_S2A_22WDT_20200717_0_L2A_146', 'Unexpected selected case')
    source_root = args.image_root.resolve()
    image_path = (source_root / case['source_image_member']).resolve()
    require(image_path.is_relative_to(source_root), 'Source path escapes image root')
    require(sha256(image_path) == case['image_sha256'], 'Source image hash differs')
    require(sha256(args.masks) == config['mask_bundle']['sha256'], 'Saved-mask bundle hash differs')
    image = tifffile.imread(image_path)
    require(image.shape == (256, 256, 11), 'Unexpected source image shape')

    with np.load(args.masks, allow_pickle=False) as masks:
        ids = masks[config['mask_bundle']['ids_key']].tolist()
        require(len(ids) == len(set(ids)), 'Duplicate prediction IDs')
        index = ids.index(case['id'])
        decoded = {}
        for role in ('prediction', 'target', 'valid'):
            packed = masks[config['mask_bundle'][role + '_key']][index]
            require(packed.shape == (256, 32) and packed.dtype == np.uint8,
                    'Unexpected packed-mask shape or type')
            require(hashlib.sha256(packed.tobytes()).hexdigest() == case[role + '_packed_sha256'],
                    role + ' mask identity differs')
            decoded[role] = np.unpackbits(packed, axis=-1, bitorder='big').astype(bool)
    pred, gt, valid = (decoded[key] for key in ('prediction', 'target', 'valid'))
    require(valid.all(), 'Selected case must have full valid support')
    counts = {'tp': int((pred & gt).sum()), 'fp': int((pred & ~gt).sum()),
              'fn': int((~pred & gt).sum()), 'tn': int((~pred & ~gt).sum())}
    require(counts == case['counts'], 'Selected-case counts differ')

    rgb = image[..., config['rgb_zero_based_slots']].astype(np.float64)
    require(np.isfinite(rgb).all(), 'Nonfinite RGB')
    low, high = np.asarray(case['rgb_low']), np.asarray(case['rgb_high'])
    require(np.all(high > low), 'Invalid display stretch')
    rgb = np.clip((rgb - low) / (high - low), 0.0, 1.0)
    # Preserve the native mask. Color its actual pixels and inner pixel boundary.
    padded = np.pad(pred, 1, constant_values=False)
    interior = np.ones_like(pred)
    for dy in range(3):
        for dx in range(3):
            interior &= padded[dy:dy + 256, dx:dx + 256]
    boundary = pred & ~interior
    overlay = rgb.copy()
    fill = np.asarray([34, 211, 238], dtype=np.float64) / 255
    edge = np.asarray([165, 243, 252], dtype=np.float64) / 255
    overlay[pred] = 0.72 * overlay[pred] + 0.28 * fill
    overlay[boundary] = 0.06 * overlay[boundary] + 0.94 * edge

    canvas = Image.new('RGB', (1280, 640), '#0B1C29')
    draw = ImageDraw.Draw(canvas)
    font_dir = Path(matplotlib.get_data_path()) / 'fonts/ttf'
    font_files = {'regular': font_dir / 'DejaVuSans.ttf', 'bold': font_dir / 'DejaVuSans-Bold.ttf'}
    def font(size, bold=False):
        return ImageFont.truetype(str(font_files['bold' if bold else 'regular']), size)
    def text(x, y, value, size, color='#F3FAFC', bold=False):
        draw.text((x, y), value, font=font(size, bold), fill=color, anchor='lt')

    text(48, 24, 'GlacioLens', 54, bold=True)
    text(50, 88, 'Multimodal glacial lake mapping with spatial refinement', 23, '#BCD3DF')
    text(48, 132, 'SATELLITE IMAGE', 16, '#D3E5EB', bold=True)
    text(504, 132, 'MODEL PREDICTION', 16, '#D3E5EB', bold=True)
    draw.rectangle((793, 133, 807, 147), fill='#22D3EE')
    text(815, 132, 'Predicted lake', 15, '#BCD3DF')
    for x, array in [(48, rgb), (504, overlay)]:
        tile = Image.fromarray(np.rint(array * 255).astype(np.uint8))
        tile = tile.resize((430, 430), Image.Resampling.NEAREST)
        canvas.paste(tile, (x, 160))
        draw.rectangle((x - 1, 159, x + 430, 590), outline='#53707E', width=1)

    draw.rounded_rectangle((966, 130, 1232, 591), radius=16, fill='#143140')
    text(989, 155, 'RECORDED GLB TEST', 16, '#A6CAD8', bold=True)
    text(989, 183, '1,913 tiles', 18, '#E1EFF4')
    text(986, 239, f'{class_miou:.2f}%', 47, bold=True)
    text(989, 294, 'Two-class mIoU', 18, '#C2DCE6')
    draw.line((989, 335, 1209, 335), fill='#385460', width=1)
    text(986, 363, f'{lake_iou:.2f}%', 47, bold=True)
    text(989, 419, 'Lake IoU', 18, '#C2DCE6')
    text(989, 494, 'Public weights', 19, '#8FE7E1', bold=True)
    text(989, 528, 'CPU-reproducible', 18, '#C2DCE6')
    text(989, 554, 'evaluation', 18, '#C2DCE6')
    text(48, 612, 'Selected successful example · GLB / Kaushik et al. · CC BY 4.0 · GlacioLens prediction', 13, '#AAC3CF')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, format='PNG', optimize=True, compress_level=9)
    require(args.output.stat().st_size < 1_000_000, 'Social preview must remain below 1 MB')
    receipt = {
        'schema_version': 1,
        'model': config['model'],
        'image': {'path': 'docs/figures/glaciolens_social.png', 'width': 1280, 'height': 640,
                  'bytes': args.output.stat().st_size, 'sha256': sha256(args.output)},
        'renderer': {'path': 'tools/render_social_preview.py', 'sha256': sha256(__file__),
                     'numpy': np.__version__, 'pillow': PIL.__version__, 'matplotlib': matplotlib.__version__,
                     'tifffile': tifffile.__version__,
                     'font_sha256': {k: sha256(v) for k, v in font_files.items()}},
        'source': {'case_id': case['id'], 'showcase_config': 'docs/figures/glaciolens_qualitative.json',
                   'showcase_config_sha256': sha256(args.config), 'image_sha256': case['image_sha256'],
                   'mask_bundle_sha256': config['mask_bundle']['sha256'],
                   'prediction_packed_sha256': case['prediction_packed_sha256'],
                   'target_packed_sha256': case['target_packed_sha256'], 'counts': counts},
        'display': {'native_shape': [256, 256], 'full_extent': True, 'rendered_panel_shape': [430, 430],
                    'resize': 'nearest', 'rgb_zero_based_slots': config['rgb_zero_based_slots'],
                    'rgb_low': case['rgb_low'], 'rgb_high': case['rgb_high'],
                    'prediction_fill_alpha': 0.28, 'prediction_inner_boundary_alpha': 0.94,
                    'scientific_mask_modified': False, 'generative_imagery': False},
        'headline_metrics': {'population': 'recorded GLB TEST', 'images': 1913,
                             'two_class_miou_percent': class_miou, 'lake_iou_percent': lake_iou,
                             'scope': 'Pooled dataset metrics; not selected-image metrics',
                             'source': 'results/metrics.csv', 'source_sha256': sha256(args.metrics)},
        'attribution': config['attribution'],
        'github_requirements': 'https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/customizing-your-repositorys-social-media-preview',
        'model_calls': 0, 'gpu_allocations': 0,
    }
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'bytes': args.output.stat().st_size,
                      'sha256': sha256(args.output), 'source_counts': counts}, indent=2))


if __name__ == '__main__':
    main()
