"""Render the documented GlacioLens showcase from public saved masks and source RGB.

Requires NumPy, Matplotlib, Pillow and tifffile. No model or GPU is used.
The configuration fixes source-image hashes, native geometry, display stretch,
mask identities, counts and matched crop coordinates. Example:

python tools/render_qualitative.py --image-root /path/to/extracted-dataset \
    --masks results/reproduction/test_masks.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def grid_from_tiff(page):
    tags = page.tags
    scale = tags['ModelPixelScaleTag'].value
    tie = tags['ModelTiepointTag'].value
    require(tie[:3] == (0.0, 0.0, 0.0), 'Unsupported non-origin tie point')
    keys = tags['GeoKeyDirectoryTag'].value
    epsg = None
    for i in range(4, len(keys), 4):
        key, location, count, value = keys[i:i + 4]
        if key == 3072 and location == 0 and count == 1:
            epsg = int(value)
    return {'epsg': epsg, 'transform': [scale[0], 0.0, tie[3], 0.0, -scale[1], tie[4]],
            'height': page.imagelength, 'width': page.imagewidth}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--config', type=Path, default=root / 'docs/figures/glaciolens_qualitative.json')
    parser.add_argument('--image-root', type=Path, required=True,
                        help='Directory containing the original Glacial_Lake_Bench/img_dir/test tree')
    parser.add_argument('--masks', type=Path, default=root / 'results/reproduction/test_masks.npz')
    parser.add_argument('--output', type=Path, default=root / 'docs/figures/glaciolens_qualitative.png')
    parser.add_argument('--receipt', type=Path, help='Optional rendering/count-verification receipt')
    args = parser.parse_args()

    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    from PIL import Image
    import tifffile

    config = json.loads(args.config.read_text(encoding='utf-8'))
    require(config['schema_version'] == 2, 'Unsupported figure configuration')
    require(len(config['cases']) == 3, 'This layout requires exactly three cases')
    source_root = args.image_root.resolve()
    colors = config['colors']

    with np.load(args.masks, allow_pickle=False) as masks:
        ids = masks[config['mask_bundle']['ids_key']].tolist()
        require(len(ids) == len(set(ids)), 'Duplicate prediction IDs')
        lookup = {name: i for i, name in enumerate(ids)}
        selected = []
        for case in config['cases']:
            ident = case['id']
            require(ident in lookup, f'Missing mask ID: {ident}')
            index = lookup[ident]
            decoded = {}
            for role in ('prediction', 'target', 'valid'):
                key = config['mask_bundle'][role + '_key']
                packed = masks[key][index]
                require(packed.shape == (256, 32) and packed.dtype == np.uint8,
                        f'Unexpected packed {role} array for {ident}')
                require(sha256(packed.tobytes()) == case[role + '_packed_sha256'],
                        f'{role} mask identity differs for {ident}')
                decoded[role] = np.unpackbits(packed, axis=-1, bitorder='big').astype(bool)
            pred, gt, valid = (decoded[x] for x in ('prediction', 'target', 'valid'))
            require(valid.all(), 'Showcase scope requires full recorded valid support')
            counts = {'tp': int((pred & gt & valid).sum()),
                      'fp': int((pred & ~gt & valid).sum()),
                      'fn': int((~pred & gt & valid).sum()),
                      'tn': int((~pred & ~gt & valid).sum())}
            require(counts == case['counts'], f'Saved counts differ for {ident}')

            image_path = (source_root / case['source_image_member']).resolve()
            require(image_path.is_relative_to(source_root), 'Image path escapes supplied root')
            raw = image_path.read_bytes()
            require(sha256(raw) == case['image_sha256'], f'Source image differs for {ident}')
            with tifffile.TiffFile(image_path) as source:
                image = source.asarray()
                geometry = grid_from_tiff(source.pages[0])
            require(image.shape == (256, 256, 11), f'Unexpected source shape for {ident}')
            require(geometry == case['geometry'], f'Unexpected source grid for {ident}')
            require(np.isfinite(image[..., [0, 1, 2, 3, 4, 5, 9, 10]]).all(),
                    f'Nonfinite model input values for {ident}')
            rgb = image[..., config['rgb_zero_based_slots']].astype(np.float64)
            low, high = np.asarray(case['rgb_low']), np.asarray(case['rgb_high'])
            require(np.all(high > low), 'Invalid display stretch')
            rgb = np.clip((rgb - low) / (high - low), 0.0, 1.0)

            crop = case['inset_xyxy']
            if crop:
                x0, y0, x1, y1 = crop
                require(0 <= x0 < x1 <= 256 and 0 <= y0 < y1 <= 256, 'Invalid crop')
                covered = np.zeros((256, 256), dtype=bool)
                covered[y0:y1, x0:x1] = True
                require(not ((gt | pred) & ~covered).any(),
                        'Tight inset would omit a lake/error pixel in this small-lake case')
                # Inset occupies the lower-right portion of the full panel.
                require(not (gt | pred)[130:256, 130:256].any(),
                        'Inset would obscure a full-panel lake/error pixel')
            union = counts['tp'] + counts['fp'] + counts['fn']
            selected.append((case, rgb, gt, pred, counts, 100 * counts['tp'] / union))

    def color_rgb(hex_color):
        return np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.uint8)

    def mask_display(mask):
        out = np.empty((256, 256, 3), dtype=np.uint8)
        out[:] = color_rgb(colors['background'])
        out[mask] = color_rgb(colors['tp'])
        return out

    matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 15})
    fig = plt.figure(figsize=(16, 1544 / 120), dpi=120, facecolor='white')
    lefts = [0.029, 0.273, 0.517, 0.761]
    width = 0.210
    height = width * 1920 / 1544
    tops = [0.930, 0.630, 0.330]
    headings = ['Satellite RGB', 'Ground truth', 'Prediction', 'Error map']
    for x, title in zip(lefts, headings):
        fig.text(x + width / 2, 0.985, title, ha='center', va='top',
                 fontsize=22, fontweight='bold', color='#172B3A')

    for row, (case, rgb, gt, pred, counts, iou) in enumerate(selected):
        top = tops[row]
        fig.text(lefts[0], top + 0.004, case['title'], fontsize=17, fontweight='bold',
                 color='#172B3A', va='bottom')
        fig.text(lefts[-1] + width, top + 0.004, f'Lake IoU {iou:.1f}%',
                 fontsize=15, color='#4B6070', ha='right', va='bottom')
        error = mask_display(np.zeros_like(gt))
        error[gt & pred] = color_rgb(colors['tp'])
        error[pred & ~gt] = color_rgb(colors['fp'])
        error[gt & ~pred] = color_rgb(colors['fn'])
        panels = [rgb, mask_display(gt), mask_display(pred), error]
        for column, (x, array) in enumerate(zip(lefts, panels)):
            ax = fig.add_axes([x, top - height, width, height])
            ax.imshow(array, interpolation='nearest')
            ax.set_axis_off()
            crop = case['inset_xyxy']
            if crop:
                x0, y0, x1, y1 = crop
                edge = 'white' if column == 0 else '#172B3A'
                ax.add_patch(Rectangle((x0 - .5, y0 - .5), x1-x0, y1-y0,
                                       fill=False, edgecolor=edge, linewidth=0.8))
                inset = ax.inset_axes([.51, .03, .46, .46])
                inset.imshow(array[y0:y1, x0:x1], interpolation='nearest')
                inset.set_xticks([]); inset.set_yticks([])
                for spine in inset.spines.values():
                    spine.set_color('#172B3A'); spine.set_linewidth(1.1)
                if column == 0:
                    inset.text(.5, 1.01, f'{x1-x0} × {y1-y0} crop', transform=inset.transAxes,
                               ha='center', va='bottom', fontsize=10.5, color='white',
                               bbox={'facecolor':'#172B3A','edgecolor':'none','pad':3})

    handles = [Patch(facecolor=colors[k], label=label) for k,label in
               [('tp','True positive'), ('fp','False positive'), ('fn','False negative')]]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .012),
               ncol=3, frameon=False, fontsize=16, columnspacing=2.8, handlelength=1.7)
    fig.canvas.draw()
    pixels = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(args.output, format='PNG', optimize=True, compress_level=9)
    plt.close(fig)
    receipt = {'figure': str(args.output.name), 'bytes': args.output.stat().st_size,
               'sha256': sha256(args.output.read_bytes()), 'width': int(pixels.shape[1]),
               'height': int(pixels.shape[0]), 'cases': [{'id':c['id'],'counts':n,
               'lake_iou_percent':v,'inset_xyxy':c['inset_xyxy']} for c,_,_,_,n,v in selected],
               'model_calls': 0, 'gpu_allocations': 0}
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
