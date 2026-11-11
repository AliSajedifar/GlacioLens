"""Rebuild published GlacioLens TEST comparisons from saved masks on CPU.

Run from a source checkout: python tools/reproduce_results.py --output artifacts/reproduced_results
No torch import, model weights, model call, raw imagery or GPU is required.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / 'src'))
from glacial_lake import uncertainty
from object_boundary_metrics import tile_statistics, add_statistics, finalize_statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(actual, expected, path='', residuals=None):
    residuals = [] if residuals is None else residuals
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), 'Keys differ at ' + path)
        for k in expected:
            compare(actual[k], expected[k], path + '/' + k, residuals)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), 'Length differs at ' + path)
        for i, (a, e) in enumerate(zip(actual, expected)):
            compare(a, e, path + '/' + str(i), residuals)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        require(isinstance(actual, (int, float)), 'Non-numeric output at ' + path)
        difference = abs(actual - expected)
        residuals.append(float(difference))
        require(difference == 0 if isinstance(expected, int) else difference <= 1e-10,
                f'Result mismatch at {path}: {actual} versus {expected}')
    else:
        require(actual == expected, 'Value differs at ' + path)
    return residuals


def write_comparison_tables(summaries, bootstrap, names, images, output):
    comparison = []
    for model in names:
        s = summaries[model]
        tiny, small = s['size']['complete']['16-63'], s['size']['complete']['64-255']
        comparison.append(dict(model=model, images=images,
            two_class_miou_pct=s['pixel_scores_pct']['two_class_miou'],
            lake_iou_pct=s['pixel_scores_pct']['lake_iou'],
            object_f1_iou50_pct=100*s['matching']['0.5']['F1'],
            tiny_gt_count=tiny['objects'], tiny_coverage_ge50_count=tiny['coverage_ge50'],
            small_gt_count=small['objects'], small_coverage_ge50_count=small['coverage_ge50'],
            boundary_f1_pct=100*s['boundary']['F1'],
            zero_overlap_false_objects=s['false_objects']['count'],
            zero_overlap_false_object_area_px=s['false_objects']['area']))
    with (output / 'matched_reference_comparison.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison[0])); writer.writeheader(); writer.writerows(comparison)
    intervals = []
    for regime, b in bootstrap.items():
        for model in names:
            for metric, value in b['models'][model].items():
                intervals.append(dict(regime=regime, units=b['units'], comparison=model,
                    metric=metric, scale='percent', point=value['point_pct'],
                    lower=value['percentile_95_pct'][0], upper=value['percentile_95_pct'][1]))
        for model, values in b['paired_candidate_minus_reference'].items():
            for metric, value in values.items():
                intervals.append(dict(regime=regime, units=b['units'], comparison=names[0]+' minus '+model,
                    metric=metric, scale='percentage_points', point=value['point_pp'],
                    lower=value['percentile_95_pp'][0], upper=value['percentile_95_pp'][1]))
    with (output / 'bootstrap_intervals.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(intervals[0])); writer.writeheader(); writer.writerows(intervals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=REPOSITORY / 'results/reproduction')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    data = args.data_dir.resolve()
    config = json.loads((data / 'metric_config.json').read_text(encoding='utf-8'))
    manifest = json.loads((data / 'source_manifest.json').read_text(encoding='utf-8'))
    require(config['schema_version'] == 1 and config['mask']['bitorder'] == 'big', 'Unsupported schema or bit order')
    require(config['components']['matching_thresholds'] == [0.25, 0.5]
            and config['components']['connectivity'] == 8
            and config['components']['tiny_gt_pixels'] == [16, 63]
            and config['components']['small_gt_pixels'] == [64, 255], 'Unsupported object convention')
    require(config['boundary']['erosion_kernel'] == [3, 3]
            and config['boundary']['safe_support_erosion_kernel'] == [7, 7]
            and config['boundary']['matching_dilation_kernel'] == [5, 5]
            and config['boundary']['erosion_border_value'] == 0, 'Unsupported boundary convention')
    require(config['bootstrap']['batch_replicates'] == 100
            and config['bootstrap']['quantile_method'] == 'linear'
            and config['bootstrap']['quantiles'] == [0.025, 0.975], 'Unsupported bootstrap convention')
    for relative, receipt in manifest['distributed_files'].items():
        path = (data / relative).resolve()
        require(path.parent == data, 'Nonlocal data member')
        require(path.stat().st_size == receipt['bytes'] and sha(path) == receipt['sha256'],
                'Data identity mismatch: ' + relative)
    require(sha(Path(uncertainty.__file__)) == config['public_bootstrap_sha256'],
            'The recorded public bootstrap module bytes differ')
    names = config['model_order']
    with np.load(data / 'test_masks.npz', allow_pickle=False) as z:
        require(set(z.files) == {'ids', 'gt', 'valid', *names}, 'Unexpected NPZ members')
        ids = z['ids'].tolist()
        require(all(isinstance(s, str) for s in ids), 'Non-string IDs')
        require(len(ids) == len(set(ids)) == config['images'], 'Incorrect or duplicate IDs')
        arrays = {k: z[k] for k in ('gt', 'valid', *names)}
    for key, value in arrays.items():
        require(value.dtype == np.uint8 and value.shape == (len(ids), 256, 32), 'Mask layout: ' + key)
    with (data / 'group_mapping.csv').open(newline='', encoding='utf-8') as f:
        groups = list(csv.DictReader(f))
    require([r['tile_id'] for r in groups] == ids, 'Group row order differs')
    labels = [r['group_id'] for r in groups]
    require(labels == [sid.rsplit('_', 1)[0] for sid in ids], 'Group definition differs')
    unique, inverse = np.unique(labels, return_inverse=True)
    require(len(unique) == config['bootstrap']['group_units'], 'Group population differs')
    require([int(r['group_index']) for r in groups] == inverse.tolist(), 'Group index differs')
    with (data / 'per_tile_counts.csv').open(newline='', encoding='utf-8') as f:
        retained_rows = list(csv.DictReader(f))
    count_fields = ('tp', 'fp', 'fn', 'tn')
    require(len(retained_rows) == len(ids) * len(names), 'Count row population differs')
    total_counts = np.empty((len(ids), len(names), 4), np.int64)
    summaries, rebuilt_rows = {}, []
    for j, model in enumerate(names):
        total = {}
        for i, sid in enumerate(ids):
            y, v, p = (np.unpackbits(arrays[k][i], axis=-1, bitorder='big').astype(bool)
                       for k in ('gt', 'valid', model))
            row = tile_statistics(y, p, v)
            add_statistics(total, row)
            total_counts[i, j] = [row['counts'][k] for k in count_fields]
            rebuilt_rows.append(dict(model=model, tile_id=sid, **row['counts']))
        summaries[model] = finalize_statistics(total)
        pct = uncertainty.metric_percent(total_counts[:, j].sum(0))
        summaries[model]['pixel_scores_pct'] = dict(lake_iou=float(pct[0]), two_class_miou=float(pct[1]))
        print(f'Computed {model}: {summaries[model]["pixel_scores_pct"]}', flush=True)
    for actual, retained in zip(rebuilt_rows, retained_rows):
        require(actual['model'] == retained['model'] and actual['tile_id'] == retained['tile_id'], 'Count row identity differs')
        require(all(actual[k] == int(retained[k]) for k in count_fields), 'Count row mismatch')
    group_counts = np.zeros((len(unique), len(names), 4), np.int64)
    np.add.at(group_counts, inverse, total_counts)
    point = uncertainty.metric_percent(total_counts.sum(0))
    bootstrap = {}
    for regime, units in (('tile', total_counts), ('filename_source_group', group_counts)):
        value = uncertainty.paired_bootstrap(units, point, names,
                    config['bootstrap']['seed'], config['bootstrap']['replicates'])
        value['paired_candidate_minus_reference'] = value.pop('paired_P56_minus')
        bootstrap[regime] = value
    result = dict(metrics=summaries, uncertainty=bootstrap)
    expected = json.loads((data / 'expected_results.json').read_text(encoding='utf-8'))
    residuals = compare(result, expected)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'reconstructed_results.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    with (args.output / 'reconstructed_per_tile.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'tile_id', *count_fields]); writer.writeheader(); writer.writerows(rebuilt_rows)
    write_comparison_tables(summaries, bootstrap, names, len(ids), args.output)
    verification = dict(status='matched_retained_results', images=len(ids), models=names,
        count_rows=len(rebuilt_rows), group_units=len(unique), numeric_values_compared=len(residuals),
        max_absolute_numeric_residual=max(residuals, default=0.),
        integer_comparison='exact', floating_tolerance=1e-10,
        numpy=np.__version__, scipy=scipy.__version__,
        existing_bootstrap_sha256=config['public_bootstrap_sha256'],
        elapsed_seconds=time.perf_counter()-start, model_calls=0, gpu_allocations=0)
    (args.output / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
