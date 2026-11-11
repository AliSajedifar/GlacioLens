"""CPU evaluation adapted from the retained GlacioLens diagnostic functions.

No model execution. Object matching, completeness and boundary conventions are
recorded in results/reproduction/metric_config.json and source_manifest.json.
"""
from collections import Counter

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


def tile_statistics(target, prediction, valid):
    """Return integer sufficient statistics, preserving the original definitions."""
    y, p, v = (np.asarray(x, dtype=bool) for x in (target, prediction, valid))
    if y.ndim != 2 or not (y.shape == p.shape == v.shape):
        raise ValueError('Target, prediction and valid support need equal 2-D grids')
    y, p = y & v, p & v
    labels, ngt = ndimage.label(y, np.ones((3, 3)))
    predicted, npred = ndimage.label(p, np.ones((3, 3)))
    ga = np.bincount(labels.ravel(), minlength=ngt + 1)[1:]
    pa = np.bincount(predicted.ravel(), minlength=npred + 1)[1:]
    overlap = np.bincount(
        (labels.astype(np.int64) * (npred + 1) + predicted).ravel(),
        minlength=(ngt + 1) * (npred + 1),
    ).reshape(ngt + 1, npred + 1)[1:, 1:]
    iou = overlap / (ga[:, None] + pa[None, :] - overlap) if ngt and npred else np.zeros((ngt, npred))
    matching = {}
    for threshold in (0.25, 0.5):
        eligible = iou >= threshold
        if ngt and npred:
            a, b = linear_sum_assignment(
                np.where(eligible, min(ngt, npred) + 1 + iou, 0.), maximize=True
            )
            hits = sum(bool(eligible[g, q]) for g, q in zip(a, b))
        else:
            hits = 0
        matching[str(threshold)] = dict(gt=int(ngt), predicted=int(npred), tp=hits,
                                        fp=int(npred)-hits, fn=int(ngt)-hits)

    invalid_neighbor = ndimage.binary_dilation(~v, structure=np.ones((3, 3)))
    edge = np.zeros_like(v)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    incomplete = set(np.unique(labels[edge])) | set(np.unique(labels[invalid_neighbor]))
    intersections = np.bincount(labels[p], minlength=ngt + 1)
    bins = ('1-15', '16-63', '64-255', '>=256')
    sizes = {scope: {k: Counter(objects=0, GT_area=0, TP_area=0, coverage_ge25=0,
                              coverage_ge50=0, omitted=0) for k in bins}
             for scope in ('all_visible', 'complete', 'incomplete')}
    for label, area in enumerate(ga, 1):
        area, intersection = int(area), int(intersections[label])
        name = '1-15' if area < 16 else '16-63' if area < 64 else '64-255' if area < 256 else '>=256'
        counts = dict(objects=1, GT_area=area, TP_area=intersection,
                      coverage_ge25=int(intersection / area >= .25),
                      coverage_ge50=int(intersection / area >= .5), omitted=int(intersection == 0))
        for scope in ('all_visible', 'incomplete' if label in incomplete else 'complete'):
            sizes[scope][name].update(counts)

    safe = ndimage.binary_erosion(v, np.ones((7, 7)), border_value=0)
    gb = (y & ~ndimage.binary_erosion(y, np.ones((3, 3)), border_value=0)) & safe
    pb = (p & ~ndimage.binary_erosion(p, np.ones((3, 3)), border_value=0)) & safe
    boundary = dict(gt_support=int(gb.sum()), pred_support=int(pb.sum()),
                    matched_pred=int((pb & ndimage.binary_dilation(gb, np.ones((5, 5)))).sum()),
                    matched_gt=int((gb & ndimage.binary_dilation(pb, np.ones((5, 5)))).sum()))
    overlap_per_pred = np.bincount(predicted[y], minlength=npred + 1)[1:]
    false = overlap_per_pred == 0
    false_objects = dict(count=int(false.sum()), area=int(pa[false].sum()))
    counts = dict(tp=int((p & y).sum()), fp=int((p & ~y).sum()),
                  fn=int((~p & y).sum()), tn=int((~p & ~y & v).sum()))
    return dict(counts=counts, matching=matching, boundary=boundary,
                size=sizes, false_objects=false_objects)


def add_statistics(total, row):
    """Accumulate integer leaves without averaging per-image ratios."""
    for key, value in row.items():
        if isinstance(value, dict):
            add_statistics(total.setdefault(key, {}), value)
        else:
            total[key] = total.get(key, 0) + int(value)


def finalize_statistics(total):
    for value in total['matching'].values():
        den = 2 * value['tp'] + value['fp'] + value['fn']
        value['F1'] = 2 * value['tp'] / den if den else None
    b = total['boundary']
    precision = b['matched_pred'] / b['pred_support'] if b['pred_support'] else None
    recall = b['matched_gt'] / b['gt_support'] if b['gt_support'] else None
    b['precision'], b['recall'] = precision, recall
    b['F1'] = ((2 * precision * recall / (precision + recall) if precision + recall else 0.)
               if precision is not None and recall is not None else None)
    return total
