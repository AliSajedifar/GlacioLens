"""P56 CPU uncertainty: existing integer-weight resampling, new candidate counts only.
Inherited source SHA256 80b0e9452d6409cc2f93408056ee5575d1dd99a7994c4bda33d0d217ba28bc09.
"""
import numpy as np

def metric_percent(counts):
    tp, fp, fn, tn = np.moveaxis(counts.astype(np.float64), -1, 0)
    lake = tp / (tp + fp + fn)
    background = tn / (tn + fp + fn)
    return np.stack((lake, (lake + background) / 2), axis=-1) * 100

def percentile_interval(values):
    return np.quantile(values, [0.025, 0.975], method="linear").tolist()

def paired_bootstrap(unit_counts, point, names, seed, replicates):
    rng = np.random.default_rng(seed)
    n_units, n_models, _ = unit_counts.shape
    flat = unit_counts.reshape(n_units, n_models * 4)
    sampled = np.empty((replicates, n_models, 2), dtype=np.float64)
    probabilities = np.full(n_units, 1.0 / n_units)
    for start in range(0, replicates, 100):
        stop = min(start + 100, replicates)
        weights = rng.multinomial(n_units, probabilities, size=stop - start)
        pooled = (weights @ flat).reshape(stop - start, n_models, 4)
        sampled[start:stop] = metric_percent(pooled)
    out = {"units": n_units, "seed": seed, "replicates": replicates, "models": {}, "paired_P56_minus": {}}
    for j, name in enumerate(names):
        out["models"][name] = {
            metric: {"point_pct": float(point[j, k]), "percentile_95_pct": percentile_interval(sampled[:, j, k])}
            for k, metric in enumerate(("lake_iou", "two_class_miou"))
        }
        if j:
            delta = sampled[:, 0] - sampled[:, j]
            out["paired_P56_minus"][name] = {
                metric: {
                    "point_pp": float(point[0, k] - point[j, k]),
                    "percentile_95_pp": percentile_interval(delta[:, k]),
                    "positive_replicate_fraction": float(np.mean(delta[:, k] > 0)),
                }
                for k, metric in enumerate(("lake_iou", "two_class_miou"))
            }
    return out
