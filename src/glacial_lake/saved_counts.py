"""CPU confusion counts from saved, big-endian packed binary masks."""
import numpy as np
from .losses_metrics import pooled_metrics


def packed_counts(prediction, target, valid):
    arrays = [np.asarray(a) for a in (prediction, target, valid)]
    if any(a.dtype != np.uint8 or a.ndim != 3 for a in arrays):
        raise ValueError("Expected uint8 [sample, height, packed_width] arrays")
    if len({a.shape for a in arrays}) != 1:
        raise ValueError("Prediction, target and valid shapes differ")
    p, y, v = arrays
    table = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    return np.stack([table[a].sum(axis=(1, 2), dtype=np.int64) for a in
                     (p & y & v, p & ~y & v, ~p & y & v, ~p & ~y & v)], axis=1)


def recount_file(path):
    with np.load(path, allow_pickle=False) as z:
        counts = packed_counts(z["prediction"], z["target"], z["valid"])
    columns = ["tp", "fp", "fn", "tn"]
    pooled = counts.sum(axis=0).tolist()
    return {"samples": len(counts), "columns": columns,
            "pooled": pooled, "metrics": pooled_metrics(dict(zip(columns, pooled))),
            "model_calls": 0}
