"""Write a tiny deterministic saved-mask example, not trained-model predictions.

Two samples, each one row of eight bits. Only the first six pixels in each
sample are valid. Packed output has shape [2, 1, 1] and dtype uint8.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New output NPZ, for example example_output/masks.npz")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".npz":
        parser.error("Output must end in .npz")
    target = np.array([[[1, 1, 1, 0, 0, 0, 1, 0]],
                       [[1, 0, 0, 0, 0, 0, 1, 0]]], dtype=np.uint8)
    prediction = np.array([[[1, 1, 0, 1, 0, 0, 1, 1]],
                           [[1, 1, 0, 0, 0, 0, 0, 1]]], dtype=np.uint8)
    valid = np.array([[[1, 1, 1, 1, 1, 1, 0, 0]],
                      [[1, 1, 1, 1, 1, 1, 0, 0]]], dtype=np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting a user's existing mask archive.
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **{
            key: np.packbits(value, axis=-1, bitorder="big")
            for key, value in {"prediction": prediction, "target": target, "valid": valid}.items()
        })
    print(json.dumps({"output": args.output.as_posix(), "synthetic": True,
                      "samples": 2, "unpacked_shape": [2, 1, 8], "packed_shape": [2, 1, 1],
                      "expected_pooled_tp_fp_fn_tn": [3, 2, 1, 6],
                      "expected_lake_iou": 0.5, "expected_background_iou": 2 / 3,
                      "expected_two_class_miou": 7 / 12, "model_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
