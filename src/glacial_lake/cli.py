"""Lightweight CPU inspection with a separately loaded P56 prediction command."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="P56 research source, CPU inspection and native inference")
    sub = parser.add_subparsers(dest="command", required=True)
    recipe = sub.add_parser("inspect-recipe", help="Print recorded settings; does not train")
    recipe.add_argument("--config", required=True, type=Path)
    identity = sub.add_parser("inspect-identity", help="Print supplied scientific metadata")
    identity.add_argument("--file", required=True, type=Path)
    manifest = sub.add_parser("check-manifest", help="Validate local routing schema, without model calls")
    manifest.add_argument("manifest", type=Path)
    manifest.add_argument("--root", type=Path)
    manifest.add_argument("--check-files", action="store_true")
    counts = sub.add_parser("recount", help="Count saved mask pixels on CPU; report pooled counts and fractional metrics")
    counts.add_argument("packed_masks", type=Path)
    predict = sub.add_parser("predict", help="P56-native-r1: exact EMA12, local inputs, NVIDIA L4")
    predict.add_argument("--checkpoint", required=True, type=Path)
    predict.add_argument("--manifest", required=True, type=Path)
    predict.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "inspect-recipe":
        result = json.loads(args.config.read_text(encoding="utf-8"))
    elif args.command == "inspect-identity":
        result = json.loads(args.file.read_text(encoding="utf-8"))
    elif args.command == "check-manifest":
        from .manifest_check import check_manifest
        result = check_manifest(args.manifest, args.root, args.check_files)
    elif args.command == "predict":
        from .native_inference import Predictor, prepare_job
        job = prepare_job(args.manifest, args.output_dir)
        result = Predictor(args.checkpoint)._predict_prepared(job)
    else:
        from .saved_counts import recount_file
        result = recount_file(args.packed_masks)
    print(json.dumps(result, indent=2, allow_nan=False))
