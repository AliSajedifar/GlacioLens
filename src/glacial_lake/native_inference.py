"""P56-native-r1: local checkpoint and local inputs, without targets or downloads.

Import this module only for prediction. The CPU inspection CLI stays lightweight.
"""
from pathlib import Path
from dataclasses import dataclass
import copy
import hashlib
import importlib
import importlib.metadata
import json
import os
import random
import sys
import time

from .common import hash_file, utc_now, write_json
from .native_input import read_input, read_manifest

CONFIG_SHA256 = "85b3fc5c1b17681f750ee909aa35e93a9d63470930ef83dbbacafe01d8c97521"


def _read_configuration():
    package = Path(__file__).parent
    config_path = package / "inference_config.json"
    if hash_file(config_path) != CONFIG_SHA256:
        raise ValueError("P56 inference configuration identity differs")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    normalization = package / "inference_normalization.json"
    if hash_file(normalization) != config["normalization_sha256"]:
        raise ValueError("P56 TRAIN normalization identity differs")
    return config, json.loads(normalization.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class _PreparedJob:
    manifest: Path
    manifest_sha256: str
    output_dir: Path
    rows: tuple
    inputs: tuple
    configuration_sha256: str
    normalization_sha256: str


def _output_destination(output_dir):
    path = Path(output_dir)
    if path.is_symlink() or path.exists():
        raise FileExistsError(path)
    path = path.resolve()
    parent = path.parent
    while not parent.exists():
        if parent == parent.parent:
            raise FileNotFoundError("Output root is unavailable: " + str(parent))
        parent = parent.parent
    if not parent.is_dir():
        raise NotADirectoryError(parent)
    return path


def prepare_job(manifest, output_dir):
    """Validate one CPU job without a model; retain metadata, never dataset tensors."""
    config, statistics = _read_configuration()
    manifest = Path(manifest).resolve(strict=True)
    manifest_sha256 = hash_file(manifest)
    rows = tuple(read_manifest(manifest))
    output_dir = _output_destination(output_dir)
    records = []
    for row in rows:
        _, _, _, record = read_input(row, statistics)
        records.append(record)
    if hash_file(manifest) != manifest_sha256 or tuple(read_manifest(manifest)) != rows:
        raise ValueError("Manifest changed during validation")
    return _PreparedJob(manifest, manifest_sha256, output_dir, rows, tuple(records),
                        CONFIG_SHA256, config["normalization_sha256"])


def _verify_prepared(job, config, statistics):
    current_config, current_statistics = _read_configuration()
    if (job.configuration_sha256 != CONFIG_SHA256
            or job.normalization_sha256 != current_config["normalization_sha256"]
            or current_config != config or current_statistics != statistics):
        raise ValueError("Prepared configuration differs from the loaded model")
    if (hash_file(job.manifest) != job.manifest_sha256
            or tuple(read_manifest(job.manifest)) != job.rows):
        raise ValueError("Manifest changed after validation")
    if _output_destination(job.output_dir) != job.output_dir:
        raise ValueError("Output destination changed after validation")


class Predictor:
    """Strict P56 EMA12 restore; predict_many uses consecutive pairs and a real tail."""

    def __init__(self, checkpoint):
        started = time.monotonic()
        package = Path(__file__).parent
        self.config, self.statistics = _read_configuration()
        normalization = package / "inference_normalization.json"
        checkpoint = Path(checkpoint).resolve(strict=True)
        selected = self.config["checkpoint"]
        if checkpoint.stat().st_size != selected["bytes"] or hash_file(checkpoint) != selected["sha256"]:
            raise ValueError("Only the exact selected P56 EMA12 checkpoint is accepted")
        print("P56_CHECKPOINT_VERIFIED", file=sys.stderr, flush=True)
        versions = {k: importlib.metadata.version(k) for k in self.config["versions"]}
        if versions != self.config["versions"]:
            raise RuntimeError("Use the exact recorded scientific dependency versions")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import numpy as np
        import torch
        from .model import build_model
        from .p17_model import state_hash
        from .p56_spatial_binding import install_spatial_path

        if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA L4":
            raise RuntimeError("This release is qualified only for NVIDIA L4 CUDA execution")
        sources = {}
        for name, suffix in (("encoder_embeddings.py", "encoder_embeddings"),
                             ("modality_info.py", "modality_info"), ("terramind_vit.py", "terramind_vit")):
            module = importlib.import_module("terratorch.models.backbones.terramind.model." + suffix)
            sources[name] = hash_file(module.__file__)
        if sources != self.config["pinned_sources"]:
            raise RuntimeError("TerraMind source fingerprints differ")
        random.seed(234); np.random.seed(234); torch.manual_seed(234); torch.cuda.manual_seed_all(234)
        cfg = copy.deepcopy(self.config["configuration"])
        cfg["model_args"]["backbone_ckpt_path"] = None
        cfg["model_args"]["backbone_pretrained"] = False
        model, audit = build_model(cfg, None, resume=True)
        install_spatial_path(model, initial=False, seed=234)
        binding = model._p22_audit
        if (binding["decoder_forward_source_sha256"] != "59c08eb136b091686d50e4ec9785fe4ad80f212aea54e91c3d5a3eb1360c4bbf"
                or binding["original_forward_source_sha256"] != "a110845ef02c44a258d495dcf8f6394af34a5bd81af74aacc7ee52d4e5079bd2"):
            raise RuntimeError("Live decoder/forward source differs")
        if sum(p.numel() for p in model.parameters()) != self.config["parameters"]:
            raise RuntimeError("Full P56 graph parameter count differs")
        if sum(p.numel() for p in model.joint_spatial.parameters()) != self.config["spatial_parameters"]:
            raise RuntimeError("P56 spatial path differs")
        if not all(p.dtype == torch.float32 for p in model.parameters()):
            raise RuntimeError("FP32 model parameters required")
        if not all(getattr(b, "_glb_checkpoint_enabled", False) for b in model.encoder.encoder):
            raise RuntimeError("Recorded encoder checkpoint wrappers differ")
        # Deserialization is allowed only after the exact immutable file SHA above.
        state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
        if (state["epoch"] != 12 or state["global_step"] != 11484
                or state["contract_sha256"] != self.config["contract_sha256"]):
            raise ValueError("Checkpoint scientific identity differs")
        model.load_state_dict(state["state_dict"], strict=True)
        del state
        self.before = state_hash(model.state_dict())
        if self.before != selected["state_sha256"]:
            raise ValueError("Strict restored named state differs")
        self.model = model.cuda().eval()
        self.model.requires_grad_(False)
        self.setup = dict(identity="P56-native-r1", checkpoint=selected,
                          configuration_sha256=CONFIG_SHA256, normalization_sha256=hash_file(normalization),
                          versions=versions, upstream_sources=sources, model_audit=audit, spatial_binding=binding,
                          public_source_sha256={p.name: hash_file(p) for p in package.glob("*.py")},
                          device=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                          cudnn=torch.backends.cudnn.version(),
                          backend_flags=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                                             cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                                             cudnn_benchmark=torch.backends.cudnn.benchmark,
                                             cudnn_deterministic=torch.backends.cudnn.deterministic),
                          named_state_before=self.before, setup_seconds=time.monotonic() - started,
                          initialized_at_utc=utc_now(), initializer_loaded=False, strict_state_load=True,
                          parameters=self.config["parameters"], spatial_parameters=self.config["spatial_parameters"],
                          python=sys.version.split()[0])
        print("P56_MODEL_LOADED", file=sys.stderr, flush=True)

    def predict_many(self, manifest, output_dir):
        """Write masks and an auditable receipt; refuse an existing output directory."""
        return self._predict_prepared(prepare_job(manifest, output_dir))

    def _predict_prepared(self, job):
        _verify_prepared(job, self.config, self.statistics)
        import numpy as np
        import rasterio
        import torch
        from .p17_model import state_hash

        rows = job.rows
        output_dir = job.output_dir
        output_dir.mkdir(parents=True, exist_ok=False)
        receipt_path = output_dir / "receipt.json"
        receipt = dict(**self.setup, manifest_sha256=job.manifest_sha256, order=[r["id"] for r in rows],
                       started_at_utc=utc_now(), status="preparing", calls_started=0, calls_completed=0,
                       views_started=0, views_completed=0, batches=[], inputs=list(job.inputs), outputs=[])
        write_json(receipt_path, receipt)
        try:
            # Reuse the one whole-job CPU validation; reread each batch before forward.
            with torch.inference_mode():
                for start in range(0, len(rows), 2):
                    batch = [read_input(r, self.statistics) for r in rows[start:start + 2]]
                    for offset, item in enumerate(batch):
                        if item[3] != receipt["inputs"][start + offset]:
                            raise ValueError("Input changed after validation")
                    x = {k: torch.stack([torch.from_numpy(v[0][k]) for v in batch]).cuda(non_blocking=True)
                         for k in ("S2L2A", "S1GRD", "DEM")}
                    record = dict(ids=[r["id"] for r in rows[start:start + 2]], size=len(batch),
                                  started_at_utc=utc_now(), input_shapes={k: list(v.shape) for k, v in x.items()})
                    receipt["batches"].append(record)
                    receipt["calls_started"] += 1; receipt["views_started"] += len(batch)
                    receipt["status"] = "running"
                    write_json(receipt_path, receipt)
                    tick = time.monotonic()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = self.model(x).output
                    if logits.ndim == 4:
                        logits = logits[:, 0]
                    logits = logits.float()
                    if logits.shape != (len(batch), 256, 256) or not torch.isfinite(logits).all():
                        raise ValueError("Nonfinite or unexpected model output")
                    predictions = (logits >= 0).cpu().numpy()
                    record.update(completed_at_utc=utc_now(), forward_seconds=time.monotonic() - tick,
                                  logits_shape=list(logits.shape), logits_dtype=str(logits.dtype), finite=True)
                    receipt["calls_completed"] += 1; receipt["views_completed"] += len(batch)
                    write_json(receipt_path, receipt)
                    for prediction, (_, valid, profile, inp) in zip(predictions, batch):
                        pixels = np.where(valid, prediction, 255).astype(np.uint8)
                        path = output_dir / (inp["id"] + ".tif")
                        if path.exists():
                            raise FileExistsError(path)
                        with rasterio.open(path, "w", **profile) as ds:
                            ds.write(pixels, 1)
                        with rasterio.open(path) as ds:
                            if (not np.array_equal(ds.read(1), pixels) or ds.crs != profile["crs"]
                                    or ds.transform != profile["transform"] or ds.nodata != 255):
                                raise ValueError("GeoTIFF round-trip mismatch")
                        receipt["outputs"].append(dict(id=inp["id"], file=path.name, bytes=path.stat().st_size,
                            sha256=hash_file(path), grid_sha256=inp["grid_sha256"], roundtrip_exact=True,
                            pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                            packed_sha256=hashlib.sha256(np.packbits(prediction & valid, axis=1, bitorder="big").tobytes()).hexdigest(),
                            lake_pixels=int((pixels == 1).sum()), nodata_pixels=int((pixels == 255).sum())))
                        write_json(receipt_path, receipt)
            receipt["named_state_after"] = state_hash(self.model.state_dict())
            if receipt["named_state_after"] != self.before:
                raise ValueError("Inference changed the selected state")
            receipt.update(status="complete", completed_at_utc=utc_now())
        except BaseException as exc:
            receipt.update(status="failed", error_type=type(exc).__name__, failed_at_utc=utc_now())
            write_json(receipt_path, receipt)
            raise
        write_json(receipt_path, receipt)
        return receipt
