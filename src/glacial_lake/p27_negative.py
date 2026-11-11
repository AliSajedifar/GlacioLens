"""P27 reference: TRAIN-only large false-region auxiliary.

Not a published new algorithm or a completed TerraMind integration. Component
membership uses detached current predictions; gradients flow only to eligible
GT-background logits. This module does not relabel positives, prune inference
masks, select validation cases, or compute the original/positive losses.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

@dataclass(frozen=True)
class NegativeRegionPlan:
    labels: np.ndarray
    # For every component id, number of valid ground-truth background pixels.
    background_counts: np.ndarray
    background_mask: np.ndarray
    medium_ids: np.ndarray
    large_ids: np.ndarray
    diagnostic: dict[str, int]

def negative_coefficient(epoch_one_based: int) -> float:
    """Fixed ramp: 0,0,.025,.05,.075,.10, then .10. Not reset on resume."""
    if isinstance(epoch_one_based, bool) or not isinstance(epoch_one_based, int):
        raise TypeError("epoch must be a one-based integer")
    if not 1 <= epoch_one_based <= 24:
        raise ValueError("P27 defines exactly 24 epochs per arm")
    return 0.025 * min(4, max(0, epoch_one_based - 2))

def make_negative_plans(logits: torch.Tensor, targets: torch.Tensor,
                        valid: torch.Tensor) -> list[NegativeRegionPlan]:
    """Build plans for a logical microbatch after its actual augmentation.

    Fixed rule: raw logit >= 0, 8-connected region area >= 256, GT-positive
    fraction <= 0.10. Drop invalid-touching regions but KEEP image-border
    regions. A region with a small true lake keeps that lake out of this loss.
    Targets are used only to form a TRAIN loss, never an inference feature.
    """
    def nhw(t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 4 and t.shape[1] == 1:
            t = t[:, 0]
        if t.ndim != 3 or t.shape[0] == 0:
            raise ValueError("Expected nonempty N,H,W or N,1,H,W")
        return t
    z, y, v = nhw(logits), nhw(targets), nhw(valid)
    if z.shape != y.shape or z.shape != v.shape:
        raise ValueError("logits, targets and valid must be pixel-aligned")
    z_cpu = z.detach().float().cpu().numpy()
    y_cpu = y.detach().to(device="cpu", dtype=torch.bool).numpy()
    v_cpu = v.detach().to(device="cpu", dtype=torch.bool).numpy()
    result: list[NegativeRegionPlan] = []
    structure = np.ones((3, 3), dtype=np.uint8)
    for score, target, support in zip(z_cpu, y_cpu, v_cpu):
        if not np.isfinite(score[support]).all():
            raise FloatingPointError("Nonfinite logits on valid support")
        lab, count = ndimage.label((score >= 0) & support, structure=structure)
        lab = np.ascontiguousarray(lab, dtype=np.int32)
        areas = np.bincount(lab.ravel(), minlength=count + 1)
        positives = np.bincount(lab[target & support].ravel(), minlength=count + 1)
        backgrounds = areas - positives
        near_invalid = ndimage.binary_dilation(~support, structure=structure,
                                               border_value=0)
        excluded = np.zeros(count + 1, dtype=bool)
        excluded[np.unique(lab[near_invalid])] = True
        eligible = (areas >= 256) & (10 * positives <= areas) & ~excluded
        eligible[0] = False
        medium = np.flatnonzero(eligible & (areas < 4096)).astype(np.int64)
        large = np.flatnonzero(eligible & (areas >= 4096)).astype(np.int64)
        selected_bg = support & ~target & eligible[lab]
        ids = np.concatenate((medium, large))
        diagnostic = {
            "predicted_regions": int(count),
            "medium_regions": int(len(medium)),
            "large_regions": int(len(large)),
            "eligible_background_pixels": int(selected_bg.sum()),
            "eligible_positive_pixels_excluded_from_auxiliary": int(positives[ids].sum()),
            "eligible_prediction_pixels": int(areas[ids].sum()),
            "invalid_touching_large_regions_excluded": int(((areas >= 256) & excluded & (np.arange(count + 1) > 0)).sum()),
        }
        result.append(NegativeRegionPlan(lab, backgrounds.astype(np.int64),
                      np.ascontiguousarray(selected_bg), medium, large, diagnostic))
    return result

def negative_region_loss(logits: torch.Tensor,
                         plans: Sequence[NegativeRegionPlan], statistics: dict | None = None) -> torch.Tensor:
    """Mean(image, mean(nonempty area strata, mean(components, mean(bg BCE)))).

    An image without eligible components contributes zero; do NOT divide by
    just images that contain errors. Scalar loss is FP32, differentiable into
    the FULL live model. No pixelwise top-k, inverse-area weight or extra pass.
    """
    z = logits[:, 0] if logits.ndim == 4 and logits.shape[1] == 1 else logits
    if z.ndim != 3 or len(plans) != z.shape[0] or z.shape[0] == 0:
        raise ValueError("Expected N aligned plans and a nonempty N,H,W tensor")
    losses = []
    if statistics is not None:
        statistics.setdefault("images",0);statistics["images"]+=len(plans)
    for image, plan in zip(z, plans):
        flat = image.float().reshape(-1)
        if tuple(image.shape) != plan.labels.shape:
            raise ValueError("Plan/output geometry mismatch")
        if len(plan.medium_ids) + len(plan.large_ids) == 0:
            # Empty slice creates a zero with a gradient path and no NaN read.
            if statistics is not None:
                for key,value in plan.diagnostic.items():statistics[key]=statistics.get(key,0)+value
            losses.append(flat[:0].sum())
            continue
        ids = torch.as_tensor(plan.labels.reshape(-1), dtype=torch.long, device=z.device)
        bg = torch.as_tensor(plan.background_mask.reshape(-1), dtype=torch.bool, device=z.device)
        counts = torch.as_tensor(plan.background_counts, dtype=torch.float32, device=z.device)
        sums = torch.zeros(len(plan.background_counts), dtype=torch.float32, device=z.device)
        sums = sums.scatter_add(0, ids[bg], F.softplus(flat[bg]))
        strata = []
        if statistics is not None:
            for key,value in plan.diagnostic.items():statistics[key]=statistics.get(key,0)+value
            area=np.bincount(plan.labels.ravel(),minlength=len(plan.background_counts))
        for label,members in (("256_4095",plan.medium_ids),("ge4096",plan.large_ids)):
            if len(members):
                keep = torch.as_tensor(members, dtype=torch.long, device=z.device)
                # Every eligible component has >=90% GT-background.
                region_means=sums[keep] / counts[keep]
                strata.append(region_means.mean())
                if statistics is not None:
                    values={"regions":len(members),"prediction_pixels":int(area[members].sum()),
                        "background_pixels":int(plan.background_counts[members].sum()),
                        "positive_pixels_excluded":int((area[members]-plan.background_counts[members]).sum()),
                        "purity_sum":float(((area[members]-plan.background_counts[members])/area[members]).sum()),
                        "component_loss_sum":float(region_means.detach().sum()),"image_stratum_loss_sum":float(strata[-1].detach()),"images":1}
                    bucket=statistics.setdefault(label,{})
                    for key,value in values.items():bucket[key]=bucket.get(key,0)+value
        losses.append(torch.stack(strata).mean())
    return torch.stack(losses).mean()
