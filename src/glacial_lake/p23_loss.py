"""Reference P23 component-balanced local auxiliary loss.

This is a proposed GLB adaptation inspired by instance-balanced segmentation,
not an implementation of the published blob-loss paper. No model, data, GPU,
cloud access, sampler change, label edit, or inference policy is included.
"""
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class ComponentRegion:
    # [y0:y1, x0:x1] in ORIGINAL output coordinates.
    box: tuple[int, int, int, int]
    positive: np.ndarray
    background: np.ndarray
    area_pixels: int

def _window(lo: int, hi: int, length: int, padding: int, minimum: int) -> tuple[int, int]:
    size = min(length, max(minimum, hi - lo + 2 * padding))
    start = (lo + hi - size) // 2
    start = max(0, min(start, length - size))
    return start, start + size

def make_regions(target: np.ndarray, valid: np.ndarray, *,
                 minimum_area: int = 16, maximum_area: int = 255,
                 padding: int = 16, minimum_window: int = 64
                 ) -> list[ComponentRegion]:
    """Build TRAIN-target-only regions; 8-connectivity, complete objects only.

    Other target components are ignored in this auxiliary region, NOT relabeled
    as background. Global segmentation loss and evaluation keep their support.
    Recompute after the exact geometric augmentation, or transform an exact
    cached integer component map and then make regions. Never use val/test masks
    to form training regions or to feed the predictor.
    """
    from scipy import ndimage
    y = np.asarray(target, dtype=bool)
    v = np.asarray(valid, dtype=bool)
    if y.ndim != 2 or y.shape != v.shape:
        raise ValueError("target/valid must be aligned HxW arrays")
    if not (0 < minimum_area <= maximum_area) or padding < 0 or minimum_window < 1:
        raise ValueError("invalid fixed region configuration")
    labels, count = ndimage.label(y & v, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    boxes = ndimage.find_objects(labels)
    adjacent_invalid = ndimage.binary_dilation(~v, structure=np.ones((3, 3), dtype=bool))
    height, width = y.shape
    result: list[ComponentRegion] = []
    for label_id, slices in enumerate(boxes, start=1):
        if slices is None or not minimum_area <= sizes[label_id] <= maximum_area:
            continue
        sy, sx = slices
        if sy.start == 0 or sx.start == 0 or sy.stop == height or sx.stop == width:
            continue
        component_crop = labels[sy, sx] == label_id
        if np.any(component_crop & adjacent_invalid[sy, sx]):
            continue
        y0, y1 = _window(sy.start, sy.stop, height, padding, minimum_window)
        x0, x1 = _window(sx.start, sx.stop, width, padding, minimum_window)
        pos = np.ascontiguousarray(labels[y0:y1, x0:x1] == label_id)
        bg = np.ascontiguousarray(v[y0:y1, x0:x1] & ~y[y0:y1, x0:x1])
        if np.any(pos & bg) or int(pos.sum()) != int(sizes[label_id]):
            raise ValueError("region support failed to preserve the complete component")
        result.append(ComponentRegion((y0, y1, x0, x1), pos, bg, int(sizes[label_id])))
    return result

def component_auxiliary_loss(logits: torch.Tensor,
                             regions: Sequence[Sequence[ComponentRegion]],
                             smooth: float = 1.0) -> torch.Tensor:
    """Return mean_i mean_components [0.5*balancedBCE + 0.5*localDice].

    Images with no eligible component contribute zero, not a rescaled batch.
    For each component, positive/background BCE means receive equal weight.
    No inverse-area multiplier, ranking by predicted error, or inverse-frequency
    weights are added. All reductions use float32 and gradients reach the FULL
    combined P22 prediction (not its standalone main path).
    """
    z = logits[:, 0] if logits.ndim == 4 and logits.shape[1] == 1 else logits
    if z.ndim != 3 or len(regions) != z.shape[0] or smooth <= 0:
        raise ValueError("Expected N,H,W or N,1,H,W logits and N region lists")
    if z.shape[0] == 0:
        raise ValueError("Empty batches are not supported")
    image_losses = []
    for image_index, image_regions in enumerate(regions):
        component_losses = []
        for region in image_regions:
            y0, y1, x0, x1 = region.box
            patch = z[image_index, y0:y1, x0:x1].float()
            pos = torch.as_tensor(region.positive, device=z.device, dtype=torch.bool)
            bg = torch.as_tensor(region.background, device=z.device, dtype=torch.bool)
            if patch.shape != pos.shape or bg.shape != pos.shape:
                raise ValueError("A region mask does not match its output window")
            if region.area_pixels <= 0:
                raise ValueError("Empty target component")
            positive_bce = F.softplus(-patch[pos]).mean()
            if region.background.any():
                balanced_bce = 0.5 * (positive_bce + F.softplus(patch[bg]).mean())
            else:
                balanced_bce = positive_bce
            probability = torch.sigmoid(patch)
            local_dice = 1.0 - (
                2.0 * probability[pos].sum() + smooth
            ) / (probability[pos | bg].sum() + region.area_pixels + smooth)
            component_losses.append(0.5 * balanced_bce + 0.5 * local_dice)
        image_losses.append(
            torch.stack(component_losses).mean()
            if component_losses else z[image_index].float().sum() * 0.0
        )
    return torch.stack(image_losses).mean()

def candidate_loss(original_loss: torch.Tensor, logits: torch.Tensor,
                   regions: Sequence[Sequence[ComponentRegion]]) -> tuple[torch.Tensor, torch.Tensor]:
    auxiliary = component_auxiliary_loss(logits, regions)
    return original_loss.float() + 0.20 * auxiliary, auxiliary
