"""Preserve the original P02 pairwise objective across physical forward chunks.

Adapted from the user-supplied P02S reference; integration measurements are separate.
Logical loss groups remain consecutive pairs, even when a forward handles 4 or 8
images. Effective batches and optimizer/scheduler updates remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class LogicalGroupLoss:
    loss_sum: Tensor
    bce_sum: Tensor
    positive_dice_sum: Tensor
    valid_groups: Tensor
    finite_valid_logits: Tensor


def logical_group_loss_sums(
    logits: Tensor,
    target: Tensor,
    valid: Tensor,
    logical_group_size: int = 2,
) -> LogicalGroupLoss:
    """Return sums, NOT the mean over the physical forward batch.

    Divide loss_sum by the number of valid LOGICAL groups in the entire effective
    batch before backward. Sum diagnostics across physical forwards and divide by
    that same count for logging. Do not divide by physical chunk count as well.

    Empty-positive logical groups have zero Dice but retain BCE. All-invalid
    groups contribute zero and are excluded from the divisor. The caller must
    reject an entirely invalid effective batch. Invalid targets/logits are masked
    before arithmetic, including the invalid-NaN case. Nonfinite valid logits
    are reported and MUST prevent optimizer.step; do not silently nan_to_num them.

    There is no Python control flow depending on CUDA tensor values in this helper.
    Chunk boundaries must align with logical groups. A final incomplete group is
    allowed only at the end of the effective batch. This preserves the mathematical
    objective for fixed logits, not bitwise network/RNG equivalence across batches.
    """
    if logits.ndim == 4 and logits.shape[1] == 1:
        logits = logits[:, 0]
    if logits.ndim != 3 or target.shape != logits.shape or valid.shape != logits.shape:
        raise ValueError('Expected matching [N,H,W] logits, targets, and validity.')
    if logical_group_size < 1 or logits.shape[0] < 1 or valid.dtype != torch.bool:
        raise ValueError('Positive group size, nonempty batch, and Boolean mask required.')
    if target.device != logits.device or valid.device != logits.device:
        raise ValueError('All input tensors must be on the same device.')

    n = logits.shape[0]
    padding = (-n) % logical_group_size
    if padding:
        pad = (0, 0, 0, 0, 0, padding)
        logits = F.pad(logits, pad)
        target = F.pad(target, pad)
        valid = F.pad(valid, pad, value=False)

    safe_logits = torch.where(valid, logits.float(), 0.0)
    safe_target = torch.where(valid, target.float(), 0.0)
    finite = torch.logical_or(torch.isfinite(logits), ~valid).all()
    element_bce = F.binary_cross_entropy_with_logits(safe_logits, safe_target, reduction='none')
    element_bce = torch.where(valid, element_bce, 0.0)
    groups = logits.shape[0] // logical_group_size
    group_pixels = valid.reshape(groups, -1).sum(1)
    group_bce = element_bce.reshape(groups, -1).sum(1) / group_pixels.clamp_min(1)

    probabilities = torch.where(valid, torch.sigmoid(safe_logits), 0.0)
    positives = safe_target.flatten(1).sum(1)
    include = positives > 0
    intersection = (probabilities * safe_target).flatten(1).sum(1)
    denominator = probabilities.flatten(1).sum(1) + positives
    sample_dice = 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    sample_dice = torch.where(include, sample_dice, 0.0)
    group_positive_count = include.reshape(groups, logical_group_size).sum(1)
    group_dice = sample_dice.reshape(groups, logical_group_size).sum(1) / group_positive_count.clamp_min(1)
    active = group_pixels > 0
    bce_sum = torch.where(active, group_bce, 0.0).sum()
    dice_sum = torch.where(active, group_dice, 0.0).sum()
    return LogicalGroupLoss(
        loss_sum=bce_sum + dice_sum,
        bce_sum=bce_sum.detach(),
        positive_dice_sum=dice_sum.detach(),
        valid_groups=active.sum(),
        finite_valid_logits=finite,
    )
