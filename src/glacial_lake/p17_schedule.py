"""Reference only: an explicit one-based optimizer-update LR multiplier."""
from __future__ import annotations
import math


def lr_multiplier(update: int, warmup: int = 1914, total: int = 22968,
                  end: float = 0.01) -> float:
    """Return the multiplier to use for update 1..total, not a microbatch index."""
    if not 0 < warmup < total or not 0.0 < end <= 1.0:
        raise ValueError("Invalid fixed schedule parameters")
    if isinstance(update, bool) or not isinstance(update, int) or not 1 <= update <= total:
        raise ValueError("update must be an integer in [1, total]")
    if update <= warmup:
        return update / warmup
    progress = (update - warmup) / (total - warmup)
    return end + (1.0 - end) * (1.0 + math.cos(math.pi * progress)) / 2.0


def assign_next_lr(optimizer, peak_lrs: list[float], completed_updates: int,
                   warmup: int = 1914, total: int = 22968, end: float = 0.01) -> float:
    """Apply before optimizer.step(). Keep peak_lrs in the full-state checkpoint."""
    if len(peak_lrs) != len(optimizer.param_groups):
        raise ValueError("Optimizer groups differ from the reserved peak-LR list")
    multiplier = lr_multiplier(completed_updates + 1, warmup, total, end)
    for group, peak in zip(optimizer.param_groups, peak_lrs):
        if not math.isfinite(peak) or peak <= 0:
            raise ValueError("Invalid peak learning rate")
        group["lr"] = peak * multiplier
    return multiplier
