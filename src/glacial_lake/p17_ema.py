"""Tensor-only EMA reference; no deepcopy of activation-checkpointed model graphs.

Integrate with the existing runner. This is not a complete trainer. The shadow
updates only after successful optimizer steps. Validation temporarily swaps the
EMA state into the original graph, then restores every live tensor and mode.
"""
from __future__ import annotations
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Iterator
import torch
from torch import nn


class TensorEMA:
    def __init__(self, decay: float = 0.9995, initialize_after: int = 1914):
        if not 0.0 < decay < 1.0 or initialize_after < 1:
            raise ValueError("Invalid EMA configuration")
        self.decay = float(decay)
        self.initialize_after = int(initialize_after)
        self.completed_updates = 0
        self.averaged_updates = 0
        self.shadow: OrderedDict[str, torch.Tensor] = OrderedDict()

    @staticmethod
    def _state(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
        state = model.state_dict()
        if not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise TypeError("Non-tensor extra_state needs an explicit project handler")
        if any(t.is_complex() for t in state.values()):
            raise TypeError("This real-valued segmentation EMA does not support complex state")
        return state

    def _validate(self, live: OrderedDict[str, torch.Tensor]) -> None:
        if set(live) != set(self.shadow):
            raise ValueError("EMA/model keys differ")
        for name, value in live.items():
            old = self.shadow[name]
            if old.shape != value.shape or old.device != value.device:
                raise ValueError(f"EMA/model shape or device differs: {name}")

    @torch.no_grad()
    def after_optimizer_step(self, model: nn.Module, completed_updates: int) -> None:
        if completed_updates != self.completed_updates + 1:
            raise ValueError("EMA must be called once per successful optimizer update, in order")
        if completed_updates < self.initialize_after:
            self.completed_updates = completed_updates
            return
        live = self._state(model)
        if completed_updates == self.initialize_after:
            if self.shadow:
                raise ValueError("EMA is already initialized")
            self.shadow = OrderedDict(
                (name, value.detach().to(dtype=torch.float32 if value.is_floating_point()
                                        else value.dtype).clone())
                for name, value in live.items()
            )
        else:
            if not self.shadow:
                raise ValueError("Missing EMA initialization on resume")
            self._validate(live)
            parameter_names = set(dict(model.named_parameters()))
            for name, value in live.items():
                if name in parameter_names and value.is_floating_point():
                    self.shadow[name].lerp_(value.detach().float(), 1.0 - self.decay)
                else:
                    self.shadow[name].copy_(value)
            self.averaged_updates += 1
        self.completed_updates = completed_updates

    @contextmanager
    def evaluate(self, model: nn.Module) -> Iterator[nn.Module]:
        if not self.shadow or self.averaged_updates < 1:
            raise RuntimeError("EMA is not eligible before at least one post-warmup update")
        live = self._state(model)
        self._validate(live)
        backup = OrderedDict((name, value.detach().clone()) for name, value in live.items())
        modes = [(module, module.training) for module in model.modules()]
        try:
            with torch.no_grad():
                for name, value in live.items():
                    value.copy_(self.shadow[name])
                model.eval()
                yield model
        finally:
            with torch.no_grad():
                for name, value in live.items():
                    value.copy_(backup[name])
            for module, mode in modes:
                module.training = mode
            del backup

    def state_dict(self) -> dict[str, Any]:
        """Serialize promptly; these tensors are not a frozen asynchronous snapshot."""
        return {
            "decay": self.decay, "initialize_after": self.initialize_after,
            "completed_updates": self.completed_updates,
            "averaged_updates": self.averaged_updates, "shadow": self.shadow,
        }

    @torch.no_grad()
    def load_state_dict(self, record: dict[str, Any], model: nn.Module) -> None:
        if float(record["decay"]) != self.decay or int(record["initialize_after"]) != self.initialize_after:
            raise ValueError("EMA recipe differs from the reserved run")
        step = int(record["completed_updates"])
        averages = int(record["averaged_updates"])
        if step < 0 or averages != max(0, step - self.initialize_after):
            raise ValueError("Inconsistent EMA update counters")
        incoming = record["shadow"]
        if step < self.initialize_after:
            if incoming:
                raise ValueError("Unexpected pre-warmup EMA shadow")
            self.shadow = OrderedDict()
        else:
            live = self._state(model)
            if set(incoming) != set(live):
                raise ValueError("Restored EMA keys differ")
            restored = OrderedDict()
            for name, value in live.items():
                saved = incoming[name]
                if not isinstance(saved, torch.Tensor) or saved.shape != value.shape:
                    raise ValueError(f"Invalid restored EMA tensor: {name}")
                dtype = torch.float32 if value.is_floating_point() else value.dtype
                restored[name] = saved.detach().to(device=value.device, dtype=dtype).clone()
            self.shadow = restored
        self.completed_updates, self.averaged_updates = step, averages
