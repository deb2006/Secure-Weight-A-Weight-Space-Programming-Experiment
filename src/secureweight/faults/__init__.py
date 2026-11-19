"""Controlled CPU float32 faults, applied to isolated model copies."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import struct

import numpy as np
import torch
import yaml


@dataclass(frozen=True)
class FaultConfig:
    operator: str = "gaussian"
    seed: int = 42
    fraction: float = 0.1
    magnitude: float = 0.5
    tensors: tuple[str, ...] = ()
    neuron: int | None = None
    bit: int | None = None
    min_healthy_accuracy: float = 0.85
    damage_threshold: float = 0.15
    severe_threshold: float = 0.30

    def __post_init__(self):
        if self.operator not in ("healthy", "gaussian", "bit_flip", "sign", "scale", "offset"):
            raise ValueError("unknown fault operator")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for name in ("fraction", "magnitude", "min_healthy_accuracy", "damage_threshold", "severe_threshold"):
            v = getattr(self, name)
            if type(v) not in (int, float) or not math.isfinite(v):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.fraction <= 1 or not 0 <= self.min_healthy_accuracy <= 1:
            raise ValueError("invalid fraction or eligibility threshold")
        if not 0 < self.damage_threshold < self.severe_threshold <= 1:
            raise ValueError("require 0 < damage threshold < severe threshold <= 1")
        if self.operator == "gaussian" and self.magnitude < 0:
            raise ValueError("Gaussian standard deviation must be nonnegative")
        if self.bit is not None and (type(self.bit) is not int or not 0 <= self.bit <= 31):
            raise ValueError("bit must be an integer in [0,31]")
        if self.bit is not None and self.operator != "bit_flip":
            raise ValueError("bit only applies to bit_flip")
        if self.neuron is not None and (type(self.neuron) is not int or self.neuron < 0):
            raise ValueError("neuron must be a nonnegative integer")
        if not isinstance(self.tensors, (tuple, list)) or any(type(n) is not str for n in self.tensors):
            raise ValueError("tensors must be a list of names")
        if len(set(self.tensors)) != len(self.tensors) or (self.neuron is not None and self.tensors):
            raise ValueError("duplicate tensors or conflicting targeting modes")

    def to_dict(self):
        return asdict(self)


def load_fault_config(path):
    import json
    path = Path(path)
    if path.suffix.lower() not in (".json", ".yaml", ".yml"):
        raise ValueError("fault configuration must be JSON or YAML")
    text = path.read_text(encoding="utf-8")
    values = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(values, dict):
        raise ValueError("fault configuration must be an object")
    return FaultConfig(**values)


def flip_float32(value, bit):
    """Return the exact IEEE-754 single-bit XOR result, including nonfinite values."""
    if type(bit) is not int or not 0 <= bit < 32:
        raise ValueError("bit must be in [0,31]")
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return struct.unpack("<f", struct.pack("<I", bits ^ (1 << bit)))[0]


def inject_fault(model, config: FaultConfig):
    """Return a cloned model and complete researcher-only attempt records.

    Sampling is without replacement over eligible coordinates; each coordinate
    gets one attempt, with no retries after rejection or a rounded-away edit.
    """
    clone = deepcopy(model).cpu().eval()
    state = clone.state_dict()
    if any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("fault input must contain finite float32 tensors")
    names = list(config.tensors) or list(state)
    if any(n not in state for n in names):
        raise ValueError("unknown target tensor")
    if config.neuron is not None and config.neuron >= model.architecture["hidden_dim"]:
        raise ValueError("hidden neuron out of range")
    candidates = []
    for name in sorted(names):
        for flat in range(state[name].numel()):
            coordinate = tuple(int(i) for i in np.unravel_index(flat, tuple(state[name].shape)))
            if config.neuron is not None:
                eligible = ((name in ("hidden.weight", "hidden.bias") and coordinate[0] == config.neuron)
                            or (name == "output.weight" and coordinate[1] == config.neuron))
                if not eligible:
                    continue
            candidates.append((name, flat, coordinate))
    rng = np.random.default_rng(config.seed)
    count = 0 if config.operator == "healthy" else max(1, math.ceil(len(candidates) * config.fraction))
    selected = sorted(rng.choice(len(candidates), size=count, replace=False).tolist())
    attempts = []
    for index in selected:
        name, flat, coordinate = candidates[index]
        before = state[name].view(-1)[flat].item()
        bit = None
        if config.operator == "gaussian":
            proposed = before + float(rng.normal(0, config.magnitude))
        elif config.operator == "bit_flip":
            bit = config.bit if config.bit is not None else int(rng.integers(0, 32))
            proposed = flip_float32(before, bit)
        elif config.operator == "sign":
            proposed = -before
        elif config.operator == "scale":
            proposed = before * config.magnitude
        else:
            proposed = before + config.magnitude
        with np.errstate(over="ignore", invalid="ignore"):
            after = float(np.float32(proposed))
        accepted = math.isfinite(after)
        changed = accepted and struct.pack("<f", after) != struct.pack("<f", before)
        attempts.append({"tensor_name": name, "flat_index": flat, "coordinates": list(coordinate),
                         "before": before, "after": after if accepted else None,
                         "bit": bit, "accepted": accepted, "changed": changed,
                         "rejection": None if accepted else "nonfinite_float32"})
        if accepted:
            state[name].view(-1)[flat] = after
    clone.load_state_dict(state)
    return clone, {"eligible_coordinates": len(candidates), "attempted_count": count,
                   "changed_count": sum(a["changed"] for a in attempts),
                   "rejected_count": sum(not a["accepted"] for a in attempts), "attempts": attempts}
