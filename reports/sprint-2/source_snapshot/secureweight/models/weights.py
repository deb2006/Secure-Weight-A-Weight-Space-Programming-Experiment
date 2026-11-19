"""Versioned lossless float32 JSON representation; no patch execution."""

from pathlib import Path

import torch

from secureweight.tracking import read_json, write_json


def extract_weights(model) -> dict:
    tensors = {}
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu()
        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError("Sprint 1 serialization requires finite float32 tensors")
        tensors[name] = {"shape": list(tensor.shape), "dtype": "float32",
                         "values": tensor.tolist(),
                         "statistics": {"count": tensor.numel(), "min": tensor.min().item(),
                                        "max": tensor.max().item(), "mean": tensor.double().mean().item(),
                                        "std_population": tensor.double().std(unbiased=False).item()}}
    return {"schema_version": 1, "format": "secureweight.float32-json",
            "architecture": dict(model.architecture), "activation": "relu", "tensors": tensors}


def save_weights(model, path: str | Path):
    write_json(path, extract_weights(model))


def load_weights(model, path: str | Path):
    payload = read_json(path)
    if payload.get("schema_version") != 1 or payload.get("format") != "secureweight.float32-json":
        raise ValueError("unsupported weight format")
    if payload.get("architecture") != model.architecture or payload.get("activation") != "relu":
        raise ValueError("weight architecture mismatch")
    tensors = payload.get("tensors", {})
    expected = model.state_dict()
    if set(tensors) != set(expected):
        raise ValueError("weight tensor names mismatch")
    state = {}
    for name, target in expected.items():
        entry = tensors[name]
        if entry.get("dtype") != "float32" or entry.get("shape") != list(target.shape):
            raise ValueError(f"weight dtype/shape mismatch: {name}")
        try:
            tensor = torch.tensor(entry["values"], dtype=torch.float32)
        except (KeyError, TypeError, ValueError, RuntimeError, OverflowError) as exc:
            raise ValueError(f"invalid weight values: {name}") from exc
        if tensor.shape != target.shape or not torch.isfinite(tensor).all():
            raise ValueError(f"invalid weight values: {name}")
        state[name] = tensor
    # Validate the complete document before mutating the destination model.
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
