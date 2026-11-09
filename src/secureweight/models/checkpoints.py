"""Copy-only, checksum-verified healthy checkpoints. Never load Python objects."""

from pathlib import Path

import torch

from secureweight.models import MLP
from secureweight.tracking import file_hash


def save_checkpoint(model: MLP, path, preprocessing: dict) -> str:
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("checkpoint requires finite float32 weights")
    payload = {"schema_version": 1, "architecture": model.architecture,
               "activation": "relu", "preprocessing": preprocessing, "state_dict": state}
    with Path(path).open("xb") as stream:
        torch.save(payload, stream)
    return file_hash(path)


def load_checkpoint(path, expected_sha256: str) -> tuple[MLP, dict]:
    # Hash and deserialize the same bytes to avoid a path-replacement race.
    import hashlib
    import io

    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("checkpoint checksum mismatch")
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1 or payload.get("activation") != "relu":
        raise ValueError("unsupported checkpoint schema")
    model = MLP(**payload["architecture"])
    expected = model.state_dict()
    state = payload["state_dict"]
    if set(state) != set(expected):
        raise ValueError("checkpoint tensor names mismatch")
    for name, tensor in state.items():
        if tensor.dtype != torch.float32 or tensor.shape != expected[name].shape or not torch.isfinite(tensor).all():
            raise ValueError("checkpoint contains invalid tensors")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, payload["preprocessing"]
