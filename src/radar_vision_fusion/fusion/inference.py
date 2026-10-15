"""Explicit checkpoint inference and a deterministic example predictor."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .dkf_layer import DEFAULT_Q
from .model import CovarianceConfig, CovarianceNet


def _features(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != 23 or not len(values):
        raise ValueError("features must have shape [time >= 1, 23]")
    if not np.isfinite(values).all():
        raise ValueError("features must be finite; missing observations use validity flags")
    return values


def _diagonal(values: float | Sequence[float], size: int, name: str) -> np.ndarray:
    diagonal = np.asarray(values, dtype=np.float32)
    if diagonal.ndim == 0:
        diagonal = np.full(size, float(diagonal), dtype=np.float32)
    if diagonal.shape != (size,) or not np.isfinite(diagonal).all() or (diagonal <= 0).any():
        raise ValueError(f"{name} must be a positive scalar or {size} finite positive entries")
    return diagonal


class FixedNoisePredictor:
    """Fixed covariance for the runnable example, without trained weights."""

    def __init__(
        self, r_cam: Sequence[float] = (3.0, 3.0),
        r_rad: Sequence[float] = (3.0, 3.0, 1.0, 1.0),
        q: float | Sequence[float] = DEFAULT_Q,
    ) -> None:
        self.base_r = np.concatenate([_diagonal(r_cam, 2, "r_cam"), _diagonal(r_rad, 4, "r_rad")])
        self.q_diagonal = _diagonal(q, 4, "q")
        self.q = float(self.q_diagonal[0]) if np.all(self.q_diagonal == self.q_diagonal[0]) else self.q_diagonal.copy()
        self.max_seq_len = 256

    def predict(self, features: np.ndarray, hidden_state=None):
        length = len(_features(features))
        r = np.broadcast_to(self.base_r, (length, 6)).copy()
        q = np.broadcast_to(self.q_diagonal, (length, 4)).copy()
        return r[:, :2], r[:, 2:], q, (None, None)


def save_checkpoint(
    model: CovarianceNet, path: str | Path, *,
    q: float | Sequence[float] = DEFAULT_Q,
) -> Path:
    """Save strict weights plus architecture, statistics and fixed Q.

    Only tensor weights and simple Python values are stored. The artifact is
    compatible with safe ``torch.load(..., weights_only=True)`` loading.
    """
    q_diagonal = _diagonal(q, 4, "q")
    q_config = float(q_diagonal[0]) if np.all(q_diagonal == q_diagonal[0]) else q_diagonal.tolist()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    config = {**model.config.to_dict(), "kind": "trans", "q": q_config}
    config["mean"] = model.mean.detach().cpu().tolist()
    config["std"] = model.std.detach().cpu().tolist()
    torch.save({
        "format_version": 1,
        "state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "config": config,
    }, output)
    return output


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu",
) -> tuple[CovarianceNet, np.ndarray]:
    """Load package checkpoints or the formal full129 ``kind='trans'`` format.

    The original format also used ``{'state': ..., 'config': ...}`` with
    ``kind/mean/std/q``. Its fixed 256/6/16 architecture is used when those
    architecture fields are absent. Baseline and learned-Q ablation weights
    are rejected. Missing/mismatched checkpoints always raise an exception.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"trained Trans-DKF checkpoint not found: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state"), dict):
        raise ValueError("checkpoint must contain a state dictionary and config")
    config = payload.get("config")
    if not isinstance(config, dict) or config.get("kind") != "trans":
        raise ValueError("only the formal kind='trans' six-variance checkpoint is supported")
    if "mean" not in config or "std" not in config or "q" not in config:
        raise ValueError("checkpoint config must include mean, std and fixed q")
    architecture = {key: config[key] for key in CovarianceConfig.__dataclass_fields__ if key in config}
    model = CovarianceNet(config["mean"], config["std"], CovarianceConfig(**architecture))
    model.load_state_dict(payload["state"], strict=True)
    for key, value in model.state_dict().items():
        if not torch.isfinite(value).all():
            raise ValueError(f"checkpoint contains non-finite tensor {key}")
    if (model.std < 0).any():
        raise ValueError("checkpoint normalization std cannot be negative")
    # Statistics are duplicated in config for inspectability; guard against
    # using a state/config pair from two different training runs.
    if not torch.equal(model.mean.cpu(), torch.tensor(config["mean"], dtype=torch.float32)) or not torch.equal(
        model.std.cpu(), torch.tensor(config["std"], dtype=torch.float32)
    ):
        raise ValueError("checkpoint state and config normalization statistics disagree")
    q_diagonal = _diagonal(config["q"], 4, "checkpoint q")
    model.to(device).eval()
    return model, q_diagonal


class LearnedNoisePredictor:
    """Predict all time steps from a supplied trained Trans-DKF checkpoint.

    The transformer uses the caller's history window. ``hidden_state`` is a
    compatibility argument; the model does not have a recurrent hidden state.
    A missing or invalid checkpoint never falls back to random weights.
    """

    def __init__(self, checkpoint: str | Path, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.model, self.q_diagonal = load_checkpoint(checkpoint, self.device)
        self.max_seq_len = self.model.config.max_seq_len
        self.q = float(self.q_diagonal[0]) if np.all(self.q_diagonal == self.q_diagonal[0]) else self.q_diagonal.copy()

    def predict(self, features: np.ndarray, hidden_state=None):
        values = _features(features)
        if len(values) > self.max_seq_len:
            raise ValueError(f"history exceeds max_seq_len={self.max_seq_len}")
        tensor = torch.as_tensor(values, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            r = self.model(tensor)[0].cpu().numpy()
        q = np.broadcast_to(self.q_diagonal, (len(values), 4)).copy()
        return r[:, :2], r[:, 2:], q, (None, None)
