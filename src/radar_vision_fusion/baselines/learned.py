"""KalmanNet and Split-KalmanNet adaptations for heterogeneous observations."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from .noise import FIXED_Q, FIXED_R_CAM, FIXED_R_RAD


def observation_matrix(device, dtype):
    matrix = torch.zeros(6, 4, device=device, dtype=dtype)
    matrix[0, 0] = matrix[1, 1] = 1.0
    matrix[2, 0] = matrix[3, 1] = 1.0
    matrix[4, 2] = matrix[5, 3] = 1.0
    return matrix


def observations_from_features(features: torch.Tensor):
    z = torch.stack(
        (
            features[..., 0], features[..., 1],
            features[..., 9], features[..., 10],
            features[..., 11], features[..., 12],
        ),
        dim=-1,
    )
    camera = features[..., 18:19]
    radar = features[..., 19:20]
    mask = torch.cat((camera, camera, radar, radar, radar, radar), dim=-1)
    return z, mask


def initial_state(features: torch.Tensor) -> torch.Tensor:
    camera = features[:, 0, 18]
    radar = features[:, 0, 19]
    denominator = torch.clamp(camera + radar, min=1e-6)
    state = torch.zeros(features.shape[0], 4, device=features.device, dtype=features.dtype)
    state[:, 0] = (camera * features[:, 0, 0] + radar * features[:, 0, 9]) / denominator
    state[:, 1] = (camera * features[:, 0, 1] + radar * features[:, 0, 10]) / denominator
    state[:, 2] = radar * features[:, 0, 11]
    state[:, 3] = radar * features[:, 0, 12]
    return state


def transition(dt: torch.Tensor) -> torch.Tensor:
    batch = dt.shape[0]
    matrix = torch.eye(4, device=dt.device, dtype=dt.dtype).unsqueeze(0).repeat(batch, 1, 1)
    matrix[:, 0, 2] = dt
    matrix[:, 1, 3] = dt
    return matrix


def normalized(vector: torch.Tensor) -> torch.Tensor:
    return Fnn.normalize(vector, p=2.0, dim=-1, eps=1e-6)


class KalmanNetBaseline(nn.Module):
    """GRU-aided Kalman gain estimator following the KalmanNet principle."""

    kind = "kalmannet"

    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRUCell(20, hidden_size)
        self.gain_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, 24)
        )

    def forward(self, features: torch.Tensor, timestamps: torch.Tensor):
        batch, steps, _ = features.shape
        z_all, mask_all = observations_from_features(features)
        h_matrix = observation_matrix(features.device, features.dtype)
        state = initial_state(features)
        hidden = torch.zeros(batch, self.hidden_size, device=features.device, dtype=features.dtype)
        previous_z = z_all[:, 0]
        previous_state = state
        previous_correction = torch.zeros_like(state)
        outputs = []
        for index in range(steps):
            dt = torch.full_like(timestamps[:, index], 1e-3) if index == 0 else torch.clamp(
                timestamps[:, index] - timestamps[:, index - 1], min=1e-3
            )
            f_matrix = transition(dt)
            predicted = torch.bmm(f_matrix, state.unsqueeze(-1)).squeeze(-1)
            predicted_measurement = predicted @ h_matrix.T
            measurement = mask_all[:, index] * z_all[:, index] + (1.0 - mask_all[:, index]) * predicted_measurement
            innovation = measurement - predicted_measurement
            measurement_delta = measurement - previous_z
            state_delta = predicted - previous_state
            network_input = torch.cat(
                (
                    normalized(innovation), normalized(measurement_delta),
                    normalized(state_delta), normalized(previous_correction),
                ), dim=-1,
            )
            hidden = self.gru(self.input_norm(network_input), hidden)
            gain = torch.tanh(self.gain_head(hidden)).view(batch, 4, 6)
            gain = gain * mask_all[:, index].unsqueeze(1)
            correction = torch.bmm(gain, innovation.unsqueeze(-1)).squeeze(-1)
            state = predicted + correction
            outputs.append(state)
            previous_z = measurement
            previous_state = predicted
            previous_correction = correction
        return torch.stack(outputs, dim=1), None


def lower_triangular(raw: torch.Tensor, dimension: int) -> torch.Tensor:
    batch = raw.shape[0]
    matrix = torch.zeros(batch, dimension, dimension, device=raw.device, dtype=raw.dtype)
    indices = torch.tril_indices(dimension, dimension, device=raw.device)
    matrix[:, indices[0], indices[1]] = raw
    diagonal = torch.arange(dimension, device=raw.device)
    matrix[:, diagonal, diagonal] = Fnn.softplus(matrix[:, diagonal, diagonal]) + 1e-2
    return matrix


class SplitKalmanNetBaseline(nn.Module):
    """Split recurrent prior/innovation statistics with analytical gain."""

    kind = "split_kalmannet"

    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.prior_gru = nn.GRUCell(8, hidden_size)
        self.innovation_gru = nn.GRUCell(12, hidden_size)
        self.prior_head = nn.Linear(hidden_size, 10)
        self.innovation_head = nn.Linear(hidden_size, 21)

    def forward(self, features: torch.Tensor, timestamps: torch.Tensor):
        batch, steps, _ = features.shape
        z_all, mask_all = observations_from_features(features)
        h_matrix = observation_matrix(features.device, features.dtype)
        state = initial_state(features)
        prior_hidden = torch.zeros(batch, self.hidden_size, device=features.device, dtype=features.dtype)
        innovation_hidden = torch.zeros_like(prior_hidden)
        previous_z = z_all[:, 0]
        previous_state = state
        previous_correction = torch.zeros_like(state)
        outputs, covariances = [], []
        for index in range(steps):
            dt = torch.full_like(timestamps[:, index], 1e-3) if index == 0 else torch.clamp(
                timestamps[:, index] - timestamps[:, index - 1], min=1e-3
            )
            predicted = torch.bmm(transition(dt), state.unsqueeze(-1)).squeeze(-1)
            predicted_measurement = predicted @ h_matrix.T
            measurement = mask_all[:, index] * z_all[:, index] + (1.0 - mask_all[:, index]) * predicted_measurement
            innovation = measurement - predicted_measurement
            measurement_delta = measurement - previous_z
            prior_input = torch.cat((normalized(predicted - previous_state), normalized(previous_correction)), dim=-1)
            innovation_input = torch.cat((normalized(innovation), normalized(measurement_delta)), dim=-1)
            prior_hidden = self.prior_gru(prior_input, prior_hidden)
            innovation_hidden = self.innovation_gru(innovation_input, innovation_hidden)
            lp = lower_triangular(self.prior_head(prior_hidden), 4)
            ls = lower_triangular(self.innovation_head(innovation_hidden), 6)
            prior_covariance = torch.bmm(lp, lp.transpose(1, 2))
            innovation_covariance = torch.bmm(ls, ls.transpose(1, 2))
            h_batch = h_matrix.unsqueeze(0).expand(batch, -1, -1) * mask_all[:, index].unsqueeze(-1)
            right = torch.bmm(prior_covariance, h_batch.transpose(1, 2))
            regularized = innovation_covariance + torch.eye(6, device=features.device, dtype=features.dtype).unsqueeze(0) * 1e-3
            gain = torch.linalg.solve(regularized, right.transpose(1, 2)).transpose(1, 2)
            gain = torch.clamp(torch.nan_to_num(gain), min=-2.0, max=2.0)
            correction = torch.bmm(gain, innovation.unsqueeze(-1)).squeeze(-1)
            state = predicted + correction
            outputs.append(state)
            covariances.append(prior_covariance)
            previous_z = measurement
            previous_state = predicted
            previous_correction = correction
        return torch.stack(outputs, dim=1), torch.stack(covariances, dim=1)


def build_model(kind: Literal["kalmannet", "split_kalmannet"], hidden_size: int = 128):
    if kind == "kalmannet":
        return KalmanNetBaseline(hidden_size)
    if kind == "split_kalmannet":
        return SplitKalmanNetBaseline(hidden_size)
    raise ValueError(kind)


class LearnedStateInferenceHandler:
    """Stateless causal-window inference wrapper used by the unchanged tracker."""

    def __init__(
        self, checkpoint: str | Path, kind: str, fixed_q_scale: float = 1.0
    ) -> None:
        self.filter_mode = kind
        self.fixed_q_scale = float(fixed_q_scale)
        if self.fixed_q_scale <= 0.0:
            raise ValueError("fixed_q_scale must be positive")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(str(checkpoint), map_location=self.device, weights_only=True)
        self.model = build_model(kind, int(payload.get("hidden_size", 128))).to(self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()

    def predict(self, features, hidden_state=None):
        length = len(np.atleast_2d(features))
        return (
            np.tile(FIXED_R_CAM, (length, 1)),
            np.tile(FIXED_R_RAD, (length, 1)),
            np.tile(FIXED_Q * self.fixed_q_scale, (length, 1)),
            (None, None),
        )

    def predict_state(self, features: np.ndarray, timestamps: np.ndarray):
        feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        timestamp_tensor = torch.as_tensor(timestamps, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            states, covariances = self.model(feature_tensor, timestamp_tensor)
        state = states[0, -1].cpu().numpy().reshape(4, 1)
        if covariances is None:
            covariance = np.eye(4, dtype=np.float64)
        else:
            covariance = covariances[0, -1].cpu().numpy().astype(np.float64)
        return state.astype(np.float64), covariance
