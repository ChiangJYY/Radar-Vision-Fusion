"""Classical adaptive and robust filtering baselines for fair tracker studies."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable

import numpy as np

from .noise import FIXED_Q, FIXED_R_CAM, FIXED_R_RAD
from ..tracking.kalman_filter import KalmanFilter


@dataclass(frozen=True)
class BaselinePredictor:
    """Configuration object with the same covariance interface as Trans-DKF."""

    filter_mode: str
    fixed_q_scale: float = 1.0
    iae_window: int = 10
    iae_blend: float = 0.5
    sage_forgetting: float = 0.95
    imq_threshold: float = 3.0
    imm_scales: tuple[float, ...] = (0.25, 1.0, 4.0)
    imm_stay_probability: float = 0.90

    def predict(self, features, hidden_state=None):
        length = len(np.atleast_2d(features))
        r_cam = np.tile(FIXED_R_CAM, (length, 1))
        r_rad = np.tile(FIXED_R_RAD, (length, 1))
        q_val = np.tile(FIXED_Q * self.fixed_q_scale, (length, 1))
        return r_cam, r_rad, q_val, (None, None)


class MeasurementNoiseAdapter:
    """Per-track diagonal-R adaptation with positive-definite safeguards."""

    def __init__(self, config: BaselinePredictor) -> None:
        self.config = config
        self.history: Dict[str, Deque[np.ndarray]] = {
            "camera": deque(maxlen=config.iae_window),
            "radar": deque(maxlen=config.iae_window),
        }
        self.current = {
            "camera": FIXED_R_CAM.astype(np.float64).copy(),
            "radar": FIXED_R_RAD.astype(np.float64).copy(),
        }
        self.steps = {"camera": 0, "radar": 0}

    @staticmethod
    def _safe_diag(values: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(values, dtype=np.float64), 1e-2, 100.0)

    def covariance(
        self,
        sensor: str,
        measurement: np.ndarray,
        measurement_matrix: np.ndarray,
        prior_state: np.ndarray,
        prior_covariance: np.ndarray,
        base_covariance: np.ndarray,
    ) -> np.ndarray:
        mode = self.config.filter_mode
        if mode not in {"iae", "sage_husa", "imq"}:
            return base_covariance

        innovation = measurement - measurement_matrix @ prior_state
        innovation_vec = innovation.reshape(-1)
        projected = measurement_matrix @ prior_covariance @ measurement_matrix.T
        base_diag = np.diag(base_covariance).astype(np.float64)

        if mode == "iae":
            history = self.history[sensor]
            history.append(innovation_vec.copy())
            if len(history) < 3:
                return np.diag(self.current[sensor])
            second_moment = np.mean(np.square(np.stack(history)), axis=0)
            candidate = self._safe_diag(second_moment - np.diag(projected))
            blend = float(np.clip(self.config.iae_blend, 0.0, 1.0))
            self.current[sensor] = self._safe_diag(
                (1.0 - blend) * self.current[sensor] + blend * candidate
            )
            return np.diag(self.current[sensor])

        if mode == "sage_husa":
            self.steps[sensor] += 1
            forgetting = float(np.clip(self.config.sage_forgetting, 0.5, 0.9999))
            gain = (1.0 - forgetting) / (
                1.0 - forgetting ** (self.steps[sensor] + 1)
            )
            candidate = self._safe_diag(
                np.square(innovation_vec) - np.diag(projected)
            )
            self.current[sensor] = self._safe_diag(
                (1.0 - gain) * self.current[sensor] + gain * candidate
            )
            return np.diag(self.current[sensor])

        # Inverse-multiquadric robust weighting used in the ICML-2024
        # authors' tracking implementation: large innovations inflate R.
        scale = max(float(self.config.imq_threshold), 1e-3)
        standardized_energy = float(
            innovation_vec.T
            @ np.linalg.solve(np.diag(base_diag) + np.eye(len(base_diag)) * 1e-6, innovation_vec)
        )
        weight = scale * scale / (scale * scale + standardized_energy)
        weight = float(np.clip(weight, 1e-3, 1.0))
        return np.diag(self._safe_diag(base_diag / weight))


def _gaussian_log_likelihood(innovation: np.ndarray, covariance: np.ndarray) -> float:
    covariance = 0.5 * (covariance + covariance.T) + np.eye(covariance.shape[0]) * 1e-6
    try:
        chol = np.linalg.cholesky(covariance)
        solved = np.linalg.solve(chol, innovation)
        log_det = 2.0 * np.log(np.diag(chol)).sum()
        quadratic = float(solved.T @ solved)
    except np.linalg.LinAlgError:
        sign, log_det = np.linalg.slogdet(covariance)
        if sign <= 0:
            return -1e6
        quadratic = float(innovation.T @ np.linalg.solve(covariance, innovation))
    dimension = innovation.shape[0]
    return -0.5 * (dimension * np.log(2.0 * np.pi) + log_det + quadratic)


class InteractingMultipleModelFilter:
    """Three-model CV IMM with the KalmanFilter-compatible interface."""

    def __init__(self, scales: Iterable[float], stay_probability: float = 0.9) -> None:
        self.scales = np.asarray(tuple(scales), dtype=np.float64)
        if len(self.scales) < 2 or np.any(self.scales <= 0):
            raise ValueError("IMM requires at least two positive process-noise scales")
        self.filters = [KalmanFilter() for _ in self.scales]
        self.probabilities = np.full(len(self.filters), 1.0 / len(self.filters))
        stay = float(np.clip(stay_probability, 0.5, 0.999))
        off = (1.0 - stay) / (len(self.filters) - 1)
        self.transition = np.full((len(self.filters), len(self.filters)), off)
        np.fill_diagonal(self.transition, stay)
        self.mixed_prior = self.probabilities.copy()
        self.log_likelihood = np.zeros(len(self.filters), dtype=np.float64)
        self.state_ = np.zeros((4, 1), dtype=np.float64)
        self.covariance_ = np.eye(4, dtype=np.float64)

    def _broadcast(self, method: str, value: np.ndarray) -> None:
        for model in self.filters:
            getattr(model, method)(value.copy())

    def set_state(self, value):
        self.state_ = np.asarray(value, dtype=np.float64).copy()
        self._broadcast("set_state", self.state_)

    def set_covariance(self, value):
        self.covariance_ = np.asarray(value, dtype=np.float64).copy()
        self._broadcast("set_covariance", self.covariance_)

    def set_camera_state(self, value): self._broadcast("set_camera_state", value)
    def set_radar_state(self, value): self._broadcast("set_radar_state", value)
    def set_camera_covariance(self, value): self._broadcast("set_camera_covariance", value)
    def set_radar_covariance(self, value): self._broadcast("set_radar_covariance", value)
    def state(self): return self.state_
    def covariance(self): return self.covariance_
    def camera_state(self): return sum(p * f.camera_state() for p, f in zip(self.probabilities, self.filters))
    def radar_state(self): return sum(p * f.radar_state() for p, f in zip(self.probabilities, self.filters))

    def predict(self, transition_matrix: np.ndarray, prediction_cov: np.ndarray) -> None:
        n_models = len(self.filters)
        normalization = self.probabilities @ self.transition
        normalization = np.clip(normalization, 1e-12, None)
        mixing = (
            self.probabilities[:, None] * self.transition / normalization[None, :]
        )
        states = [model.state().copy() for model in self.filters]
        covariances = [model.covariance().copy() for model in self.filters]
        for destination, model in enumerate(self.filters):
            mean = sum(mixing[source, destination] * states[source] for source in range(n_models))
            covariance = np.zeros((4, 4), dtype=np.float64)
            for source in range(n_models):
                delta = states[source] - mean
                covariance += mixing[source, destination] * (
                    covariances[source] + delta @ delta.T
                )
            model.set_state(mean)
            model.set_covariance(covariance)
            model.set_camera_state(mean.copy())
            model.set_radar_state(mean.copy())
            model.set_camera_covariance(covariance.copy())
            model.set_radar_covariance(covariance.copy())
            model.predict(transition_matrix, prediction_cov * self.scales[destination])
        self.mixed_prior = normalization
        self.log_likelihood.fill(0.0)
        self._moment_match(self.mixed_prior)

    def camera_update(self, measurement, noise_covariance, measurement_matrix):
        for index, model in enumerate(self.filters):
            innovation = measurement - measurement_matrix @ model.state()
            covariance = measurement_matrix @ model.covariance() @ measurement_matrix.T + noise_covariance
            self.log_likelihood[index] += _gaussian_log_likelihood(innovation, covariance)
            model.camera_update(measurement, noise_covariance, measurement_matrix)

    def radar_update(self, measurement, noise_covariance, measurement_matrix):
        for index, model in enumerate(self.filters):
            innovation = measurement - measurement_matrix @ model.state()
            covariance = measurement_matrix @ model.covariance() @ measurement_matrix.T + noise_covariance
            self.log_likelihood[index] += _gaussian_log_likelihood(innovation, covariance)
            model.radar_update(measurement, noise_covariance, measurement_matrix)

    def robust_fusion(self, mask_cam: float, mask_rad: float) -> None:
        for model in self.filters:
            model.robust_fusion(mask_cam, mask_rad)
        logits = np.log(np.clip(self.mixed_prior, 1e-12, None)) + self.log_likelihood
        logits -= np.max(logits)
        probabilities = np.exp(logits)
        self.probabilities = probabilities / np.sum(probabilities)
        self._moment_match(self.probabilities)

    def _moment_match(self, probabilities: np.ndarray) -> None:
        mean = sum(p * model.state() for p, model in zip(probabilities, self.filters))
        covariance = np.zeros((4, 4), dtype=np.float64)
        for probability, model in zip(probabilities, self.filters):
            delta = model.state() - mean
            covariance += probability * (model.covariance() + delta @ delta.T)
        self.state_ = mean
        self.covariance_ = 0.5 * (covariance + covariance.T)
