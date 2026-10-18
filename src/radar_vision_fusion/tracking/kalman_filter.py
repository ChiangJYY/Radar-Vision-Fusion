from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Single-sensor updates adopt the whole branch posterior (formal replay behavior).

def _ensure_matrix(value: np.ndarray, rows: int, cols: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (rows, cols):
        raise ValueError(f"Expected shape {(rows, cols)}, got {arr.shape}")
    return arr


def _right_solve_spd(matrix: np.ndarray, right_factor: np.ndarray,
                     eps: float = 1e-6) -> np.ndarray:
    """Return right_factor @ matrix^{-1} using an SPD linear solve."""
    identity = np.eye(matrix.shape[0], dtype=np.float64)
    regularized = matrix + identity * eps
    try:
        chol = np.linalg.cholesky(regularized)
        solution = np.linalg.solve(chol.T, np.linalg.solve(chol, right_factor.T))
    except np.linalg.LinAlgError:
        solution = np.linalg.solve(regularized, right_factor.T)
    return solution.T


@dataclass
class KalmanFilter:
    state_: np.ndarray = field(default_factory=lambda: np.zeros((4, 1)))
    camera_state_: np.ndarray = field(default_factory=lambda: np.zeros((4, 1)))
    radar_state_: np.ndarray = field(default_factory=lambda: np.zeros((4, 1)))
    covariance_: np.ndarray = field(default_factory=lambda: np.zeros((4, 4)))
    camera_covariance_: np.ndarray = field(default_factory=lambda: np.zeros((4, 4)))
    radar_covariance_: np.ndarray = field(default_factory=lambda: np.zeros((4, 4)))
    camera_kalman_gain_: np.ndarray = field(default_factory=lambda: np.zeros((4, 2)))
    radar_kalman_gain_: np.ndarray = field(default_factory=lambda: np.zeros((4, 4)))
    camera_innovation: np.ndarray = field(default_factory=lambda: np.zeros((2, 1)))
    radar_innovation: np.ndarray = field(default_factory=lambda: np.zeros((4, 1)))
    cross_covariance_12: np.ndarray = field(default_factory=lambda: np.zeros((4, 4)))
    cross_covariance_2_1: np.ndarray = field(default_factory=lambda: np.zeros((4, 4)))
    radar_estimate_state: np.ndarray = field(default_factory=lambda: np.zeros((4, 1)))
    pre_state: np.ndarray = field(default_factory=lambda: np.zeros((4, 1)))
    pre_covariance: np.ndarray = field(default_factory=lambda: np.zeros((4, 4)))

    def set_state(self, state: np.ndarray) -> None:
        self.state_ = _ensure_matrix(state, 4, 1)

    def set_camera_state(self, camera_state: np.ndarray) -> None:
        self.camera_state_ = _ensure_matrix(camera_state, 4, 1)

    def set_radar_state(self, radar_state: np.ndarray) -> None:
        self.radar_state_ = _ensure_matrix(radar_state, 4, 1)

    def set_covariance(self, covariance: np.ndarray) -> None:
        self.covariance_ = _ensure_matrix(covariance, 4, 4)

    def set_camera_covariance(self, covariance: np.ndarray) -> None:
        self.camera_covariance_ = _ensure_matrix(covariance, 4, 4)

    def set_radar_covariance(self, covariance: np.ndarray) -> None:
        self.radar_covariance_ = _ensure_matrix(covariance, 4, 4)

    def state(self) -> np.ndarray:
        return self.state_

    def camera_state(self) -> np.ndarray:
        return self.camera_state_

    def radar_state(self) -> np.ndarray:
        return self.radar_state_

    def camera_kalman_gain(self) -> np.ndarray:
        return self.camera_kalman_gain_

    def radar_kalman_gain(self) -> np.ndarray:
        return self.radar_kalman_gain_

    def covariance(self) -> np.ndarray:
        return self.covariance_

    def camera_covariance(self) -> np.ndarray:
        return self.camera_covariance_

    def radar_covariance(self) -> np.ndarray:
        return self.radar_covariance_

    def predict(self, transition_matrix: np.ndarray, prediction_cov: np.ndarray) -> None:
        transition_matrix = _ensure_matrix(transition_matrix, 4, 4)
        prediction_cov = _ensure_matrix(prediction_cov, 4, 4)

        self.pre_state = self.state_.copy()
        self.pre_covariance = self.covariance_.copy()
        self.state_ = transition_matrix @ self.state_
        self.covariance_ = (
            transition_matrix @ self.covariance_ @ transition_matrix.T + prediction_cov
        )

    def camera_update(
        self,
        camera_measurement: np.ndarray,
        camera_measure_noise_cov: np.ndarray,
        camera_measure_matrix: np.ndarray,
    ) -> None:
        camera_measurement = _ensure_matrix(camera_measurement, 2, 1)
        camera_measure_noise_cov = _ensure_matrix(camera_measure_noise_cov, 2, 2)
        camera_measure_matrix = _ensure_matrix(camera_measure_matrix, 2, 4)

        # Pure camera residual; do not blend radar into the camera measurement
        self.camera_innovation = camera_measurement - camera_measure_matrix @ self.state_
        camera_innovation_covariance = (
            camera_measure_matrix
            @ self.covariance_
            @ camera_measure_matrix.T
            + camera_measure_noise_cov
        )

        self.camera_kalman_gain_ = _right_solve_spd(
            camera_innovation_covariance,
            self.covariance_ @ camera_measure_matrix.T,
        )

        self.camera_state_ = self.state_ + self.camera_kalman_gain_ @ self.camera_innovation
        innovation_gain = np.eye(4) - self.camera_kalman_gain_ @ camera_measure_matrix
        self.camera_covariance_ = (
            innovation_gain @ self.covariance_ @ innovation_gain.T
            + self.camera_kalman_gain_ @ camera_measure_noise_cov @ self.camera_kalman_gain_.T
        )
        self.camera_covariance_ = 0.5 * (
            self.camera_covariance_ + self.camera_covariance_.T
        )

    def radar_update(
        self,
        radar_measurement: np.ndarray,
        radar_measure_noise_cov: np.ndarray,
        radar_measure_matrix: np.ndarray,
    ) -> None:
        radar_measurement = _ensure_matrix(radar_measurement, 4, 1)
        radar_measure_noise_cov = _ensure_matrix(radar_measure_noise_cov, 4, 4)
        radar_measure_matrix = _ensure_matrix(radar_measure_matrix, 4, 4)

        self.radar_innovation = radar_measurement - radar_measure_matrix @ self.state_
        radar_innovation_covariance = (
            radar_measure_matrix
            @ self.covariance_
            @ radar_measure_matrix.T
            + radar_measure_noise_cov
        )
        self.radar_kalman_gain_ = _right_solve_spd(
            radar_innovation_covariance,
            self.covariance_ @ radar_measure_matrix.T,
        )

        self.radar_state_ = self.state_ + self.radar_kalman_gain_ @ self.radar_innovation
        innovation_gain = np.eye(4) - self.radar_kalman_gain_ @ radar_measure_matrix
        self.radar_covariance_ = (
            innovation_gain @ self.covariance_ @ innovation_gain.T
            + self.radar_kalman_gain_ @ radar_measure_noise_cov @ self.radar_kalman_gain_.T
        )
        self.radar_covariance_ = 0.5 * (
            self.radar_covariance_ + self.radar_covariance_.T
        )

    def track_fusion(self) -> None:
        h_c = np.zeros((2, 4))
        h_c[:2, :2] = np.eye(2)
        h_r = np.eye(4)

        camera_kalman_gain_innovation = np.eye(4) - self.camera_kalman_gain_ @ h_c
        radar_kalman_gain_innovation = np.eye(4) - self.radar_kalman_gain_ @ h_r

        self.cross_covariance_12 = (
            camera_kalman_gain_innovation @ self.covariance_ @ radar_kalman_gain_innovation.T
        )
        self.cross_covariance_2_1 = (
            radar_kalman_gain_innovation @ self.covariance_ @ camera_kalman_gain_innovation.T
        )
        cross_covariance_21 = self.cross_covariance_12.T

        cross_covariance_innovation_12 = self.camera_covariance_ - self.cross_covariance_12
        cross_covariance_innovation_21 = self.camera_covariance_ - cross_covariance_21

        innovation_covariance = (
            self.camera_covariance_
            + self.radar_covariance_
            - self.cross_covariance_12
            - cross_covariance_21
        )
        gain = _right_solve_spd(
            innovation_covariance, cross_covariance_innovation_12, eps=1e-5
        )

        self.state_ = (
            self.camera_state_
            + gain @ (self.radar_state_ - self.camera_state_)
        )
        self.covariance_ = self.camera_covariance_ - gain @ cross_covariance_innovation_21
        self.covariance_ = 0.5 * (self.covariance_ + self.covariance_.T)

    def robust_fusion(self, mask_cam: float, mask_rad: float) -> None:
        """Fuse available branches; keep the predicted state if both are missing.

        Camera and radar branch updates share the same prior. With a single
        sensor, its complete posterior is adopted. This corresponds to the
        formal experiment's VEL_ALPHA=0 behavior, including velocity updates.
        Missing sensors are represented by masks, never by a zero-state test.
        """
        has_cam = mask_cam > 0.5
        has_rad = mask_rad > 0.5
        if has_cam and has_rad:
            self.track_fusion()
        elif has_rad:
            self.state_ = self.radar_state_.copy()
            self.covariance_ = self.radar_covariance_.copy()
            self.camera_state_ = self.state_.copy()
            self.camera_covariance_ = self.covariance_.copy()
        elif has_cam:
            self.state_ = self.camera_state_.copy()
            self.covariance_ = self.camera_covariance_.copy()
            self.radar_state_ = self.state_.copy()
            self.radar_covariance_ = self.covariance_.copy()
