"""Per-object temporal features, dual Kalman fusion, and track lifecycle.

Derived from DL_RVFusion/track.py. The network is injected explicitly so a
missing checkpoint cannot silently instantiate a randomly initialized model.
"""
from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from .kalman_filter import KalmanFilter
from .math_utils import Vec2d
from .sensors_detection import CameraDetection, RadarDetection


class NoisePredictor(Protocol):
    """Return (camera_R[T,2], radar_R[T,4], Q[T,4], hidden) for features[T,23]."""

    def predict(self, features: np.ndarray, hidden_state=None): ...


@dataclass(frozen=True)
class TrackConfig:
    init_position_variance: float = 4.0
    init_velocity_variance: float = 2.0
    process_noise: tuple[float, float, float, float] = (0.025, 0.025, 0.025, 0.025)
    camera_noise: tuple[float, float] = (3.0, 3.0)
    radar_noise: tuple[float, float, float, float] = (3.0, 3.0, 1.0, 1.0)
    temporal_window: int = 25
    confirm_hits: int = 2
    tentative_lost_seconds: float = 0.5
    confirmed_lost_seconds: float = 1.0

    def __post_init__(self) -> None:
        _validate_variances(self.camera_noise, 2, "camera_noise")
        _validate_variances(self.radar_noise, 4, "radar_noise")
        _validate_variances(self.process_noise, 4, "process_noise", allow_zero=True)
        initial = [self.init_position_variance, self.init_velocity_variance]
        _validate_variances(initial, 2, "initial covariance")
        if self.temporal_window < 1 or self.confirm_hits < 1:
            raise ValueError("temporal_window and confirm_hits must be positive")
        timeouts = np.asarray([self.tentative_lost_seconds, self.confirmed_lost_seconds])
        if not np.isfinite(timeouts).all() or (timeouts < 0).any():
            raise ValueError("Track timeouts must be finite and nonnegative")


def _validate_variances(values, size: int, name: str, allow_zero: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {size} finite variances")
    if (array < 0).any() or (not allow_zero and (array == 0).any()):
        raise ValueError(f"{name} contains invalid variances")
    return array


class TrackState(Enum):
    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3


class FusionTrack:
    next_track_id = 1

    def __init__(self, detection: CameraDetection | RadarDetection,
                 predictor: NoisePredictor | None = None,
                 config: TrackConfig | None = None, *,
                 track_id: int | None = None) -> None:
        if not isinstance(detection, (CameraDetection, RadarDetection)):
            raise TypeError("Unsupported detection type")
        self.config = config or TrackConfig()
        self.predictor = predictor
        self.filter_mode = "transdkf" if predictor is not None else "fixed_noise"
        if track_id is None:
            self.id_ = FusionTrack.next_track_id
            FusionTrack.next_track_id += 1
        else:
            self.id_ = track_id
        self.created_timestamp_ = float(detection.timestamp)
        self.timestamp_ = float(detection.timestamp)
        self.last_update_timestamp_ = float(detection.timestamp)
        self.last_radar_update_timestamp_ = 0.0
        self.last_camera_update_timestamp_ = 0.0
        self.position_ = detection.position
        self.velocity_ = Vec2d()
        self.acceleration_ = Vec2d()
        self.confs_ = 0.0
        self.associated_radar_detections_cnt_ = 0
        self.associated_camera_detections_cnt_ = 0
        self.feature_buffer = deque(maxlen=self.config.temporal_window)
        self.timestamp_buffer = deque(maxlen=self.config.temporal_window)
        self.state = TrackState.TENTATIVE
        self.hits = 1
        self.age_steps = 0
        if self.config.confirm_hits == 1:
            self.state = TrackState.CONFIRMED

        state = np.array([[self.position_.x], [self.position_.y], [0.0], [0.0]], dtype=np.float64)
        covariance = np.diag([self.config.init_position_variance] * 2 +
                             [self.config.init_velocity_variance] * 2)
        if isinstance(detection, RadarDetection):
            state[2:, 0] = [detection.velocity.x, detection.velocity.y]
            self.velocity_ = detection.velocity
            self.acceleration_ = detection.acceleration
            self.last_radar_update_timestamp_ = detection.timestamp
            self.associated_radar_detections_cnt_ = 1
        else:
            self.last_camera_update_timestamp_ = detection.timestamp
            self.associated_camera_detections_cnt_ = 1
            self.confs_ = detection.score
        self.motion_kf_ = KalmanFilter()
        self.motion_kf_.set_state(state)
        self.motion_kf_.set_covariance(covariance)
        self.motion_kf_.set_camera_state(state.copy())
        self.motion_kf_.set_radar_state(state.copy())
        self.motion_kf_.set_camera_covariance(covariance.copy())
        self.motion_kf_.set_radar_covariance(covariance.copy())

    @staticmethod
    def feature_vector(camera: np.ndarray | None, radar: np.ndarray | None) -> np.ndarray:
        """cam[9] + radar[9] + valid_cam + valid_rad + dx + dy + distance."""
        cam = np.zeros(9, np.float32) if camera is None else np.asarray(camera, np.float32)
        rad = np.zeros(9, np.float32) if radar is None else np.asarray(radar, np.float32)
        if cam.shape != (9,) or rad.shape != (9,):
            raise ValueError("Camera and radar feature vectors must have 9 components")
        both_present = camera is not None and radar is not None
        dx, dy = (cam[:2] - rad[:2]) if both_present else (0.0, 0.0)
        extra = [float(camera is not None), float(radar is not None), dx, dy, math.hypot(dx, dy)]
        result = np.concatenate([cam, rad, np.asarray(extra, np.float32)])
        if not np.isfinite(result).all():
            raise ValueError("Detection features must be finite")
        return result

    def _run_temporal_inference(self, cur_cam: np.ndarray | None,
                                cur_rad: np.ndarray | None,
                                update_buffer: bool = True):
        current = self.feature_vector(cur_cam, cur_rad)
        if update_buffer:
            self.feature_buffer.append(current.copy())
            sequence = np.asarray(self.feature_buffer, dtype=np.float32)
        else:
            sequence = np.asarray((list(self.feature_buffer) + [current])[-self.config.temporal_window:],
                                  dtype=np.float32)
        if self.predictor is None:
            return (np.asarray(self.config.camera_noise, np.float64),
                    np.asarray(self.config.radar_noise, np.float64),
                    np.asarray(self.config.process_noise, np.float64))
        r_cam_seq, r_rad_seq, q_seq, _ = self.predictor.predict(sequence, hidden_state=None)
        outputs = []
        for values, size, label, allow_zero in (
            (r_cam_seq, 2, "camera_R", False),
            (r_rad_seq, 4, "radar_R", False),
            (q_seq, 4, "Q", True),
        ):
            array = np.asarray(values)
            if array.shape != (len(sequence), size):
                raise ValueError(f"Predictor {label} must have shape {(len(sequence), size)}, got {array.shape}")
            outputs.append(_validate_variances(array[-1], size, label, allow_zero))
        return tuple(outputs)

    def set_id(self, value: int) -> None:
        self.id_ = value

    def set_confs(self, value: float) -> None:
        self.confs_ = value

    def id(self) -> int:
        return self.id_

    def created_timestamp(self) -> float:
        return self.created_timestamp_

    def timestamp(self) -> float:
        return self.timestamp_

    def last_update_timestamp(self) -> float:
        return self.last_update_timestamp_

    def age(self) -> float:
        return self.timestamp_ - self.created_timestamp_

    def position(self) -> Vec2d:
        return self.position_

    def camera_position(self) -> Vec2d:
        state = self.motion_kf_.camera_state()
        return Vec2d(float(state[0, 0]), float(state[1, 0]))

    def radar_position(self) -> Vec2d:
        state = self.motion_kf_.radar_state()
        return Vec2d(float(state[0, 0]), float(state[1, 0]))

    def velocity(self) -> Vec2d:
        return self.velocity_

    def acceleration(self) -> Vec2d:
        return self.acceleration_

    def confs(self) -> float:
        return self.confs_

    def _refresh_state(self) -> None:
        state = self.motion_kf_.state()
        self.position_ = Vec2d(float(state[0, 0]), float(state[1, 0]))
        self.velocity_ = Vec2d(float(state[2, 0]), float(state[3, 0]))

    def _predict_with_noise(self, timestamp: float, q: np.ndarray) -> None:
        dt = timestamp - self.timestamp_
        if dt <= 0:
            return
        transition = np.eye(4, dtype=np.float64)
        transition[0, 2] = transition[1, 3] = dt
        self.motion_kf_.predict(transition, np.diag(q))
        self.timestamp_ = float(timestamp)
        self.age_steps += 1
        self._refresh_state()

    def predict(self, timestamp: float) -> None:
        if not np.isfinite(timestamp) or timestamp < self.timestamp_:
            raise ValueError("Track timestamps must be finite and nondecreasing")
        if timestamp > self.timestamp_:
            _, _, q = self._run_temporal_inference(None, None, update_buffer=False)
            self._predict_with_noise(timestamp, q)

    def update_synced(self, timestamp: float,
                      camera_detection: CameraDetection | None,
                      radar_detection: RadarDetection | None) -> None:
        """Update both branches from the same prior, with None for missing sensors."""
        if not np.isfinite(timestamp) or timestamp < self.timestamp_:
            raise ValueError("Track timestamps must be finite and nondecreasing")
        cur_cam = None
        cur_rad = None
        if camera_detection is not None:
            camera = camera_detection
            x, y = camera.position.x, camera.position.y
            cur_cam = np.array([x, y, camera.position.length(), math.atan2(y, x),
                                camera.bbox.a1, camera.bbox.a2, camera.bbox.a3, camera.bbox.a4,
                                camera.score], dtype=np.float32)
            self.confs_ = camera.score
        if radar_detection is not None:
            radar = radar_detection
            x, y = radar.position.x, radar.position.y
            cur_rad = np.array([x, y, radar.velocity.x, radar.velocity.y,
                                radar.acceleration.x, radar.acceleration.y,
                                radar.position.length(), math.atan2(y, x), radar.confidence_level],
                               dtype=np.float32)
            self.acceleration_ = radar.acceleration
        r_cam, r_rad, q = self._run_temporal_inference(cur_cam, cur_rad)
        self.timestamp_buffer.append(float(timestamp))
        self._predict_with_noise(timestamp, q)
        has_cam = camera_detection is not None
        has_rad = radar_detection is not None
        if has_cam:
            measure = np.zeros((2, 4), dtype=np.float64)
            measure[:2, :2] = np.eye(2)
            self.motion_kf_.camera_update(cur_cam[:2].astype(np.float64).reshape(2, 1),
                                          np.diag(r_cam), measure)
            self.last_camera_update_timestamp_ = timestamp
            self.associated_camera_detections_cnt_ += 1
        if has_rad:
            self.motion_kf_.radar_update(cur_rad[:4].astype(np.float64).reshape(4, 1),
                                         np.diag(r_rad), np.eye(4))
            self.last_radar_update_timestamp_ = timestamp
            self.associated_radar_detections_cnt_ += 1
        self.motion_kf_.robust_fusion(float(has_cam), float(has_rad))
        self._refresh_state()
        if has_cam or has_rad:
            self.last_update_timestamp_ = timestamp
            self.hits += 1
            if self.state == TrackState.TENTATIVE and self.hits >= self.config.confirm_hits:
                self.state = TrackState.CONFIRMED

    def is_lost(self) -> bool:
        threshold = (self.config.tentative_lost_seconds if self.state == TrackState.TENTATIVE
                     else self.config.confirmed_lost_seconds)
        if self.state == TrackState.DELETED or self.timestamp_ - self.last_update_timestamp_ > threshold:
            self.state = TrackState.DELETED
            return True
        return False

    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    @staticmethod
    def reset_track_id() -> None:
        FusionTrack.next_track_id = 1

    def __deepcopy__(self, memo):
        result = self.__class__.__new__(self.__class__)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, None if key == "predictor" else copy.deepcopy(value, memo))
        return result


class FusionTrackFrame:
    def __init__(self, index: int, timestamp: float, detections: list[FusionTrack]) -> None:
        self.index_ = index
        self.timestamp_ = timestamp
        self.detections_ = detections

    def index(self) -> int:
        return self.index_

    def timestamp(self) -> float:
        return self.timestamp_

    def detections(self) -> list[FusionTrack]:
        return self.detections_
