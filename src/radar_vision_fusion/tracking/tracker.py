"""Frame alignment, Hungarian association, and multi-object track management.

Derived from DL_RVFusion/tracker.py. Fusion mode retains the original
camera-driven track initiation policy; radar_only initiates from radar.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from typing import Sequence

from .hungarian import hungarian_minimize
from .sensors_detection import CameraDetection, CameraFrame, RadarDetection, RadarFrame
from .track import FusionTrack, FusionTrackFrame, NoisePredictor, TrackConfig


@dataclass(frozen=True)
class TrackerConfig:
    camera_gating: float = 10.0
    radar_gating: float = 10.0
    frame_association_gating: float = 5.0
    retain_history: bool = True

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) or value <= 0 for value in
               (self.camera_gating, self.radar_gating, self.frame_association_gating)):
            raise ValueError("Spatial and temporal gates must be finite and positive")


def _distance_camera(track: FusionTrack, detection: CameraDetection) -> float:
    return (detection.position - track.position()).length()


def _distance_radar(track: FusionTrack, detection: RadarDetection) -> float:
    return (detection.position - track.position()).length()


def _camera_distance_matrix(tracks: Sequence[FusionTrack], detections: Sequence[CameraDetection]):
    return [[_distance_camera(track, detection) for detection in detections] for track in tracks]


def _radar_distance_matrix(tracks: Sequence[FusionTrack], detections: Sequence[RadarDetection]):
    return [[_distance_radar(track, detection) for detection in detections] for track in tracks]


class Tracker:
    def __init__(self, scene_name: str = "demo", predictor: NoisePredictor | None = None,
                 sensor_mode: str = "fusion", config: TrackerConfig | None = None,
                 track_config: TrackConfig | None = None) -> None:
        if sensor_mode not in {"fusion", "camera_only", "radar_only"}:
            raise ValueError(f"Unsupported sensor mode: {sensor_mode}")
        self.scene_name_ = scene_name
        self.predictor = predictor
        self.sensor_mode = sensor_mode
        self.config = config or TrackerConfig()
        self.track_config = track_config or TrackConfig()
        self.frame_index_ = 0
        self.tracks_num_ = 0
        self.frame_timestamp_ = 0.0
        self.tracks_: list[FusionTrack] = []
        self.fusion_tracks_frames_: list[FusionTrackFrame] = []
        self._last_timestamp: float | None = None
        self._next_track_id = 1

    def frame_index(self) -> int:
        return self.frame_index_

    def tracks_num(self) -> int:
        return self.tracks_num_

    def timestamp(self) -> float:
        return self.frame_timestamp_

    def step(self, camera_frame: CameraFrame | None,
             radar_frame: RadarFrame | None) -> list[FusionTrack]:
        """Process one aligned frame and return confirmed-track snapshots.

        Use None for an unavailable sensor. The later sensor timestamp is the
        shared update time, so a new track cannot be born in the future.
        run() performs nearest timestamp alignment for separate streams.
        """
        if camera_frame is None and radar_frame is None:
            raise ValueError("At least one frame is required; use an empty frame for a missing update")
        reference = camera_frame if camera_frame is not None else radar_frame
        sensor_times = [frame.timestamp for frame in (camera_frame, radar_frame) if frame is not None]
        if not all(math.isfinite(value) for value in sensor_times):
            raise ValueError("Frame timestamps must be finite and strictly increasing")
        timestamp = max(sensor_times)
        if (
            self._last_timestamp is not None and timestamp <= self._last_timestamp
        ):
            raise ValueError("Frame timestamps must be finite and strictly increasing")
        camera_frame = camera_frame or CameraFrame(reference.index, timestamp, [])
        radar_frame = radar_frame or RadarFrame(reference.index, timestamp, [])
        self.frame_index_ = reference.index
        self.frame_timestamp_ = timestamp
        self._last_timestamp = timestamp
        self.predict_tracks(timestamp)
        # Expire before association: a late observation starts a new lifecycle
        # rather than reviving an ID that exceeded its allowed missing time.
        self.tracks_ = [track for track in self.tracks_ if not track.is_lost()]
        association = self.update_data_association(camera_frame, radar_frame)
        cam_pairs, rad_pairs, unmatched_cam_tracks, unmatched_rad_tracks, unmatched_cam, unmatched_rad = association
        self.update_tracks(camera_frame, radar_frame, cam_pairs, rad_pairs)
        self.management_tracks(camera_frame, radar_frame, unmatched_cam_tracks,
                               unmatched_cam, unmatched_rad_tracks, unmatched_rad)
        self.tracks_num_ = len(self.tracks_)
        return self.publish_tracks()

    def run(self, camera_frames: Sequence[CameraFrame],
            radar_frames: Sequence[RadarFrame]) -> list[FusionTrackFrame]:
        """Align streams and process matched and unmatched frames without ground truth."""
        camera_frames = list(camera_frames)
        radar_frames = list(radar_frames)
        pairs, unmatched_cam, unmatched_rad = self.time_alignment(camera_frames, radar_frames)
        work = [(max(camera_frames[i].timestamp, radar_frames[j].timestamp),
                 camera_frames[i], radar_frames[j]) for i, j in pairs]
        work.extend((camera_frames[i].timestamp, camera_frames[i], None) for i in unmatched_cam)
        work.extend((radar_frames[j].timestamp, None, radar_frames[j]) for j in unmatched_rad)
        results = []
        for _, camera, radar in sorted(work, key=lambda item: item[0]):
            tracks = self.step(camera, radar)
            results.append(FusionTrackFrame(self.frame_index_, self.frame_timestamp_, tracks))
        return results

    def predict_tracks(self, timestamp: float) -> None:
        for track in self.tracks_:
            track.predict(timestamp)

    def _detections(self, camera_frame: CameraFrame, radar_frame: RadarFrame):
        camera = camera_frame.detections if self.sensor_mode != "radar_only" else []
        radar = radar_frame.detections if self.sensor_mode != "camera_only" else []
        return camera, radar

    def update_data_association(self, camera_frame: CameraFrame, radar_frame: RadarFrame):
        camera, radar = self._detections(camera_frame, radar_frame)
        if not self.tracks_:
            return [], [], [], [], list(range(len(camera))), list(range(len(radar)))
        if camera:
            cam_pairs, unmatched_cam_tracks, unmatched_cam = hungarian_minimize(
                _camera_distance_matrix(self.tracks_, camera), self.config.camera_gating)
        else:
            cam_pairs, unmatched_cam_tracks, unmatched_cam = [], list(range(len(self.tracks_))), []
        if radar:
            rad_pairs, unmatched_rad_tracks, unmatched_rad = hungarian_minimize(
                _radar_distance_matrix(self.tracks_, radar), self.config.radar_gating)
        else:
            rad_pairs, unmatched_rad_tracks, unmatched_rad = [], list(range(len(self.tracks_))), []
        return cam_pairs, rad_pairs, unmatched_cam_tracks, unmatched_rad_tracks, unmatched_cam, unmatched_rad

    def update_tracks(self, camera_frame: CameraFrame, radar_frame: RadarFrame,
                      association_camera_pairs: Sequence[tuple[int, int]],
                      association_radar_pairs: Sequence[tuple[int, int]]) -> None:
        camera, radar = self._detections(camera_frame, radar_frame)
        updates = [[None, None] for _ in self.tracks_]
        for track_index, detection_index in association_camera_pairs:
            updates[track_index][0] = camera[detection_index]
        for track_index, detection_index in association_radar_pairs:
            updates[track_index][1] = radar[detection_index]
        for track, (camera_detection, radar_detection) in zip(self.tracks_, updates):
            track.update_synced(self.frame_timestamp_, camera_detection, radar_detection)

    def management_tracks(self, camera_frame: CameraFrame, radar_frame: RadarFrame,
                          unassociated_camera_track_indices: Sequence[int],
                          unassociated_camera_detection_indices: Sequence[int],
                          unassociated_radar_track_indices: Sequence[int],
                          unassociated_radar_detection_indices: Sequence[int]) -> None:
        camera, radar = self._detections(camera_frame, radar_frame)
        for index in unassociated_camera_detection_indices:
            self._create_track(camera[index])
        if self.sensor_mode == "radar_only":
            for index in unassociated_radar_detection_indices:
                self._create_track(radar[index])
        self.tracks_ = [track for track in self.tracks_ if not track.is_lost()]

    def _create_track(self, detection: CameraDetection | RadarDetection) -> None:
        aligned = replace(detection, timestamp=self.frame_timestamp_)
        self.tracks_.append(FusionTrack(aligned, self.predictor, self.track_config,
                                       track_id=self._next_track_id))
        self._next_track_id += 1

    def publish_tracks(self) -> list[FusionTrack]:
        snapshots = [copy.deepcopy(track) for track in self.tracks_ if track.is_confirmed()]
        if self.config.retain_history:
            self.fusion_tracks_frames_.append(
                FusionTrackFrame(self.frame_index_, self.frame_timestamp_, snapshots))
        return snapshots

    def time_alignment(self, camera_frames: Sequence[CameraFrame],
                       radar_frames: Sequence[RadarFrame]):
        """Exact timestamp pairs first, then greedy nearest timestamp pairing.

        The shorter stream drives matching, as in the source implementation.
        The gate is in seconds; unpaired frames are returned for missing-sensor
        processing instead of being discarded. Reserving exact pairs prevents
        paired and unpaired work items from sharing an update timestamp.
        """
        if any(not math.isfinite(frame.timestamp) for frame in [*camera_frames, *radar_frames]):
            raise ValueError("Frame timestamps must be finite")
        for stream in (camera_frames, radar_frames):
            if any(current.timestamp <= previous.timestamp for previous, current in zip(stream, stream[1:])):
                raise ValueError("Each sensor stream must have unique, strictly increasing timestamps")
        used_camera = set()
        used_radar = set()
        pairs = []
        radar_at_time = {frame.timestamp: index for index, frame in enumerate(radar_frames)}
        for index, frame in enumerate(camera_frames):
            other_index = radar_at_time.get(frame.timestamp)
            if other_index is not None:
                used_camera.add(index)
                used_radar.add(other_index)
                pairs.append((index, other_index))
        camera_driven = len(camera_frames) <= len(radar_frames)
        primary, secondary = ((camera_frames, radar_frames) if camera_driven else (radar_frames, camera_frames))
        for index, frame in enumerate(primary):
            if index in (used_camera if camera_driven else used_radar):
                continue
            choices = []
            for other_index, other in enumerate(secondary):
                used = used_radar if camera_driven else used_camera
                distance = abs(frame.timestamp - other.timestamp)
                if other_index not in used and distance < self.config.frame_association_gating:
                    choices.append((distance, other_index))
            if choices:
                _, other_index = min(choices)
                i, j = (index, other_index) if camera_driven else (other_index, index)
                used_camera.add(i)
                used_radar.add(j)
                pairs.append((i, j))
        return (pairs, [i for i in range(len(camera_frames)) if i not in used_camera],
                [j for j in range(len(radar_frames)) if j not in used_radar])
