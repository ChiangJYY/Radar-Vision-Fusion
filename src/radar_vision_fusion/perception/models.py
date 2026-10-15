"""Adapters for the bundled third-party inference models. Checkpoints are explicit."""
from pathlib import Path
import sys
import numpy as np
import torch


def _add_source(directory):
    root = Path(directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


class RTDETRDetector:
    def __init__(self, source_directory, config, checkpoint, *, device="cpu", threshold=0.6):
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"RT-DETR checkpoint is required: {checkpoint}")
        root = _add_source(source_directory)
        existing = sys.modules.get("src")
        if existing is not None and Path(existing.__file__).resolve().parent != root / "src":
            raise RuntimeError("A different 'src' package is already loaded; use a fresh Python process")
        from src.core import YAMLConfig
        from torchvision.transforms import Compose, Resize, ToTensor
        cfg = YAMLConfig(str(config), PResNet={"pretrained": False})
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and "ema" in payload:
            state = payload["ema"]["module"]
        elif isinstance(payload, dict) and "model" in payload:
            state = payload["model"]
        else:
            state = payload
        cfg.model.load_state_dict(state, strict=True)
        self.model = cfg.model.deploy().to(device).eval()
        self.postprocessor = cfg.postprocessor.deploy().to(device).eval()
        self.transform = Compose([Resize((640, 640)), ToTensor()])
        if not 0 <= threshold <= 1:
            raise ValueError("Detection threshold must be in [0,1]")
        self.device, self.threshold = torch.device(device), threshold

    def __call__(self, pil_image):
        width, height = pil_image.size
        size = torch.tensor([[width, height]], device=self.device)
        with torch.inference_mode():
            prediction = self.model(self.transform(pil_image).unsqueeze(0).to(self.device))
            labels, boxes, scores = self.postprocessor(prediction, size)
        keep = scores[0] >= self.threshold
        return [{"label": int(label), "bbox": box, "score": float(score)}
                for label, box, score in zip(labels[0][keep].cpu().tolist(),
                                             boxes[0][keep].cpu().tolist(), scores[0][keep].cpu().tolist())]


class MetricDepthEstimator:
    def __init__(self, source_directory, checkpoint, *, encoder="vitb", max_depth=50., device="cpu", input_size=518):
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"Metric Depth Anything V2 checkpoint is required: {checkpoint}")
        root = _add_source(Path(source_directory) / "metric_depth")
        existing = sys.modules.get("depth_anything_v2")
        if existing is not None and not any(Path(p).resolve() == root / "depth_anything_v2" for p in existing.__path__):
            raise RuntimeError("Relative-depth implementation already loaded; use a fresh Python process for metric depth")
        from depth_anything_v2.dpt import DepthAnythingV2
        architectures = {
            "vits": {"features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]},
            "vitg": {"features": 384, "out_channels": [1536, 1536, 1536, 1536]},
        }
        if encoder not in architectures or max_depth <= 0 or input_size <= 0:
            raise ValueError("Invalid depth architecture, max_depth or input_size")
        self.model = DepthAnythingV2(encoder=encoder, max_depth=max_depth, **architectures[encoder])
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval()
        self.input_size = input_size

    def __call__(self, image_bgr):
        with torch.inference_mode():
            return np.asarray(self.model.infer_image(image_bgr, self.input_size), dtype=float)
