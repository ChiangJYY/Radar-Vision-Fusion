"""Configurable baseline defaults; no learned statistics or selected checkpoints."""
import numpy as np
FIXED_Q = np.full(4, 0.1, dtype=np.float32)
FIXED_R_CAM = np.array([3., 3.], dtype=np.float32)
FIXED_R_RAD = np.array([3., 3., 1., 1.], dtype=np.float32)
