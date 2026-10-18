"""Differentiable constant-velocity filter and common-prior track fusion.

State order is (x, y, vx, vy). Camera observations contain position and radar
observations contain position and velocity. Both branches update the same
prediction, with their cross covariance retained in the fusion step.
"""

from __future__ import annotations

import torch

DEFAULT_Q = 0.025


def torch_update(
    x: torch.Tensor, P: torch.Tensor, z: torch.Tensor,
    r: torch.Tensor, H: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Joseph-form Kalman update; return posterior state, covariance and I-KH."""
    PH = P @ H.transpose(-1, -2)
    S = H @ PH + torch.diag_embed(r)
    eye = torch.eye(S.size(-1), device=S.device, dtype=S.dtype)
    L = torch.linalg.cholesky(S + 1e-6 * eye)
    K = torch.cholesky_solve(PH.transpose(-1, -2), L).transpose(-1, -2)
    state = x + K @ (z - H @ x)
    A = torch.eye(4, device=P.device, dtype=P.dtype) - K @ H
    cov = A @ P @ A.transpose(-1, -2) + K @ torch.diag_embed(r) @ K.transpose(-1, -2)
    return state, (cov + cov.transpose(-1, -2)) / 2, A


def _process_diagonal(q: float | torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
    """Accept fixed scalar or four diagonal entries, optionally per batch."""
    diagonal = torch.as_tensor(q, device=feature.device, dtype=feature.dtype)
    b = feature.shape[0]
    if diagonal.ndim == 0:
        diagonal = diagonal.expand(b, 4)
    elif diagonal.shape == (4,):
        diagonal = diagonal.expand(b, 4)
    elif diagonal.shape != (b, 4):
        raise ValueError("q must be a scalar, [4], or [batch, 4]")
    if not torch.isfinite(diagonal).all() or (diagonal < 0).any():
        raise ValueError("process variances must be finite and nonnegative")
    return diagonal


def torch_step(
    x: torch.Tensor, P: torch.Tensor, feature: torch.Tensor,
    r: torch.Tensor, q: float | torch.Tensor, dt: float | torch.Tensor,
    initial: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One prediction, two sensor updates and correlated track-fusion step.

    ``feature`` is [B,23], ``r`` is [B,6]. Validity entries 18/19 select
    camera/radar measurements. A single available sensor uses its posterior;
    when both are absent, the predicted state and covariance are retained.
    Q is added once per step, as in the original experiment (not scaled by dt).
    """
    if feature.ndim != 2 or feature.shape[1] != 23:
        raise ValueError("feature must have shape [batch, 23]")
    b = feature.shape[0]
    if x.shape != (b, 4, 1) or P.shape != (b, 4, 4) or r.shape != (b, 6):
        raise ValueError("expected x[B,4,1], P[B,4,4], r[B,6]")
    if not torch.isfinite(feature).all() or not torch.isfinite(r).all() or (r <= 0).any():
        raise ValueError("features must be finite and measurement variances strictly positive")
    I = torch.eye(4, device=feature.device, dtype=feature.dtype).expand(b, 4, 4)
    Hc = I[:, :2, :]
    delta = torch.as_tensor(dt, device=feature.device, dtype=feature.dtype)
    if delta.ndim == 0:
        delta = delta.expand(b)
    if delta.shape != (b,) or not torch.isfinite(delta).all() or (delta < 0).any():
        raise ValueError("dt must be a nonnegative finite scalar or [batch]")
    Ft = I.clone()
    Ft[:, 0, 2] = delta
    Ft[:, 1, 3] = delta
    if not initial:
        x = Ft @ x
        P = Ft @ P @ Ft.transpose(-1, -2) + torch.diag_embed(_process_diagonal(q, feature))
    mc = feature[:, 18].view(b, 1, 1) > 0.5
    mr = feature[:, 19].view(b, 1, 1) > 0.5
    xc, Pc, Ac = torch_update(x, P, feature[:, :2, None], r[:, :2], Hc)
    xr, Pr, Ar = torch_update(x, P, feature[:, 9:13, None], r[:, 2:], I)
    Pcr = Ac @ P @ Ar.transpose(-1, -2)
    omega = Pc + Pr - Pcr - Pcr.transpose(-1, -2) + 1e-5 * I
    gain = torch.linalg.solve(omega, (Pc - Pcr).transpose(-1, -2)).transpose(-1, -2)
    xf = xc + gain @ (xr - xc)
    Pf = Pc - gain @ (Pc - Pcr.transpose(-1, -2))
    Pf = (Pf + Pf.transpose(-1, -2)) / 2
    return (
        torch.where(mc & mr, xf, torch.where(mc, xc, torch.where(mr, xr, x))),
        torch.where(mc & mr, Pf, torch.where(mc, Pc, torch.where(mr, Pr, P))),
    )


def initial_state(feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize position from available sensors and velocity from radar."""
    if feature.ndim != 2 or feature.shape[-1] != 23 or not torch.isfinite(feature).all():
        raise ValueError("feature must be a finite [batch, 23] tensor")
    b = feature.shape[0]
    mc = (feature[:, 18:19] > 0.5).to(feature.dtype)
    mr = (feature[:, 19:20] > 0.5).to(feature.dtype)
    pos = (mc * feature[:, :2] + mr * feature[:, 9:11]) / (mc + mr).clamp(min=1)
    vel = mr * feature[:, 11:13]
    x = torch.cat([pos, vel], -1).unsqueeze(-1)
    p = feature.new_ones(b, 4)
    p[:, 2:] = mr * 2 + (1 - mr) * 100
    return x, torch.diag_embed(p)


def rollout(
    features: torch.Tensor, times: torch.Tensor, variances: torch.Tensor,
    q: float | torch.Tensor = DEFAULT_Q, detach: bool = False,
) -> torch.Tensor:
    """Filter a batch of windows; return positions [B,T,2].

    ``times`` has shape [B,T] in seconds and must be strictly increasing.
    ``detach=True`` truncates gradients through the previous filter state.
    """
    if features.ndim != 3 or features.shape[-1] != 23 or features.shape[1] < 1:
        raise ValueError("features must have shape [batch, time >= 1, 23]")
    if times.shape != features.shape[:2] or variances.shape != (*features.shape[:2], 6):
        raise ValueError("times must be [B,T] and variances [B,T,6]")
    if not torch.isfinite(times).all() or (times[:, 1:] <= times[:, :-1]).any():
        raise ValueError("timestamps must be finite and strictly increasing")
    x, P = initial_state(features[:, 0])
    outputs = []
    for t in range(features.size(1)):
        if detach and t:
            x, P = x.detach(), P.detach()
        dt = torch.zeros_like(times[:, 0]) if t == 0 else (times[:, t] - times[:, t - 1]).clamp(min=0.001)
        x, P = torch_step(x, P, features[:, t], variances[:, t], q, dt, initial=t == 0)
        outputs.append(x[:, :2, 0])
    return torch.stack(outputs, 1)
