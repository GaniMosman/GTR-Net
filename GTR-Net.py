"""GTR-Net: Gated Tail-Residual Network.

GTR-Net applies a learned, gated residual correction to a permanently frozen
Stage-1 forecast. The encoder operates independently at each history timestep;
temporal compression is performed only by the learned history-to-horizon layer.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

import torch
import torch.nn as nn


def extract_forecast(value: Any, horizon: int, target_index: int = -1) -> torch.Tensor:
    """Extract a ``[batch, horizon]`` forecast from common model outputs."""
    if torch.is_tensor(value):
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim == 2 and value.shape[1] == horizon:
            return value
        if value.ndim == 3 and value.shape[1] == horizon:
            return value[..., 0] if value.shape[-1] == 1 else value[..., target_index]
    if isinstance(value, Mapping):
        preferred = ("y_hat", "forecast", "prediction", "predictions", "pred", "output")
        candidates = [value[key] for key in preferred if key in value]
        candidates.extend(reversed(list(value.values())))
    elif isinstance(value, (tuple, list)):
        candidates = reversed(value)
    else:
        candidates = ()
    for candidate in candidates:
        try:
            return extract_forecast(candidate, horizon, target_index)
        except (TypeError, ValueError):
            continue
    raise ValueError(f"Could not extract a [B,{horizon}] Stage-1 forecast")


def freeze_stage1(stage1_forecaster: nn.Module) -> nn.Module:
    """Permanently freeze Stage 1, clear stale gradients, and set eval mode."""
    for parameter in stage1_forecaster.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    stage1_forecaster.eval()
    return stage1_forecaster


class PerTimestepFeatureEncoder(nn.Module):
    """Encode each history timestep independently: ``[B,T,F] -> [B,T,D]``."""

    def __init__(self, input_dim: int, model_dim: int, dropout: float):
        super().__init__()
        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)
        self.proj = nn.Linear(self.input_dim, self.model_dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(self.model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected x [B,T,{self.input_dim}], got {tuple(x.shape)}")
        return self.dropout(self.norm(self.act(self.proj(x))))


class HistoryToHorizonProjection(nn.Module):
    """Learn the temporal mapping from history length to forecast horizon."""

    def __init__(self, window_size: int, horizon: int):
        super().__init__()
        self.window_size = int(window_size)
        self.horizon = int(horizon)
        self.time_projection = nn.Linear(self.window_size, self.horizon)
        self.forward_calls = 0

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[1] != self.window_size:
            raise ValueError(
                f"Expected history [B,{self.window_size},D], got {tuple(history.shape)}"
            )
        self.forward_calls += 1
        return self.time_projection(history.transpose(1, 2)).transpose(1, 2)


class SharedRepresentation(nn.Module):
    """Refine the learned History-to-Horizon representation."""

    def __init__(self, model_dim: int, dropout: float):
        super().__init__()
        self.model_dim = int(model_dim)
        self.input_projection = nn.Linear(self.model_dim, self.model_dim)
        self.refine = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.norm = nn.LayerNorm(self.model_dim)

    def forward(self, projected: torch.Tensor) -> torch.Tensor:
        if projected.ndim != 3 or projected.shape[-1] != self.model_dim:
            raise ValueError(f"Expected projected [B,H,{self.model_dim}], got {tuple(projected.shape)}")
        base = self.input_projection(projected)
        return self.norm(base + self.refine(base))


class _SignedExpert(nn.Module):
    """Unrestricted signed residual expert with zero initial correction."""

    def __init__(self, model_dim: int, dropout: float):
        super().__init__()
        hidden_dim = max(int(model_dim) // 2, 16)
        self.hidden = nn.Sequential(
            nn.Linear(model_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.output_layer = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, shared: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.hidden(shared)).squeeze(-1)


class SpikeExpert(_SignedExpert):
    """Signed positive-tail residual expert."""


class CrashExpert(_SignedExpert):
    """Signed negative-tail residual expert."""


class Gate(nn.Module):
    """Three-way Normal/Spike/Crash softmax gate."""

    def __init__(self, model_dim: int, dropout: float):
        super().__init__()
        hidden_dim = max(int(model_dim) // 2, 16)
        self.net = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, shared: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.net(shared)
        return {"gate_logits": logits, "gate_probs": torch.softmax(logits, dim=-1)}


class GTRNet(nn.Module):
    """Frozen Stage 1 followed by the gated residual correction module."""

    def __init__(
        self,
        stage1_forecaster: nn.Module,
        input_dim: int,
        window_size: int,
        horizon: int = 288,
        model_dim: int = 128,
        dropout: float = 0.15,
        stage1_target_index: int = -1,
    ):
        super().__init__()
        self.stage1 = freeze_stage1(stage1_forecaster)
        self.horizon = int(horizon)
        self.model_dim = int(model_dim)
        self.stage1_target_index = int(stage1_target_index)

        self.encoder = PerTimestepFeatureEncoder(input_dim, model_dim, dropout)
        self.history_to_horizon = HistoryToHorizonProjection(window_size, horizon)
        self.shared_representation = SharedRepresentation(model_dim, dropout)
        self.spike_expert = SpikeExpert(model_dim, dropout)
        self.crash_expert = CrashExpert(model_dim, dropout)
        self.gate = Gate(model_dim, dropout)

    def train(self, mode: bool = True):
        super().train(mode)
        freeze_stage1(self.stage1)
        return self

    def forward(
        self, x: torch.Tensor, y_core: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        if y_core is None:
            with torch.no_grad():
                y_core = extract_forecast(
                    self.stage1(x), self.horizon, self.stage1_target_index
                )
        expected = (x.shape[0], self.horizon)
        if y_core.shape != expected:
            raise ValueError(f"Expected y_core {expected}, got {tuple(y_core.shape)}")
        y_core = y_core.detach()

        history = self.encoder(x)
        projected = self.history_to_horizon(history)
        shared = self.shared_representation(projected)
        spike_residual = self.spike_expert(shared)
        crash_residual = self.crash_expert(shared)
        gate_output = self.gate(shared)
        p_normal, p_spike, p_crash = gate_output["gate_probs"].unbind(dim=-1)
        applied_spike_residual = p_spike * spike_residual
        applied_crash_residual = p_crash * crash_residual
        correction = applied_spike_residual + applied_crash_residual
        y_hat = y_core + correction

        outputs = {
            "y_core": y_core,
            "history": history,
            "projected": projected,
            "shared_representation": shared,
            "spike_residual": spike_residual,
            "crash_residual": crash_residual,
            "gate_logits": gate_output["gate_logits"],
            "gate_probs": gate_output["gate_probs"],
            "p_normal": p_normal,
            "p_spike": p_spike,
            "p_crash": p_crash,
            "applied_spike_residual": applied_spike_residual,
            "applied_crash_residual": applied_crash_residual,
            "correction": correction,
            "y_hat": y_hat,
        }
        self._assert_outputs(x, outputs)
        return outputs

    def _assert_outputs(self, x: torch.Tensor, outputs: Mapping[str, torch.Tensor]) -> None:
        expected = (x.shape[0], self.horizon)
        for key in (
            "y_core", "spike_residual", "crash_residual", "p_normal",
            "p_spike", "p_crash", "applied_spike_residual",
            "applied_crash_residual", "correction", "y_hat",
        ):
            if outputs[key].shape != expected:
                raise AssertionError(f"{key}: expected {expected}, got {tuple(outputs[key].shape)}")
        if outputs["gate_logits"].shape != (*expected, 3):
            raise AssertionError("Gate logits must have shape [B,H,3]")
        if outputs["projected"].shape != (*expected, self.model_dim):
            raise AssertionError("History-to-Horizon output must have shape [B,H,D]")
        if outputs["shared_representation"].shape != (*expected, self.model_dim):
            raise AssertionError("Shared Representation output must have shape [B,H,D]")
        if outputs["y_core"].requires_grad:
            raise AssertionError("y_core must be detached")
        if not torch.allclose(
            outputs["gate_probs"].sum(-1), torch.ones_like(outputs["p_normal"]), atol=1e-6
        ):
            raise AssertionError("Gate probabilities must sum to one")
        expected_spike = outputs["p_spike"] * outputs["spike_residual"]
        expected_crash = outputs["p_crash"] * outputs["crash_residual"]
        if not torch.allclose(outputs["applied_spike_residual"], expected_spike, atol=1e-6, rtol=1e-5):
            raise AssertionError("Applied Spike correction identity failed")
        if not torch.allclose(outputs["applied_crash_residual"], expected_crash, atol=1e-6, rtol=1e-5):
            raise AssertionError("Applied Crash correction identity failed")
        expected_forecast = outputs["y_core"] + expected_spike + expected_crash
        if not torch.allclose(outputs["y_hat"], expected_forecast, atol=1e-6, rtol=1e-5):
            raise AssertionError("GTR-Net additive forecast identity failed")

    def assert_stage1_frozen(self, require_no_gradients: bool = False) -> None:
        if self.stage1.training:
            raise AssertionError("Stage 1 must remain in evaluation mode")
        if any(parameter.requires_grad for parameter in self.stage1.parameters()):
            raise AssertionError("A Stage-1 parameter has requires_grad=True")
        if require_no_gradients and any(
            parameter.grad is not None for parameter in self.stage1.parameters()
        ):
            raise AssertionError("Stage 1 received a gradient")

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        modules = (
            self.encoder,
            self.history_to_horizon,
            self.shared_representation,
            self.spike_expert,
            self.crash_expert,
            self.gate,
        )
        return (
            parameter
            for module in modules
            for parameter in module.parameters()
            if parameter.requires_grad
        )


__all__ = [
    "PerTimestepFeatureEncoder",
    "HistoryToHorizonProjection",
    "SharedRepresentation",
    "SpikeExpert",
    "CrashExpert",
    "Gate",
    "GTRNet",
    "extract_forecast",
    "freeze_stage1",
]
