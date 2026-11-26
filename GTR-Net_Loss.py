"""Regime-balanced ECDF objective and validation accumulator for GTR-Net."""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import torch
import torch.nn as nn


class TrainingPriceECDF(nn.Module):
    """Midpoint ECDF fitted exclusively to finite raw training prices."""

    def __init__(self, train_rrp):
        super().__init__()
        prices = torch.as_tensor(
            np.asarray(train_rrp, dtype=np.float32).reshape(-1), dtype=torch.float32
        )
        prices = prices[torch.isfinite(prices)].detach()
        if prices.numel() == 0:
            raise ValueError("train_rrp contains no finite observations")
        self.register_buffer("train_prices_sorted", torch.sort(prices).values)

    @torch.no_grad()
    def forward(self, raw_target_price: torch.Tensor) -> torch.Tensor:
        shape = raw_target_price.shape
        values = raw_target_price.detach().reshape(-1).contiguous().float()
        left = torch.searchsorted(self.train_prices_sorted, values, right=False)
        right = torch.searchsorted(self.train_prices_sorted, values, right=True)
        midpoint_rank = 0.5 * (left.float() + right.float())
        return (midpoint_rank / self.train_prices_sorted.numel()).reshape(shape)


class ECDFRegimeMembership(nn.Module):
    """Geometric Normal/Spike/Crash memberships from the training-price ECDF."""

    def __init__(self, train_rrp):
        super().__init__()
        self.training_ecdf = TrainingPriceECDF(train_rrp)

    @torch.no_grad()
    def forward(self, raw_target_price: torch.Tensor) -> Dict[str, torch.Tensor]:
        u = self.training_ecdf(raw_target_price)
        q_normal = 2.0 * torch.sqrt(torch.clamp(u * (1.0 - u), min=0.0))
        q_tail = 1.0 - q_normal
        q_spike = u * q_tail
        q_crash = (1.0 - u) * q_tail
        return {
            "u": u,
            "q_normal": q_normal,
            "q_tail": q_tail,
            "q_spike": q_spike,
            "q_crash": q_crash,
        }


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor, eps: float) -> torch.Tensor:
    return (weights * values).sum() / weights.sum().clamp_min(eps)


def _validate_inputs(
    outputs: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    raw_target_price: torch.Tensor,
) -> None:
    required = (
        "y_core", "y_hat", "spike_residual", "crash_residual",
        "p_normal", "p_spike", "p_crash",
    )
    missing = [key for key in required if key not in outputs]
    if missing:
        raise KeyError(f"Missing GTR-Net outputs: {missing}")
    if raw_target_price.shape != target.shape:
        raise ValueError("raw_target_price and target must have identical shapes")
    for key in required:
        if outputs[key].shape != target.shape:
            raise ValueError(f"{key} and target must have identical shapes")


class GTRNetLoss(nn.Module):
    """Parameter-free, regime-balanced Stage-2 objective."""

    def __init__(self, train_rrp, eps: float = 1e-8):
        super().__init__()
        self.membership = ECDFRegimeMembership(train_rrp)
        self.eps = float(eps)

    def memberships(self, raw_target_price: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.membership(raw_target_price)

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        target: torch.Tensor,
        raw_target_price: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        _validate_inputs(outputs, target, raw_target_price)
        membership = self.memberships(raw_target_price)
        q_normal = membership["q_normal"].to(target.dtype)
        q_spike = membership["q_spike"].to(target.dtype)
        q_crash = membership["q_crash"].to(target.dtype)

        final_error = torch.abs(outputs["y_hat"] - target)
        normal_forecast_loss = _weighted_mean(final_error, q_normal, self.eps)
        spike_forecast_loss = _weighted_mean(final_error, q_spike, self.eps)
        crash_forecast_loss = _weighted_mean(final_error, q_crash, self.eps)
        regime_balanced_forecast_loss = (
            normal_forecast_loss + spike_forecast_loss + crash_forecast_loss
        ) / 3.0

        required_residual = target - outputs["y_core"].detach()
        spike_expert_loss = _weighted_mean(
            torch.abs(outputs["spike_residual"] - required_residual), q_spike, self.eps
        )
        crash_expert_loss = _weighted_mean(
            torch.abs(outputs["crash_residual"] - required_residual), q_crash, self.eps
        )
        expert_loss = 0.5 * (spike_expert_loss + crash_expert_loss)

        normal_gate_loss = -(
            q_normal * torch.log(outputs["p_normal"] + self.eps)
        ).sum() / (q_normal.sum() + self.eps)
        spike_gate_loss = -(
            q_spike * torch.log(outputs["p_spike"] + self.eps)
        ).sum() / (q_spike.sum() + self.eps)
        crash_gate_loss = -(
            q_crash * torch.log(outputs["p_crash"] + self.eps)
        ).sum() / (q_crash.sum() + self.eps)
        gate_loss = (normal_gate_loss + spike_gate_loss + crash_gate_loss) / 3.0

        total_loss = regime_balanced_forecast_loss + expert_loss + gate_loss
        return {
            "loss": total_loss,
            "total_loss": total_loss,
            "regime_balanced_forecast_loss": regime_balanced_forecast_loss,
            "normal_forecast_loss": normal_forecast_loss,
            "spike_forecast_loss": spike_forecast_loss,
            "crash_forecast_loss": crash_forecast_loss,
            "expert_loss": expert_loss,
            "spike_expert_loss": spike_expert_loss,
            "crash_expert_loss": crash_expert_loss,
            "gate_loss": gate_loss,
            "normal_gate_loss": normal_gate_loss,
            "spike_gate_loss": spike_gate_loss,
            "crash_gate_loss": crash_gate_loss,
        }


class GTRNetValidation:
    """Globally aggregate balanced losses and raw-price selection metrics."""

    def __init__(self, criterion: GTRNetLoss):
        if not isinstance(criterion, GTRNetLoss):
            raise TypeError("criterion must be GTRNetLoss")
        self.criterion = criterion
        self.reset()

    def reset(self) -> None:
        self.normal_forecast_error_sum = 0.0
        self.spike_forecast_error_sum = 0.0
        self.crash_forecast_error_sum = 0.0
        self.spike_expert_error_sum = 0.0
        self.crash_expert_error_sum = 0.0
        self.normal_gate_sum = 0.0
        self.spike_gate_sum = 0.0
        self.crash_gate_sum = 0.0
        self.normal_weight_sum = 0.0
        self.spike_weight_sum = 0.0
        self.crash_weight_sum = 0.0
        self.raw_normal_error_sum = 0.0
        self.raw_spike_error_sum = 0.0
        self.raw_crash_error_sum = 0.0
        self.raw_absolute_error_sum = 0.0
        self.element_count = 0

    @torch.no_grad()
    def update(
        self,
        outputs: Mapping[str, torch.Tensor],
        target: torch.Tensor,
        raw_target_price: torch.Tensor,
        raw_prediction: torch.Tensor,
    ) -> None:
        _validate_inputs(outputs, target, raw_target_price)
        if raw_prediction.shape != raw_target_price.shape:
            raise ValueError("raw_prediction and raw_target_price must have identical shapes")

        membership = self.criterion.memberships(raw_target_price)
        q_normal = membership["q_normal"].to(target.dtype)
        q_spike = membership["q_spike"].to(target.dtype)
        q_crash = membership["q_crash"].to(target.dtype)
        final_error = torch.abs(outputs["y_hat"] - target)
        required_residual = target - outputs["y_core"].detach()

        self.normal_forecast_error_sum += float((q_normal * final_error).sum().item())
        self.spike_forecast_error_sum += float((q_spike * final_error).sum().item())
        self.crash_forecast_error_sum += float((q_crash * final_error).sum().item())
        self.spike_expert_error_sum += float((
            q_spike * torch.abs(outputs["spike_residual"] - required_residual)
        ).sum().item())
        self.crash_expert_error_sum += float((
            q_crash * torch.abs(outputs["crash_residual"] - required_residual)
        ).sum().item())
        self.normal_gate_sum += float((
            -q_normal * torch.log(outputs["p_normal"] + self.criterion.eps)
        ).sum().item())
        self.spike_gate_sum += float((
            -q_spike * torch.log(outputs["p_spike"] + self.criterion.eps)
        ).sum().item())
        self.crash_gate_sum += float((
            -q_crash * torch.log(outputs["p_crash"] + self.criterion.eps)
        ).sum().item())

        self.normal_weight_sum += float(q_normal.sum().item())
        self.spike_weight_sum += float(q_spike.sum().item())
        self.crash_weight_sum += float(q_crash.sum().item())
        raw_error = torch.abs(raw_prediction - raw_target_price)
        self.raw_normal_error_sum += float((q_normal * raw_error).sum().item())
        self.raw_spike_error_sum += float((q_spike * raw_error).sum().item())
        self.raw_crash_error_sum += float((q_crash * raw_error).sum().item())
        self.raw_absolute_error_sum += float(raw_error.sum().item())
        self.element_count += int(target.numel())

    def compute(self) -> Dict[str, float]:
        eps = self.criterion.eps
        normal_denominator = max(self.normal_weight_sum, eps)
        spike_denominator = max(self.spike_weight_sum, eps)
        crash_denominator = max(self.crash_weight_sum, eps)

        normal_forecast_loss = self.normal_forecast_error_sum / normal_denominator
        spike_forecast_loss = self.spike_forecast_error_sum / spike_denominator
        crash_forecast_loss = self.crash_forecast_error_sum / crash_denominator
        regime_balanced_forecast_loss = (
            normal_forecast_loss + spike_forecast_loss + crash_forecast_loss
        ) / 3.0
        spike_expert_loss = self.spike_expert_error_sum / spike_denominator
        crash_expert_loss = self.crash_expert_error_sum / crash_denominator
        expert_loss = 0.5 * (spike_expert_loss + crash_expert_loss)
        normal_gate_loss = self.normal_gate_sum / normal_denominator
        spike_gate_loss = self.spike_gate_sum / spike_denominator
        crash_gate_loss = self.crash_gate_sum / crash_denominator
        gate_loss = (normal_gate_loss + spike_gate_loss + crash_gate_loss) / 3.0
        total_loss = regime_balanced_forecast_loss + expert_loss + gate_loss

        return {
            "loss": total_loss,
            "total_loss": total_loss,
            "regime_balanced_forecast_loss": regime_balanced_forecast_loss,
            "normal_forecast_loss": normal_forecast_loss,
            "spike_forecast_loss": spike_forecast_loss,
            "crash_forecast_loss": crash_forecast_loss,
            "expert_loss": expert_loss,
            "spike_expert_loss": spike_expert_loss,
            "crash_expert_loss": crash_expert_loss,
            "gate_loss": gate_loss,
            "normal_gate_loss": normal_gate_loss,
            "spike_gate_loss": spike_gate_loss,
            "crash_gate_loss": crash_gate_loss,
            "soft_normal_mae": self.raw_normal_error_sum / normal_denominator,
            "soft_spike_mae": self.raw_spike_error_sum / spike_denominator,
            "soft_crash_mae": self.raw_crash_error_sum / crash_denominator,
            "overall_mae": self.raw_absolute_error_sum / max(self.element_count, 1),
            "n": self.element_count,
        }


__all__ = [
    "TrainingPriceECDF",
    "ECDFRegimeMembership",
    "GTRNetLoss",
    "GTRNetValidation",
]
