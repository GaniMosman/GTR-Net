# GTR-Net: Gated Tail-Residual Network

GTR-Net is a two-stage framework for multi-step electricity price forecasting
that targets both price spikes (upper tail) and price crashes (lower tail).

1. **Stage 1 (base forecaster).** Any multi-step forecaster produces a core
   forecast over the full horizon. It is trained first and then kept frozen.
2. **Stage 2 (tail-correction module).** A trainable module learns
   horizon-specific residual corrections to the frozen core forecast:
   - a **per-timestep encoder** maps each historical time step into a latent space;
   - a learned **History-to-Horizon Projection (HHP)** maps the encoded history
     to a separate representation for each forecast position;
   - a **shared horizon representation** refines these representations;
   - separate **signed Spike and Crash experts** predict residual corrections
     for the upper and lower tails;
   - a three-way **Normal/Spike/Crash soft Gate** weights the two experts at
     each forecast position.

The final forecast is

```
y_hat[h] = y_core[h] + p_spike[h] * r_spike[h] + p_crash[h] * r_crash[h]
```

where `y_core` is the frozen Stage-1 forecast, `r_spike` and `r_crash` are the
expert residuals, and `p_spike` and `p_crash` are Gate probabilities.

### Regime-balanced objective

Stage 2 is trained with a regime-balanced objective. Each observation receives
continuous Normal, Spike and Crash memberships from its position `u` in the
empirical distribution (mid-rank ECDF) of the training prices:

```
q_normal = 2 * sqrt(u * (1 - u))
q_spike  = u * (1 - q_normal)
q_crash  = (1 - u) * (1 - q_normal)
```

The objective combines three terms:

```
L = (L_normal + L_spike + L_crash) / 3     # regime-balanced forecast loss (membership-weighted MAE)
  + (L_expert_spike + L_expert_crash) / 2  # expert residual loss
  + (L_gate_normal + L_gate_spike + L_gate_crash) / 3  # membership-weighted Gate cross-entropy
```

Each regime contributes equally to the forecast loss regardless of how often it
occurs. No hard percentile thresholds are used during training.

## Repository contents

| File | Contents |
|---|---|
| `GTR-Net.py` | Stage-2 architecture: `PerTimestepFeatureEncoder`, `HistoryToHorizonProjection`, `SharedRepresentation`, `SpikeExpert`, `CrashExpert`, `Gate`, and the full `GTRNet` model, plus `freeze_stage1` and `extract_forecast` helpers |
| `GTR-Net_Loss.py` | `TrainingPriceECDF`, `ECDFRegimeMembership`, the regime-balanced `GTRNetLoss`, and `GTRNetValidation` (validation-set accumulator) |
| `Baseline_Models.py` | Stage-1 base forecasters: `MLPForecaster`, `LSTMForecaster`, `TCNForecaster`, `TransformerForecaster` |

The NBEATSx base forecaster used in the study is not included here; see the
original implementation by Olivares et al.
([cchallu/nbeatsx](https://github.com/cchallu/nbeatsx)).

## Requirements

- Python 3.9 or later
- PyTorch 2.0 or later
- NumPy

```bash
pip install -r requirements.txt
```

## Usage

The file names contain hyphens, so load them with `importlib`:

```python
import importlib.util
import sys

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

gtr = load_module("gtr_net", "GTR-Net.py")
gtr_loss = load_module("gtr_net_loss", "GTR-Net_Loss.py")
baselines = load_module("baseline_models", "Baseline_Models.py")
```

### Shapes and conventions

| Quantity | Shape | Notes |
|---|---|---|
| `x` | `[batch, window_size, input_dim]` | input history (scaled features) |
| `y_core` | `[batch, horizon]` | Stage-1 forecast, in the same target space as `target` |
| `target` | `[batch, horizon]` | target in the model's (scaled) target space |
| `raw_target_price` | `[batch, horizon]` | the same target in original price units, used only for the ECDF memberships |
| `train_rrp` | 1-D array | raw training-period prices used to fit the ECDF |

### Example

```python
import numpy as np
import torch

window_size, horizon, input_dim = 576, 288, 9

# Stage 1: train a base forecaster first (training loop not shown), then freeze it inside GTR-Net.
stage1 = baselines.LSTMForecaster(input_size=input_dim, horizon=horizon)

# Stage 2: GTR-Net freezes the Stage-1 model automatically.
model = gtr.GTRNet(
    stage1_forecaster=stage1,
    input_dim=input_dim,
    window_size=window_size,
    horizon=horizon,
    model_dim=128,
    dropout=0.15,
)

train_rrp = np.random.standard_t(df=3, size=100_000) * 50 + 80  # replace with real training prices
criterion = gtr_loss.GTRNetLoss(train_rrp)
optimiser = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)

x = torch.randn(32, window_size, input_dim)
target = torch.randn(32, horizon)
raw_target_price = target * 50 + 80  # inverse-scaled target, used for the memberships

outputs = model(x)  # or model(x, y_core=cached_stage1_forecast)
losses = criterion(outputs, target, raw_target_price)

optimiser.zero_grad()
losses["loss"].backward()
optimiser.step()

forecast = outputs["y_hat"]  # [batch, horizon]
```

`model(x)` returns a dictionary that also includes the Stage-1 forecast
(`y_core`), the expert residuals (`spike_residual`, `crash_residual`), the Gate
probabilities (`p_normal`, `p_spike`, `p_crash`) and the applied correction
(`correction`). These outputs support analysis of the correction mechanism.

### Notes

- The Stage-1 model is never updated: `GTRNet` sets its parameters to
  `requires_grad=False` and keeps it in evaluation mode, even when `model.train()`
  is called. Pass only `model.trainable_parameters()` to the optimiser.
- Stage-1 forecasts can be pre-computed and passed as `y_core` to avoid
  recomputing them at every step.
- The expert output layers are initialised to zero, so GTR-Net starts from the
  Stage-1 forecast.
- `GTRNetValidation` accumulates the same loss terms over a full validation set,
  plus Normal, Spike and Crash membership-weighted MAE in original price units,
  for model selection.

## Citation

A citation will be added when the associated paper is published.

## License

This project is released under the [MIT License](LICENSE).
