"""Univariate time-series forecasting with a multilayer perceptron.

Requires numpy and scikit-learn: pip install numpy scikit-learn
Run a synthetic example: python mlp_forecasting.py
Use a numeric CSV column: python mlp_forecasting.py --csv data.csv --column value
CSV input must have a header and rows ordered from oldest to newest.
"""

import argparse
import csv

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_windows(series, lookback):
    """Return lag windows and their next-step targets without shuffling."""
    series = np.asarray(series, dtype=float)
    if series.ndim != 1 or not np.isfinite(series).all():
        raise ValueError("Series must be one-dimensional and contain finite values.")
    if lookback < 1 or len(series) <= lookback:
        raise ValueError("Lookback must be positive and smaller than the series length.")
    windows = np.lib.stride_tricks.sliding_window_view(series, lookback + 1)
    return windows[:, :-1].copy(), windows[:, -1].copy()


def build_model(seed=42):
    """Scale inputs and targets using training data only."""
    return TransformedTargetRegressor(
        regressor=make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                max_iter=1000,
                random_state=seed,
                shuffle=False,
                early_stopping=False,
            ),
        ),
        transformer=StandardScaler(),
    )


def forecast(model, history, lookback, horizon):
    """Recursively predict future values, feeding each prediction back as a lag."""
    if horizon < 1:
        raise ValueError("Horizon must be positive.")
    history = np.asarray(history, dtype=float)
    if history.ndim != 1 or not np.isfinite(history).all():
        raise ValueError("History must be one-dimensional and contain finite values.")
    values = list(history)
    if lookback < 1 or len(values) < lookback:
        raise ValueError("History must contain at least lookback values.")
    predictions = []
    for _ in range(horizon):
        prediction = float(model.predict(np.asarray(values[-lookback:]).reshape(1, -1))[0])
        predictions.append(prediction)
        values.append(prediction)
    return np.asarray(predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="CSV file containing an ordered time series")
    parser.add_argument("--column", default="value", help="Numeric CSV column (default: value)")
    parser.add_argument("--lookback", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if not 0 < args.test_fraction < 1 or args.horizon < 1:
        parser.error("test-fraction must be between 0 and 1; horizon must be positive")

    if args.csv:
        with open(args.csv, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if args.column not in (reader.fieldnames or []):
                parser.error(f"CSV is missing column {args.column!r}")
            series = np.asarray([float(row[args.column]) for row in reader])
    else:
        rng = np.random.default_rng(42)
        time = np.arange(600)
        series = 0.01 * time + np.sin(2 * np.pi * time / 24) + rng.normal(0, 0.1, len(time))

    X, y = make_windows(series, args.lookback)
    split = int(len(y) * (1 - args.test_fraction))
    if split < 2 or split >= len(y):
        parser.error("Provide enough observations for at least two training windows and one test window")

    model = build_model()
    model.fit(X[:split], y[:split])
    # Rolling one-step evaluation: each test window uses observed past values.
    predicted = model.predict(X[split:])
    print(f"One-step test MAE: {mean_absolute_error(y[split:], predicted):.4f}")
    print(f"One-step test RMSE: {np.sqrt(mean_squared_error(y[split:], predicted)):.4f}")
    print(f"Persistence baseline MAE: {mean_absolute_error(y[split:], X[split:, -1]):.4f}")
    # Refit on all observations before forecasting beyond the end of the series.
    model.fit(X, y)
    future = forecast(model, series, args.lookback, args.horizon)
    print("Future forecast:", np.array2string(future, precision=4))


if __name__ == "__main__":
    main()
