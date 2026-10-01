"""Revision-only nine-model forecasts under forecast protocol V1.

Forecast functions receive only past panel rows and prebuilt future predictors.
They never receive held-out target values.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = json.loads((ROOT / "revision/forecast_protocol/protocol_v1.json").read_text(encoding="utf-8"))
SCALE = 1_000_000.0
EXOG = ("y_lag12", "month_days", "weekday_holidays_asof")


def read_panel(path: Path | None = None) -> pd.DataFrame:
    path = path or ROOT / PROTOCOL["panel"]
    out = pd.read_csv(path, encoding="utf-8-sig")
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    assert out["date"].is_unique and len(out) == 60
    return out


def read_future(year: int, rolling: bool = False) -> pd.DataFrame:
    suffix = "rolling_h1" if rolling else "fixed_h1_12"
    path = ROOT / f"revision/forecast_protocol/origin_inputs_{year}_{suffix}.csv"
    out = pd.read_csv(path, encoding="utf-8-sig")
    assert len(out) == 12 and out["target_month"].is_unique
    assert all(col in out and out[col].notna().all() for col in EXOG)
    return out


def train_frame(panel: pd.DataFrame, end_year: int) -> pd.DataFrame:
    end = f"{end_year}-12-01"
    out = panel.loc[(panel.date >= "2022-01-01") & (panel.date <= end)].copy()
    out["weekday_holidays_asof"] = out["weekday_holidays"]
    assert len(out) == 12 * (end_year - 2021)
    return out


def validate_forecast(pred: np.ndarray, future: pd.DataFrame) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    if len(pred) != len(future) or not np.isfinite(pred).all():
        raise ValueError("forecast length or finiteness failure")
    return pred


def scores(truth: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    truth, pred = np.asarray(truth, dtype=float), np.asarray(pred, dtype=float)
    assert len(truth) == len(pred) == 12 and np.all(truth > 0) and np.isfinite(pred).all()
    error = truth - pred
    return {"MAPE": float(np.mean(np.abs(error) / truth) * 100),
            "MAE": float(np.mean(np.abs(error))),
            "RMSE": float(np.sqrt(np.mean(error ** 2)))}


def candidates(model: str, selected_base: dict | None = None) -> list[dict]:
    specs = PROTOCOL["models"]
    if model in ("M1_seasonal_naive", "M2_holt_winters"):
        return [{}]
    if model in ("M3_sarima", "M4_sarimax_3"):
        f = PROTOCOL["statistical_fit"]
        return [{"order": o, "seasonal_order": s}
                for o, s in itertools.product(f["regular_orders"], f["seasonal_orders"])]
    if model == "M5_prophet":
        return [{"seasonality_mode": mode} for mode in specs[model]["seasonality_modes"]]
    if model == "M6_xgboost":
        spec = specs[model]
        return [{"max_depth": depth, "n_estimators": n}
                for depth, n in itertools.product(spec["depth"], spec["estimators"])]
    if model == "M7_lstm":
        spec = specs[model]
        return [{"lookback": l, "hidden": h, "epochs": e}
                for l, h, e in itertools.product(spec["past_target_lookback"], spec["hidden_units"], spec["epoch_candidates"])]
    if model == "M8_transformer":
        spec = specs[model]
        return [{"lookback": l, "d_model": d, "epochs": e}
                for l, d, e in itertools.product(spec["past_target_lookback"], spec["d_model"], spec["epoch_candidates"])]
    if model == "M9_sarimax_lstm_residual":
        assert selected_base is not None
        spec = specs[model]
        return [{"lookback": l, "epochs": e, "base": selected_base}
                for l, e in itertools.product(spec["residual_lookback"], spec["epoch_candidates"])]
    raise KeyError(model)


def complexity(model: str, candidate: dict) -> int:
    if model in ("M3_sarima", "M4_sarimax_3"):
        return sum(candidate["order"]) + sum(candidate["seasonal_order"][:3]) + (3 if model == "M4_sarimax_3" else 0)
    if model == "M5_prophet":
        return 0
    if model == "M6_xgboost":
        return candidate["n_estimators"] * (2 ** candidate["max_depth"] - 1)
    if model in ("M7_lstm", "M8_transformer"):
        return candidate.get("hidden", candidate.get("d_model", 0))
    if model == "M9_sarimax_lstm_residual":
        return candidate["lookback"]
    return 0


def _stat_fit(train: pd.DataFrame, candidate: dict, use_exog: bool):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = train["hsr_pax_total"].to_numpy(dtype=float) / SCALE
    x = None
    if use_exog:
        x = train.loc[:, EXOG].to_numpy(dtype=float)
        x[:, 0] /= SCALE
        if np.any(np.std(x, axis=0) < 1e-12) or np.linalg.matrix_rank(x) < x.shape[1]:
            raise ValueError("unidentified training regressors")
    cfg = PROTOCOL["statistical_fit"]
    fitted = SARIMAX(y, exog=x, order=tuple(candidate["order"]),
                     seasonal_order=tuple(candidate["seasonal_order"]),
                     trend=cfg["trend"], enforce_stationarity=cfg["enforce_stationarity"],
                     enforce_invertibility=cfg["enforce_invertibility"])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fitted.fit(disp=False, maxiter=cfg["maxiter"])
    if not bool(result.mle_retvals.get("converged", False)):
        raise RuntimeError("SARIMA/SARIMAX optimiser did not converge")
    return result


def _stat_forecast(result, future: pd.DataFrame, use_exog: bool) -> np.ndarray:
    x = None
    if use_exog:
        x = future.loc[:, EXOG].to_numpy(dtype=float)
        x[:, 0] /= SCALE
    return np.asarray(result.forecast(len(future), exog=x), dtype=float) * SCALE


def _xgb_forecast(train: pd.DataFrame, future: pd.DataFrame,
                  panel: pd.DataFrame, candidate: dict, seed: int) -> np.ndarray:
    import xgboost as xgb

    spec = PROTOCOL["models"]["M6_xgboost"]
    dates = panel["date"].tolist()
    targets = (panel["hsr_pax_total"].to_numpy(dtype=float) / SCALE).tolist()
    final_train_date = train["date"].iloc[-1]
    n_history = dates.index(final_train_date) + 1
    target_history = targets[:n_history]
    lags = spec["target_lags"]

    def row_features(y_before: list[float], x: np.ndarray) -> list[float]:
        return ([y_before[-lag] for lag in lags]
                + [float(np.mean(y_before[-width:])) for width in spec["rolling_means"]]
                + x.tolist())

    features, labels = [], []
    first = dates.index("2022-01-01")
    for i in range(first, n_history):
        x = panel.loc[i, ["y_lag12", "month_days", "weekday_holidays"]].to_numpy(dtype=float).copy()
        x[0] /= SCALE
        features.append(row_features(targets[:i], x))
        labels.append(targets[i])
    assert len(labels) == len(train)
    model = xgb.XGBRegressor(
        n_estimators=candidate["n_estimators"], max_depth=candidate["max_depth"],
        learning_rate=spec["learning_rate"], subsample=spec["subsample"],
        colsample_bytree=spec["colsample_bytree"], objective=spec["objective"],
        random_state=seed, n_jobs=1, tree_method="hist")
    model.fit(np.asarray(features, dtype=float), np.asarray(labels, dtype=float))
    path = []
    for _, item in future.iterrows():
        x = np.array([item["y_lag12"] / SCALE, item["month_days"], item["weekday_holidays_asof"]], dtype=float)
        pred = float(model.predict(np.asarray([row_features(target_history, x)]))[0])
        path.append(pred * SCALE)
        target_history.append(pred)  # recursive: never append future truth
    return np.asarray(path)


def _prophet_forecast(train: pd.DataFrame, future: pd.DataFrame, candidate: dict) -> np.ndarray:
    import logging
    from prophet import Prophet

    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    train_data = pd.DataFrame({"ds": pd.to_datetime(train["date"]),
                               "y": train["hsr_pax_total"].to_numpy(dtype=float) / SCALE})
    future_data = pd.DataFrame({"ds": pd.to_datetime(future["target_month"])})
    for key in EXOG:
        train_data[key] = train[key].to_numpy(dtype=float) / (SCALE if key == "y_lag12" else 1)
        future_data[key] = future[key].to_numpy(dtype=float) / (SCALE if key == "y_lag12" else 1)
    spec = PROTOCOL["models"]["M5_prophet"]
    model = Prophet(yearly_seasonality=spec["yearly_seasonality"],
                    weekly_seasonality=spec["weekly_seasonality"],
                    daily_seasonality=spec["daily_seasonality"],
                    changepoint_prior_scale=spec["changepoint_prior_scale"],
                    seasonality_mode=candidate["seasonality_mode"])
    for key in EXOG:
        model.add_regressor(key)
    model.fit(train_data)
    return model.predict(future_data)["yhat"].to_numpy(dtype=float) * SCALE


def _nn_forecast(model_name: str, train: pd.DataFrame, future: pd.DataFrame,
                 panel: pd.DataFrame, candidate: dict, seed: int) -> np.ndarray:
    import torch
    from torch import nn

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    lookback = candidate["lookback"]
    spec = PROTOCOL["models"][model_name]
    dates = panel["date"].tolist()
    all_y = panel["hsr_pax_total"].to_numpy(dtype=float) / SCALE
    end_idx = dates.index(train["date"].iloc[-1]) + 1
    first = dates.index("2022-01-01")
    y_fit = all_y[first:end_idx]
    y_mean, y_std = float(np.mean(y_fit)), float(np.std(y_fit))
    assert y_std > 0
    x_fit = train.loc[:, EXOG].to_numpy(dtype=float).copy()
    x_fit[:, 0] /= SCALE
    x_mean, x_std = x_fit.mean(axis=0), x_fit.std(axis=0)
    x_std[x_std == 0] = 1
    x_scaled = (x_fit - x_mean) / x_std
    past = np.asarray([(all_y[i - lookback:i] - y_mean) / y_std
                       for i in range(first, end_idx)], dtype=np.float32)
    assert len(past) == len(train) and past.shape[1] == lookback
    label = np.asarray((y_fit - y_mean) / y_std, dtype=np.float32)

    if model_name == "M7_lstm":
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.LSTM(1, candidate["hidden"], batch_first=True)
                self.head = nn.Linear(candidate["hidden"] + 3, 1)

            def forward(self, history, current):
                encoded, _ = self.encoder(history)
                return self.head(torch.cat([encoded[:, -1, :], current], dim=1)).squeeze(1)
    else:
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                d = candidate["d_model"]
                self.proj = nn.Linear(1, d)
                self.pos = nn.Parameter(torch.randn(1, lookback, d) * 0.02)
                layer = nn.TransformerEncoderLayer(d_model=d, nhead=spec["heads"],
                                                   dim_feedforward=2 * d,
                                                   dropout=spec["dropout"], batch_first=True)
                self.encoder = nn.TransformerEncoder(layer, num_layers=spec["layers"])
                self.head = nn.Linear(d + 3, 1)

            def forward(self, history, current):
                encoded = self.encoder(self.proj(history) + self.pos)
                return self.head(torch.cat([encoded[:, -1, :], current], dim=1)).squeeze(1)

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=spec["learning_rate"],
                           weight_decay=spec["weight_decay"])
    history_tensor = torch.from_numpy(past).unsqueeze(-1)
    x_tensor = torch.from_numpy(np.asarray(x_scaled, dtype=np.float32))
    label_tensor = torch.from_numpy(label)
    for _ in range(candidate["epochs"]):
        net.train()
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(history_tensor, x_tensor), label_tensor)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite neural training loss")
        loss.backward()
        opt.step()
    net.eval()
    y_history = all_y[:end_idx].tolist()
    predictions = []
    for _, item in future.iterrows():
        x = np.array([item["y_lag12"] / SCALE, item["month_days"], item["weekday_holidays_asof"]])
        x = ((x - x_mean) / x_std).astype(np.float32)
        hist = np.asarray([(v - y_mean) / y_std for v in y_history[-lookback:]], dtype=np.float32)
        with torch.no_grad():
            scaled = float(net(torch.from_numpy(hist).reshape(1, lookback, 1),
                               torch.from_numpy(x).reshape(1, 3)).item())
        value = scaled * y_std + y_mean
        predictions.append(value * SCALE)
        y_history.append(value)
    return np.asarray(predictions)


def _hybrid_forecast(train: pd.DataFrame, future: pd.DataFrame,
                     candidate: dict, seed: int) -> np.ndarray:
    import torch
    from torch import nn

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    base = _stat_fit(train, candidate["base"], True)
    base_prediction = _stat_forecast(base, future, True)
    residuals = train["hsr_pax_total"].to_numpy(dtype=float) / SCALE - np.asarray(base.fittedvalues)
    lookback = candidate["lookback"]
    assert len(residuals) > lookback
    mean, std = float(residuals.mean()), float(residuals.std())
    if std <= 1e-12:
        raise ValueError("zero residual variance")
    scaled = ((residuals - mean) / std).astype(np.float32)
    windows = np.asarray([scaled[i - lookback:i] for i in range(lookback, len(scaled))], dtype=np.float32)
    labels = scaled[lookback:]

    class ResidualNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.LSTM(1, 16, batch_first=True)
            self.head = nn.Linear(16, 1)

        def forward(self, x):
            encoded, _ = self.encoder(x)
            return self.head(encoded[:, -1, :]).squeeze(1)

    net = ResidualNet()
    spec = PROTOCOL["models"]["M9_sarimax_lstm_residual"]
    opt = torch.optim.Adam(net.parameters(), lr=spec["learning_rate"],
                           weight_decay=spec["weight_decay"])
    x_t = torch.from_numpy(windows).unsqueeze(-1)
    y_t = torch.from_numpy(labels)
    for _ in range(candidate["epochs"]):
        net.train()
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(x_t), y_t)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite residual training loss")
        loss.backward()
        opt.step()
    net.eval()
    history = scaled.tolist()
    correction = []
    for _ in range(len(future)):
        x = torch.tensor(history[-lookback:], dtype=torch.float32).reshape(1, lookback, 1)
        with torch.no_grad():
            guess = float(net(x).item())
        correction.append((guess * std + mean) * SCALE)
        history.append(guess)
    return base_prediction + np.asarray(correction)


def forecast(model: str, candidate: dict, panel: pd.DataFrame,
             train: pd.DataFrame, future: pd.DataFrame, seed: int = 42) -> np.ndarray:
    assert train["date"].iloc[-1] < future["target_month"].iloc[0]
    if model == "M1_seasonal_naive":
        pred = future["y_lag12"].to_numpy(dtype=float)
    elif model == "M2_holt_winters":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        y = train["hsr_pax_total"].to_numpy(dtype=float) / SCALE
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = ExponentialSmoothing(y, trend="add", seasonal="add",
                                          seasonal_periods=12, damped_trend=False).fit(optimized=True)
        pred = np.asarray(fitted.forecast(len(future))) * SCALE
    elif model in ("M3_sarima", "M4_sarimax_3"):
        use_exog = model == "M4_sarimax_3"
        pred = _stat_forecast(_stat_fit(train, candidate, use_exog), future, use_exog)
    elif model == "M5_prophet":
        pred = _prophet_forecast(train, future, candidate)
    elif model == "M6_xgboost":
        pred = _xgb_forecast(train, future, panel, candidate, seed)
    elif model in ("M7_lstm", "M8_transformer"):
        pred = _nn_forecast(model, train, future, panel, candidate, seed)
    elif model == "M9_sarimax_lstm_residual":
        pred = _hybrid_forecast(train, future, candidate, seed)
    else:
        raise KeyError(model)
    return validate_forecast(pred, future)
