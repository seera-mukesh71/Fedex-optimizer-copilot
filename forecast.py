"""
forecast.py  -  XGBoost + Temporal Fusion Transformer (TFT) weighted-blend demand forecasting.

    python forecast.py train                 # weekly: backtest, choose blend weights, refit on all data, save models
    python forecast.py forecast              # nightly: load saved models, write data/forecasts.csv (next 7 days)
    python forecast.py train --no-tft        # XGBoost only (use this if PyTorch is not installed)
    python forecast.py train --tft-epochs 15 # more TFT training

What it forecasts: for each of 18 series (6 routes x 3 categories), the next 7 days of
    weight_kg, volume_m3, item_count   as P10 / P50 / P90.

Design
  * XGBoost: one model per (series, target); quantile objective (P10/P50/P90 in one model);
    every lag is >= 7 days, so ONE model covers horizons 1..7 directly (no recursive forecasting).
  * TFT: ONE global model trained on all 18 series (weight_kg + volume_m3 jointly).
  * Blend: per (series, target) weight w in {0,0.1,...,1} on TFT, chosen by pinball loss on the last 8 weeks.
    If TFT does not help on a series, w ends up 0 - the backtest file shows that honestly.
  * item_count is forecast by XGBoost only (the optimizer does not use it).
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))   # always run relative to the project folder
import argparse, json, os, pickle, sys, time, warnings
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

DATA, MODELS = "data/demand_data.csv", "models"
OUT_F, OUT_M = "data/forecasts.csv", "data/backtest_metrics.csv"
QS = [0.1, 0.5, 0.9]
TARGETS = ["weight_kg", "volume_m3", "item_count"]
TFT_TARGETS = ["weight_kg", "volume_m3"]
HORIZON, HOLDOUT, ENC_LEN = 7, 56, 28
KEYS = ["route_id", "category"]
XGB_PARAMS = dict(n_estimators=250, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=5, objective="reg:quantileerror", quantile_alpha=np.array(QS),
                  tree_method="hist", n_jobs=2, random_state=0)


# ------------------------------------------------------------------ data + features
def load():
    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values(KEYS + ["date"]).reset_index(drop=True)
    last_hist = df.loc[df.weight_kg.notna(), "date"].max()
    return df, last_hist


def add_features(df):
    """Every feature is knowable at the forecast origin: lags >= 7 days, or calendar / plan / price columns."""
    df = df.copy()
    g = df.groupby(KEYS, sort=False)
    for t in TARGETS:
        s7 = g[t].shift(7)
        df[f"{t}_scale"] = s7.groupby([df.route_id, df.category]).transform(lambda s: s.rolling(28, min_periods=14).mean())
        sc = df[f"{t}_scale"]
        for L in (7, 14, 21, 28):
            df[f"{t}_lag{L}r"] = g[t].shift(L) / sc
        df[f"{t}_rm7r"] = s7.groupby([df.route_id, df.category]).transform(lambda s: s.rolling(7, min_periods=4).mean()) / sc
    df["signal_ratio"] = df.client_forecast_signal_kg / df["weight_kg_scale"]
    df["price_ratio"] = df.our_price_per_kg / df.competitor_price_per_kg
    df["fuel_lag7"] = g["fuel_price_index"].shift(7)
    df["weather_lag7"] = g["weather_severity"].shift(7)
    df["pre_holiday"] = np.maximum(g["is_holiday"].shift(-1).fillna(0), g["is_holiday"].shift(-2).fillna(0))
    doy = df.date.dt.dayofyear
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365)
    return df


COMMON = ["day_of_week", "month", "doy_sin", "doy_cos", "is_holiday", "pre_holiday", "is_peak_season", "promo_flag",
          "signal_ratio", "price_ratio", "fuel_lag7", "weather_lag7"]


def feat_cols(t):
    return [f"{t}_lag{L}r" for L in (7, 14, 21, 28)] + [f"{t}_rm7r"] + COMMON


# ------------------------------------------------------------------ XGBoost
def xgb_fit(d, t):
    d = d.dropna(subset=feat_cols(t) + [t, f"{t}_scale"])
    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(d[feat_cols(t)], d[t] / d[f"{t}_scale"])
    return m


def xgb_predict(m, d, t):
    """returns (n,3) quantiles in original units, sorted"""
    p = np.asarray(m.predict(d[feat_cols(t)])).reshape(len(d), -1) * d[f"{t}_scale"].values[:, None]
    return np.sort(np.clip(p, 0, None), axis=1)


# ------------------------------------------------------------------ TFT (optional, lazy imports)
def tft_available():
    try:
        import torch, lightning, pytorch_forecasting  # noqa: F401
        return True
    except Exception:
        return False


def tft_frame(df, last_hist):
    """TFT-ready frame: integer time index, string categoricals, dimensionless extras, placeholders for future targets."""
    d = df.copy()
    d["series"] = d.route_id + "_" + d.category
    d["time_idx"] = (d.date - d.date.min()).dt.days
    d["price_ratio"] = d.our_price_per_kg / d.competitor_price_per_kg
    hist_mean = d[d.date <= last_hist].groupby("series").weight_kg.mean()
    d["signal_rel"] = d.client_forecast_signal_kg / d.series.map(hist_mean)
    for c in ["fuel_price_index", "weather_severity"] + TFT_TARGETS:
        d[c] = d.groupby("series")[c].ffill()                # future placeholders only; the decoder never reads them
    for c in ["day_of_week", "month", "is_holiday", "is_peak_season", "promo_flag"]:
        d[c] = d[c].astype(str)
    return d


def tft_dataset(frame):
    from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer
    from pytorch_forecasting.data import MultiNormalizer
    return TimeSeriesDataSet(
        frame, time_idx="time_idx", target=TFT_TARGETS, group_ids=["series"],
        max_encoder_length=ENC_LEN, max_prediction_length=HORIZON,
        static_categoricals=["series", "route_id", "category"],
        time_varying_known_categoricals=["day_of_week", "month", "is_holiday", "is_peak_season", "promo_flag"],
        time_varying_known_reals=["signal_rel", "price_ratio"],
        time_varying_unknown_reals=TFT_TARGETS + ["fuel_price_index", "weather_severity"],
        target_normalizer=MultiNormalizer([GroupNormalizer(groups=["series"], transformation="softplus") for _ in TFT_TARGETS]),
        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True)


def tft_fit(frame_train, epochs):
    import torch
    from lightning.pytorch import Trainer
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss, MultiLoss
    torch.manual_seed(0)
    ds = tft_dataset(frame_train)
    dl = ds.to_dataloader(train=True, batch_size=128, num_workers=0)
    model = TemporalFusionTransformer.from_dataset(
        ds, learning_rate=0.01, hidden_size=16, attention_head_size=2, dropout=0.1, hidden_continuous_size=8,
        loss=MultiLoss([QuantileLoss(QS) for _ in TFT_TARGETS]), reduce_on_plateau_patience=2)
    tr = Trainer(max_epochs=epochs, accelerator="cpu", gradient_clip_val=0.1, enable_progress_bar=False,
                 enable_checkpointing=False, logger=False, enable_model_summary=False)
    tr.fit(model, dl)
    return model, ds, tr


def tft_predict(model, ds, frame, **kw):
    """-> DataFrame[series, time_idx(first decoder step), h, target, q10, q50, q90] in original units"""
    dsp = type(ds).from_dataset(ds, frame, stop_randomization=True, **kw)
    dl = dsp.to_dataloader(train=False, batch_size=256, num_workers=0)
    res = model.predict(dl, mode="quantiles", return_index=True)
    out, idx = (res.output, res.index) if hasattr(res, "output") else (res[0], res[1])
    rows = []
    for ti, tname in enumerate(TFT_TARGETS):
        arr = out[ti].detach().cpu().numpy() if isinstance(out, (list, tuple)) else out[..., ti, :].detach().cpu().numpy()
        n, H, _ = arr.shape
        for h in range(H):
            rows.append(pd.DataFrame({"series": idx["series"].values, "start": idx["time_idx"].values, "h": h,
                                      "target": tname, "q10": arr[:, h, 0], "q50": arr[:, h, 1], "q90": arr[:, h, 2]}))
    r = pd.concat(rows, ignore_index=True)
    r["time_idx"] = r.start + r.h
    return r


# ------------------------------------------------------------------ metrics + blending
def pinball(y, q):
    """mean pinball loss over the three quantiles; y (n,), q (n,3)"""
    return float(np.mean([np.mean(np.maximum(a * (y - q[:, i]), (a - 1) * (y - q[:, i]))) for i, a in enumerate(QS)]))


def wape(y, p):
    return float(np.abs(y - p).sum() / max(np.abs(y).sum(), 1e-9))


def widen(q, k):
    """stretch P10/P90 away from P50 by factor k (interval calibration)"""
    return np.column_stack([np.clip(q[:, 1] - k * (q[:, 1] - q[:, 0]), 0, None), q[:, 1], q[:, 1] + k * (q[:, 2] - q[:, 1])])


def choose_k(y, q, target=0.80):
    """smallest widening factor (1.0 .. 2.5) whose holdout P10-P90 coverage reaches the 80% target"""
    for k in np.round(np.arange(1.0, 2.51, 0.1), 1):
        w = widen(q, k)
        if np.mean((y >= w[:, 0]) & (y <= w[:, 2])) >= target:
            return float(k)
    return 2.5


def choose_weight(y, qx, qt):
    best = (1e18, 0.0)
    for w in np.round(np.arange(0, 1.01, 0.1), 1):
        b = np.sort(w * qt + (1 - w) * qx, axis=1)
        best = min(best, (pinball(y, b), float(w)))
    return best[1]


# ------------------------------------------------------------------ train
def train(epochs, use_tft):
    t0 = time.time()
    df, last_hist = load()
    feats = add_features(df)
    hist = feats[feats.date <= last_hist]
    cut = last_hist - pd.Timedelta(days=HOLDOUT)                 # holdout = last 8 weeks
    tr_b, te_b = hist[hist.date <= cut], hist[hist.date > cut]
    os.makedirs(MODELS, exist_ok=True)

    # ---- 1. XGBoost backtest
    print("[1/4] XGBoost backtest ...", flush=True)
    bt = {}                                                      # (route,cat,target) -> (y, xgb_q, dates)
    for (r, c), dtr in tr_b.groupby(KEYS):
        dte = te_b[(te_b.route_id == r) & (te_b.category == c)]
        for t in TARGETS:
            bt[r, c, t] = dict(y=dte[t].values, dates=dte.date.values, qx=xgb_predict(xgb_fit(dtr, t), dte, t))

    # ---- 2. TFT backtest (trained only on data before the holdout, predicts 8 non-overlapping 7-day blocks)
    tft_ok = use_tft and tft_available()
    if use_tft and not tft_ok:
        print("      PyTorch / pytorch-forecasting not importable -> TFT skipped (XGBoost only)")
    if tft_ok:
        print(f"[2/4] TFT backtest ({epochs} epochs) ...", flush=True)
        frame = tft_frame(df[df.date <= last_hist], last_hist)
        cut_idx = int(frame.loc[frame.date == cut, "time_idx"].iloc[0])
        model, ds, _ = tft_fit(frame[frame.time_idx <= cut_idx], epochs)
        pr = tft_predict(model, ds, frame, min_prediction_idx=cut_idx + 1)
        pr = pr[(pr.start - (cut_idx + 1)) % HORIZON == 0]
        d0 = frame.date.min()
        pr["date"] = d0 + pd.to_timedelta(pr.time_idx, unit="D")
        for (r, c, t), b in bt.items():
            if t not in TFT_TARGETS: continue
            sub = pr[(pr.series == f"{r}_{c}") & (pr.target == t)].drop_duplicates("date").set_index("date")
            sub = sub.reindex(pd.to_datetime(b["dates"]))
            b["qt"] = np.sort(np.clip(sub[["q10", "q50", "q90"]].values, 0, None), axis=1)
    else:
        print("[2/4] TFT backtest skipped")

    # ---- 3. blend weights + metrics
    print("[3/4] choosing blend weights ...", flush=True)
    weights, rows = {}, []
    for (r, c, t), b in bt.items():
        y, qx = b["y"], b["qx"]
        qt = b.get("qt")
        ok = qt is not None and not np.isnan(qt).any()
        w = choose_weight(y, qx, qt) if ok else 0.0
        bl = np.sort(w * qt + (1 - w) * qx, axis=1) if ok else qx
        k = choose_k(y, bl)
        cov_raw = float(np.mean((y >= bl[:, 0]) & (y <= bl[:, 2])))
        bl = widen(bl, k)
        weights[f"{r}|{c}|{t}"] = {"w_tft": w, "k": k}
        rows.append(dict(route_id=r, category=c, target=t, w_tft=w, wape_xgb=wape(y, qx[:, 1]),
                         wape_tft=wape(y, qt[:, 1]) if ok else np.nan, wape_blend=wape(y, bl[:, 1]),
                         pinball_xgb=pinball(y, qx), pinball_tft=pinball(y, qt) if ok else np.nan,
                         pinball_blend=pinball(y, bl), interval_k=k, coverage_raw=cov_raw,
                         coverage_p10_p90=float(np.mean((y >= bl[:, 0]) & (y <= bl[:, 2])))))
    met = pd.DataFrame(rows).round(4)
    met.to_csv(OUT_M, index=False)

    # ---- 4. refit on ALL history and save
    print("[4/4] refitting on all history and saving ...", flush=True)
    models = {(r, c, t): xgb_fit(d, t) for (r, c), d in hist.groupby(KEYS) for t in TARGETS}
    pickle.dump(models, open(f"{MODELS}/xgb_models.pkl", "wb"))
    if tft_ok:
        frame = tft_frame(df[df.date <= last_hist], last_hist)
        model, ds, tr = tft_fit(frame, epochs)
        tr.save_checkpoint(f"{MODELS}/tft.ckpt")
        pickle.dump(ds.get_parameters(), open(f"{MODELS}/tft_dataset.pkl", "wb"))
    meta = dict(trained_at=datetime.now().isoformat(timespec="seconds"), history_end=str(last_hist.date()),
                model_version=("xgb+tft-" if tft_ok else "xgb-only-") + datetime.now().strftime("%Y%m%d"),
                tft=bool(tft_ok), weights=weights)
    json.dump(meta, open(f"{MODELS}/blend_weights.json", "w"), indent=1)

    print(f"\nDone in {time.time() - t0:.0f}s.  model_version = {meta['model_version']}")
    s = met.groupby("target")[["w_tft", "wape_xgb", "wape_tft", "wape_blend", "coverage_raw", "coverage_p10_p90"]].mean().round(3)
    print("Backtest (last 8 weeks, mean over 18 series):\n" + s.to_string())


# ------------------------------------------------------------------ forecast (nightly)
def forecast():
    meta = json.load(open(f"{MODELS}/blend_weights.json"))
    models = pickle.load(open(f"{MODELS}/xgb_models.pkl", "rb"))
    df, last_hist = load()
    feats = add_features(df)
    fut = feats[feats.date > last_hist]
    met = pd.read_csv(OUT_M)

    tft_pred = None
    if meta["tft"] and tft_available() and os.path.exists(f"{MODELS}/tft.ckpt"):
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        frame = tft_frame(df, last_hist)
        ds = TimeSeriesDataSet.from_parameters(pickle.load(open(f"{MODELS}/tft_dataset.pkl", "rb")), frame, predict=True)
        model = TemporalFusionTransformer.load_from_checkpoint(f"{MODELS}/tft.ckpt")
        tft_pred = tft_predict(model, ds, frame, predict=True)
        tft_pred["date"] = frame.date.min() + pd.to_timedelta(tft_pred.time_idx, unit="D")
    elif meta["tft"]:
        print("WARNING: TFT was trained but cannot be loaded here -> falling back to XGBoost only")

    out = []
    for (r, c), d in fut.groupby(KEYS):
        d = d.sort_values("date")
        rec = pd.DataFrame({"generated_at": datetime.now().isoformat(timespec="seconds"), "model_version": meta["model_version"],
                            "target_date": d.date.dt.strftime("%Y-%m-%d").values,
                            "horizon_day": (d.date - last_hist).dt.days.values, "route_id": r, "category": c})
        for t, short in zip(TARGETS, ["weight", "volume", "items"]):
            q = xgb_predict(models[r, c, t], d, t)
            wk = meta["weights"].get(f"{r}|{c}|{t}", {"w_tft": 0.0, "k": 1.0})
            w, k = wk["w_tft"], wk["k"]
            if tft_pred is not None and t in TFT_TARGETS and w > 0:
                sub = tft_pred[(tft_pred.series == f"{r}_{c}") & (tft_pred.target == t)].set_index("date").reindex(d.date.values)
                qt = np.sort(np.clip(sub[["q10", "q50", "q90"]].values, 0, None), axis=1)
                if not np.isnan(qt).any():
                    q = np.sort(w * qt + (1 - w) * q, axis=1)
                else:
                    w = 0.0
            else:
                w = 0.0
            q = widen(q, k)
            for i, p in enumerate(["p10", "p50", "p90"]):
                rec[f"{short}_{p}"] = np.round(q[:, i], 3 if t == "volume_m3" else 1)
            if t in TFT_TARGETS:
                rec[f"w_tft_{short}"] = w
        m = met[(met.route_id == r) & (met.category == c) & (met.target == "weight_kg")].iloc[0]
        rec["backtest_wape_weight"], rec["coverage_p10_p90_weight"] = m.wape_blend, m.coverage_p10_p90
        rec["our_price_per_kg"], rec["competitor_price_per_kg"] = d.our_price_per_kg.values, d.competitor_price_per_kg.values
        out.append(rec)
    res = pd.concat(out, ignore_index=True)
    res.to_csv(OUT_F, index=False)
    print(f"wrote {OUT_F}: {len(res)} rows ({res.route_id.nunique()} routes x {res.category.nunique()} categories x {HORIZON} days), "
          f"model {meta['model_version']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "forecast"])
    ap.add_argument("--no-tft", action="store_true", help="XGBoost only")
    ap.add_argument("--tft-epochs", type=int, default=8)
    a = ap.parse_args()
    train(a.tft_epochs, not a.no_tft) if a.mode == "train" else forecast()
