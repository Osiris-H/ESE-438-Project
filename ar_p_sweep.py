#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment B (6h): AR(p) sweep on anomaly for "Appropriate Lag"
- Build single-point 6h series (from 3h by hour filter)
- Build anomaly via climatology (same as ExpA)
- For p=1..Pmax:
    * Fit AutoReg(p) on TRAIN to get AIC/BIC (1-step likelihood)
    * Fit DIRECT 24h (h=4 steps) linear AR(p) regression on TRAIN:
        y[t+h] = sum_k a_k y[t+1-k] + c
      Evaluate on TEST for RMSE/MAE/Bias
- Save CSV + plot curves

Run:
  python stat_expB_ar_sweep_anom_6h.py --data_dir ./era5_1p5deg --var t2m --analysis_year 2020 --years 2019 2020 2021
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from statsmodels.tsa.ar_model import AutoReg


# -----------------------------
# Reuse minimal loaders (same logic as ExpA)
# -----------------------------
def list_monthly_files(data_dir: Path, year: int):
    return [data_dir / f"era5_single_{year}_{m:02d}_1p5deg.nc" for m in range(1, 13)
            if (data_dir / f"era5_single_{year}_{m:02d}_1p5deg.nc").exists()]


def detect_time_dim(ds: xr.Dataset) -> str:
    if "valid_time" in ds.dims:
        return "valid_time"
    if "time" in ds.dims:
        return "time"
    for d in ds.dims:
        if "time" in d.lower():
            return d
    raise KeyError(f"Cannot find time dimension. ds.dims={dict(ds.dims)}")


def load_point_series_3h(data_dir: Path, year: int, var: str, lat: float, lon: float) -> pd.Series:
    files = list_monthly_files(data_dir, year)
    if not files:
        raise FileNotFoundError(f"No monthly files found for year={year} in {data_dir}")
    parts = []
    for f in files:
        ds = xr.open_dataset(f)
        tdim = detect_time_dim(ds)
        if var not in ds.data_vars:
            avail = list(ds.data_vars)
            ds.close()
            raise KeyError(f"{var} not in {f.name}. Available vars: {avail}")
        da = ds[var].sel(latitude=lat, longitude=lon, method="nearest")
        t = pd.to_datetime(ds[tdim].values)
        parts.append(pd.Series(da.values, index=t))
        ds.close()
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    return s


def to_6h_by_hour_filter(s_3h: pd.Series) -> pd.Series:
    idx = s_3h.index
    keep = idx.hour.isin([0, 6, 12, 18])
    return s_3h.loc[keep].dropna().sort_index()


def convert_units(s: pd.Series, var: str) -> pd.Series:
    if var in ("t2m", "d2m"):
        return s - 273.15
    return s


def build_climatology_6h(data_dir: Path, years: list[int], var: str, lat: float, lon: float, window_days: int = 61):
    assert window_days % 2 == 1
    half = window_days // 2

    all_s = []
    used = []
    for y in years:
        try:
            s3 = load_point_series_3h(data_dir, y, var, lat, lon)
            s6 = to_6h_by_hour_filter(convert_units(s3, var))
            if len(s6) > 0:
                all_s.append(s6)
                used.append(y)
        except FileNotFoundError:
            pass
    if not all_s:
        raise RuntimeError("No data for climatology years.")

    big = pd.concat(all_s).sort_index().dropna()
    df = pd.DataFrame({"x": big})
    df["doy"] = df.index.dayofyear
    df["tod"] = df.index.hour

    doys = np.arange(1, 367)
    tods = sorted(df["tod"].unique().tolist())

    clim_map = {}
    for tod in tods:
        sub = df[df["tod"] == tod]
        by_doy = sub.groupby("doy")["x"].mean()
        ext_index = np.concatenate([doys - 366, doys, doys + 366])
        ext_values = np.concatenate([by_doy.reindex(doys).values,
                                     by_doy.reindex(doys).values,
                                     by_doy.reindex(doys).values])
        ext = pd.Series(ext_values, index=ext_index)
        for d in doys:
            clim_map[(d, tod)] = float(ext.loc[d - half:d + half].mean())

    clim = pd.Series([clim_map[(t.dayofyear, t.hour)] for t in df.index], index=df.index, name="climatology")
    return clim, used


# -----------------------------
# Direct 24h regression AR(p) evaluation
# -----------------------------
def train_test_split_series(s: pd.Series, test_ratio: float = 0.2):
    n = len(s)
    n_test = max(1, int(n * test_ratio))
    train = s.iloc[:-n_test].copy()
    test = s.iloc[-n_test:].copy()
    return train, test


def direct_ar_fit_predict(train: pd.Series, test: pd.Series, p: int, h: int):
    """
    Direct h-step AR(p):
      y[t+h] = c + sum_{k=1..p} a_k * y[t+1-k]
    Fit on TRAIN only, then predict for TEST indices where features available.
    """
    y = train.values.astype(float)

    # Build design matrix on TRAIN
    # valid t: need t-p >= 0 and t+h < len(train)
    rows = []
    targets = []
    for t in range(p, len(train) - h):
        x = y[t - p:t][::-1]   # [y[t-1], y[t-2], ..., y[t-p]]
        rows.append(x)
        targets.append(y[t + h])

    X = np.asarray(rows)
    Y = np.asarray(targets)

    if len(X) < max(20, 3 * p):
        raise RuntimeError(f"Not enough samples to fit direct AR(p) with p={p}, h={h}. Got {len(X)} samples.")

    # Add intercept
    X1 = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    beta, *_ = np.linalg.lstsq(X1, Y, rcond=None)

    # Predict on TEST by using past values from (TRAIN+TEST) history
    full = pd.concat([train, test]).copy()
    full_y = full.values.astype(float)

    preds = []
    pred_index = []

    # For each test time index i_test in full:
    # we predict y[i_test] using features ending at (i_test - h)
    # because y[(t)+h] target corresponds to time (t+h)
    # => to predict at time tau, use t = tau - h
    for tau in test.index:
        i_tau = full.index.get_loc(tau)
        t = i_tau - h
        if t < p:
            continue
        x = full_y[t - p:t][::-1]
        yhat = beta[0] + float(np.dot(beta[1:], x))
        preds.append(yhat)
        pred_index.append(tau)

    y_pred = pd.Series(preds, index=pd.to_datetime(pred_index), name="pred")
    y_true = test.reindex(y_pred.index).rename("truth")

    return y_true, y_pred


def det_metrics(y_true: pd.Series, y_pred: pd.Series):
    df = pd.concat([y_true.rename("truth"), y_pred.rename("pred")], axis=1).dropna()
    if len(df) == 0:
        return {"MSE": np.nan, "RMSE": np.nan, "MAE": np.nan, "Bias": np.nan, "N": 0}
    err = df["pred"] - df["truth"]
    mse = float(np.mean(err.values ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err.values)))
    bias = float(np.mean(err.values))
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "Bias": bias, "N": int(len(df))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./era5_1p5deg")
    ap.add_argument("--var", type=str, default="t2m")
    ap.add_argument("--analysis_year", type=int, default=2020)
    ap.add_argument("--years", type=int, nargs="+", default=[2019, 2020, 2021], help="climatology years")
    ap.add_argument("--lat", type=float, default=39.9526)
    ap.add_argument("--lon", type=float, default=-75.1652)
    ap.add_argument("--pmax", type=int, default=40)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--horizon_steps", type=int, default=4, help="6h steps; 24h => 4")
    ap.add_argument("--out_dir", type=str, default="./stat_outputs/expB")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build 6h raw series
    s3 = load_point_series_3h(data_dir, args.analysis_year, args.var, args.lat, args.lon)
    raw6 = to_6h_by_hour_filter(convert_units(s3, args.var)).dropna()
    raw6.name = "raw_6h"

    # Build climatology + anomaly
    clim_full, used = build_climatology_6h(data_dir, args.years, args.var, args.lat, args.lon, window_days=61)
    clim6 = clim_full.reindex(raw6.index).fillna(clim_full.mean())
    anom6 = (raw6 - clim6).rename("anom_6h").dropna()

    # Split
    train, test = train_test_split_series(anom6, test_ratio=args.test_ratio)

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in range(1, args.pmax + 1):
            # AIC/BIC from AutoReg one-step likelihood (TRAIN only)
            try:
                ar_res = AutoReg(train, lags=p, old_names=False).fit()
                aic = float(ar_res.aic) if hasattr(ar_res, "aic") else np.nan
                bic = float(ar_res.bic) if hasattr(ar_res, "bic") else np.nan
            except Exception:
                aic = np.nan
                bic = np.nan

            # 24h (h=4) direct regression metrics on TEST
            try:
                y_true, y_pred = direct_ar_fit_predict(train, test, p=p, h=args.horizon_steps)
                met = det_metrics(y_true, y_pred)
            except Exception as e:
                met = {"MSE": np.nan, "RMSE": np.nan, "MAE": np.nan, "Bias": np.nan, "N": 0}

            rows.append({
                "var": args.var,
                "analysis_year": args.analysis_year,
                "lat": args.lat,
                "lon": args.lon,
                "clim_years_used": ",".join(map(str, used)),
                "p": p,
                "horizon_steps": args.horizon_steps,
                "AIC": aic,
                "BIC": bic,
                "MSE_24h": met["MSE"],
                "RMSE_24h": met["RMSE"],
                "MAE_24h": met["MAE"],
                "Bias_24h": met["Bias"],
                "N_test_effective": met["N"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "stat_expB_ar_sweep.csv", index=False)

    # Plot curves
    plt.figure()
    plt.plot(df["p"], df["RMSE_24h"], label="RMSE_24h (direct h=4)")
    plt.xlabel("AR order p")
    plt.ylabel("RMSE (anomaly units)")
    plt.title(f"AR(p) sweep on anomaly (6h), 24h horizon: {args.var}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "ar_sweep_rmse24h.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(df["p"], df["AIC"], label="AIC (1-step AutoReg)")
    plt.plot(df["p"], df["BIC"], label="BIC (1-step AutoReg)")
    plt.xlabel("AR order p")
    plt.ylabel("Information Criterion")
    plt.title(f"AIC/BIC vs p (TRAIN, anomaly): {args.var}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "ar_sweep_aic_bic.png", dpi=200)
    plt.close()

    # Print best p by RMSE_24h (tie-break: smaller p)
    df_valid = df.dropna(subset=["RMSE_24h"]).copy()
    if len(df_valid) > 0:
        best = df_valid.sort_values(["RMSE_24h", "p"], ascending=[True, True]).iloc[0]
        print("\n[Best p by RMSE_24h]")
        print(best[["p", "RMSE_24h", "MAE_24h", "Bias_24h", "AIC", "BIC"]])
    else:
        print("\n[WARN] All RMSE_24h are NaN. Check data length / point location / series quality.")

    print(f"\n[Saved] {out_dir/'stat_expB_ar_sweep.csv'}")
    print(f"[Saved] {out_dir/'ar_sweep_rmse24h.png'}")
    print(f"[Saved] {out_dir/'ar_sweep_aic_bic.png'}")


if __name__ == "__main__":
    main()