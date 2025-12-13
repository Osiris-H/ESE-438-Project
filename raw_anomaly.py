# raw_anomaly_global_random.py
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf

# -----------------------------
# Config
# -----------------------------
DATA_DIR = Path("era5_1p5deg")
OUT_DIR = Path("stat_outputs/expA_global")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VAR = "t2m"
YEAR = 2020
CLIM_YEARS = [2019, 2020, 2021]

K_POINTS = 20          # Random pick some 
SUBSAMPLE_STEP = 2     # 3h → 6h
ROLL_WIN = 30          # rolling variance window (≈7.5 days)

SEED = 42
np.random.seed(SEED)

# -----------------------------
# Helpers
# -----------------------------
def list_monthly_files(year):
    files = []
    for m in range(1, 13):
        f = DATA_DIR / f"era5_single_{year}_{m:02d}_1p5deg.nc"
        if f.exists():
            files.append(f)
    return files


def load_point_series(year, lat_idx, lon_idx):
    parts = []
    for f in list_monthly_files(year):
        ds = xr.open_dataset(f)
        time_name = "time" if "time" in ds.dims else "valid_time"
        da = ds[VAR].isel(latitude=lat_idx, longitude=lon_idx)
        ts = pd.Series(da.values, index=pd.to_datetime(ds[time_name].values))
        parts.append(ts - 273.15)  # K → C
        ds.close()

    s = pd.concat(parts).sort_index()
    s = s.iloc[::SUBSAMPLE_STEP]  # 3h → 6h
    return s.dropna()


def compute_climatology(lat_idx, lon_idx):
    all_series = []
    for y in CLIM_YEARS:
        all_series.append(load_point_series(y, lat_idx, lon_idx))
    df = pd.concat(all_series, axis=1)
    return df.mean(axis=1)


def fit_sarimax(ts):
    model = SARIMAX(
        ts,
        order=(2, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    return res.aic, res.bic, np.var(res.resid)


def acf_decay_lag(ts, threshold=0.1, max_lag=200):
    vals = acf(ts, nlags=max_lag, fft=True)
    for i in range(1, len(vals)):
        if abs(vals[i]) < threshold:
            return i
    return np.nan


# -----------------------------
# Main experiment
# -----------------------------
def main():
    # 1) Rand
    first = xr.open_dataset(list_monthly_files(YEAR)[0])
    nlat, nlon = first.dims["latitude"], first.dims["longitude"]
    first.close()

    points = set()
    while len(points) < K_POINTS:
        points.add((np.random.randint(0, nlat), np.random.randint(0, nlon)))
    points = list(points)

    rows = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for i, (lat_i, lon_i) in enumerate(points):
            print(f"[{i+1}/{K_POINTS}] lat_idx={lat_i}, lon_idx={lon_i}")

            raw = load_point_series(YEAR, lat_i, lon_i)
            clim = compute_climatology(lat_i, lon_i)
            anom = raw - clim.reindex(raw.index).values

            # --- basic stats
            raw_var = raw.var()
            anom_var = anom.var()

            raw_roll = raw.rolling(ROLL_WIN).var().dropna()
            anom_roll = anom.rolling(ROLL_WIN).var().dropna()

            raw_acf1 = acf(raw, nlags=1, fft=True)[1]
            anom_acf1 = acf(anom, nlags=1, fft=True)[1]

            raw_decay = acf_decay_lag(raw)
            anom_decay = acf_decay_lag(anom)

            # --- SARIMAX diagnostics
            raw_aic, raw_bic, raw_resid_var = fit_sarimax(raw)
            anom_aic, anom_bic, anom_resid_var = fit_sarimax(anom)

            rows.append({
                "lat_idx": lat_i,
                "lon_idx": lon_i,
                "raw_var": raw_var,
                "anom_var": anom_var,
                "raw_rollvar_mean": raw_roll.mean(),
                "anom_rollvar_mean": anom_roll.mean(),
                "raw_acf1": raw_acf1,
                "anom_acf1": anom_acf1,
                "raw_acf_decay_lag": raw_decay,
                "anom_acf_decay_lag": anom_decay,
                "raw_aic": raw_aic,
                "anom_aic": anom_aic,
                "raw_bic": raw_bic,
                "anom_bic": anom_bic,
                "raw_resid_var": raw_resid_var,
                "anom_resid_var": anom_resid_var,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "stat_expA_global_random.csv", index=False)

    # 2) all sums
    summary = {
        "K_points": K_POINTS,
        "raw_var_mean": df["raw_var"].mean(),
        "anom_var_mean": df["anom_var"].mean(),
        "raw_acf1_mean": df["raw_acf1"].mean(),
        "anom_acf1_mean": df["anom_acf1"].mean(),
        "raw_aic_mean": df["raw_aic"].mean(),
        "anom_aic_mean": df["anom_aic"].mean(),
        "raw_bic_mean": df["raw_bic"].mean(),
        "anom_bic_mean": df["anom_bic"].mean(),
    }

    pd.DataFrame([summary]).to_csv(
        OUT_DIR / "stat_expA_global_random_summary.csv",
        index=False
    )

    print("\n[Saved]")
    print(OUT_DIR / "stat_expA_global_random.csv")
    print(OUT_DIR / "stat_expA_global_random_summary.csv")


if __name__ == "__main__":
    main()