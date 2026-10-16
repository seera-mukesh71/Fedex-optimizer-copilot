"""
data_gen.py  -  builds the SYNTHETIC inputs for the whole project.

    python data_gen.py                 # history ends yesterday
    python data_gen.py --end 2026-10-15 --seed 7

Outputs
  data/demand_data.csv : 3 years of daily history + 7 future rows per series
                         3 states -> 6 ordered routes x 3 categories = 18 series
  data/network.json    : assets (flights / charter / trailers) per route, category properties,
                         downgrade rule and default business settings used by the optimizer

EVERYTHING here is illustrative.  Costs, capacities and elasticities are assumptions, not FedEx data.
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))   # always run relative to the project folder
import argparse, json, os
from datetime import date, timedelta

import numpy as np
import pandas as pd

HIST_DAYS = 3 * 365          # 1095 days -> 18 series * 1095 = 19,710 history rows
HORIZON = 7                  # future days we forecast

STATES = {"TN": "Tamil Nadu (Chennai)", "MH": "Maharashtra (Mumbai)", "KA": "Karnataka (Bengaluru)"}

# per unordered state pair: distance, price level, owned-trailer trip cost, charter fixed cost (Rs)
PAIRS = {
    frozenset(["TN", "MH"]): dict(km=1330, price_factor=1.00, owned_trip=60000, charter_fixed=140000),
    frozenset(["TN", "KA"]): dict(km=350,  price_factor=0.70, owned_trip=22000, charter_fixed=80000),
    frozenset(["MH", "KA"]): dict(km=980,  price_factor=0.90, owned_trip=44000, charter_fixed=120000),
}
# 6 ordered routes and a demand scale (1.0 = the "reference" lane)
ROUTES = [("TN", "MH", 1.00), ("MH", "TN", 0.85), ("TN", "KA", 0.60),
          ("KA", "TN", 0.55), ("MH", "KA", 0.70), ("KA", "MH", 0.65)]

CATEGORIES = {   # base = kg/day on the reference lane at a normal (non-peak) weekday
    "MED":  dict(name="Medicine",   base=3000, rho=0.004,  avg_item=0.4, p_ref=62, elast=-0.4, mode="air",
                 sigma=0.08, peak=1.00, promo=1.00, unmet_penalty=150, downgrade_eligible=False,
                 dow=[1.00, 1.05, 1.05, 1.00, 1.00, 0.80, 0.60]),
    "ELEC": dict(name="Electrical", base=4000, rho=0.005,  avg_item=2.5, p_ref=55, elast=-1.2, mode="air",
                 sigma=0.11, peak=1.25, promo=1.18, unmet_penalty=60, downgrade_eligible=True,
                 dow=[1.05, 1.00, 1.00, 1.00, 1.05, 0.85, 0.70]),
    "APP":  dict(name="Apparel",    base=9000, rho=0.0078, avg_item=0.6, p_ref=22, elast=-1.8, mode="ground",
                 sigma=0.14, peak=1.20, promo=1.25, unmet_penalty=10, downgrade_eligible=False,
                 dow=[0.95, 1.00, 1.00, 1.05, 1.10, 1.00, 0.70]),
}

# Diwali / Holi dates (approx.) + fixed national holidays
MOVING_HOLIDAYS = ["2022-10-24", "2023-11-12", "2024-11-01", "2025-10-20", "2026-11-08", "2027-10-29",
                   "2023-03-08", "2024-03-25", "2025-03-14", "2026-03-04", "2027-03-22"]


def holiday_set(y0, y1):
    s = {pd.Timestamp(d) for d in MOVING_HOLIDAYS}
    for y in range(y0, y1 + 1):
        for md in ["01-26", "05-01", "08-15", "10-02", "12-25"]:
            s.add(pd.Timestamp(f"{y}-{md}"))
    return s


def ar1(rng, n, phi, sd, x0=0.0):
    out, x = np.empty(n), x0
    for i in range(n):
        x = phi * x + rng.normal(0, sd)
        out[i] = x
    return out


def make_series(rng, dates, route_scale, price_factor, cat, c, fuel, hol, n_hist):
    n = len(dates)
    dow = dates.dayofweek.values
    month = dates.month.values
    doy = dates.dayofyear.values
    t_years = np.arange(n) / 365.0

    is_hol = np.array([d in hol for d in dates], dtype=int)
    pre_hol = np.zeros(n, dtype=int)
    pre_hol[:-2] = np.maximum(is_hol[1:-1], is_hol[2:])           # 1-2 days before a holiday
    is_peak = np.isin(month, [10, 11, 12]).astype(int)

    promo = np.zeros(n, dtype=int)
    if c["promo"] > 1.0:                                            # promos only for ELEC / APP
        i = 0
        while i < n:
            if rng.random() < 0.02:
                promo[i:i + 7] = 1; i += 14
            else:
                i += 1

    ours = c["p_ref"] * price_factor * (1 + 0.15 * (fuel - 1))
    comp_noise = ar1(rng, n, 0.95, 0.02)
    comp = ours * np.exp(comp_noise)
    ours = np.round(ours, 2); comp = np.round(comp, 2)
    ours[n_hist:] = ours[n_hist - 1]; comp[n_hist:] = comp[n_hist - 1]          # future: hold last known prices

    weather = np.clip(0.3 + ar1(rng, n, 0.8, 0.12), 0, 1)

    seasonal = 1 + 0.06 * np.sin(2 * np.pi * (doy - 60) / 365)
    f_peak = 1 + (c["peak"] - 1) * is_peak
    f_hol = np.where(is_hol == 1, 0.75 if cat != "MED" else 0.9, 1.0) * np.where(pre_hol == 1, 1.08 if cat != "MED" else 1.0, 1.0)
    f_promo = np.where(promo == 1, c["promo"], 1.0)
    f_trend = np.exp(0.04 * t_years)
    f_price = (ours / comp) ** c["elast"]
    f_weather = 1 - 0.10 * weather
    level = ar1(rng, n, 0.92, 0.03)

    base = c["base"] * route_scale
    det = base * np.array(c["dow"])[dow] * seasonal * f_peak * f_hol * f_promo * f_trend * f_price * f_weather
    expected = det * np.exp(level)
    weight = expected * np.exp(rng.normal(0, c["sigma"], n))
    volume = weight * c["rho"] * np.exp(rng.normal(0, 0.05, n))
    items = np.round(weight / c["avg_item"] * np.exp(rng.normal(0, 0.05, n)))
    signal = det * np.exp(0.5 * level) * np.exp(rng.normal(0, 0.12, n))        # key-account booking pipeline (known in advance)

    df = pd.DataFrame({
        "date": dates, "category": cat,
        "weight_kg": np.round(weight, 1), "volume_m3": np.round(volume, 3), "item_count": items,
        "day_of_week": dow, "month": month, "is_holiday": is_hol, "is_peak_season": is_peak, "promo_flag": promo,
        "client_forecast_signal_kg": np.round(signal, 1),
        "our_price_per_kg": ours, "competitor_price_per_kg": comp,
        "fuel_price_index": np.round(fuel, 4), "weather_severity": np.round(weather, 3)})
    df.loc[n_hist:, ["weight_kg", "volume_m3", "item_count", "fuel_price_index", "weather_severity"]] = np.nan
    return df


def build_demand(end_date, seed):
    rng = np.random.default_rng(seed)
    start = end_date - timedelta(days=HIST_DAYS - 1)
    dates = pd.date_range(start, end_date + timedelta(days=HORIZON))
    n_hist = HIST_DAYS
    hol = holiday_set(dates[0].year, dates[-1].year + 1)
    fuel = 1 + ar1(rng, len(dates), 0.99, 0.01)
    frames = []
    for o, d, sc in ROUTES:
        pf = PAIRS[frozenset([o, d])]["price_factor"]
        for cat, c in CATEGORIES.items():
            df = make_series(rng, dates, sc, pf, cat, c, fuel, hol, n_hist)
            df.insert(1, "route_id", f"{o}-{d}"); df.insert(2, "origin_state", o); df.insert(3, "dest_state", d)
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


def rnd(x, base=50):
    return int(base * round(x / base))


def build_network():
    routes = {}
    for o, d, sc in ROUTES:
        p = PAIRS[frozenset([o, d])]
        pf = p["price_factor"]
        routes[f"{o}-{d}"] = {
            "origin": o, "dest": d, "distance_km": p["km"], "price_factor": pf,
            "typical_spend_cap": int(round(200000 * pf, -3)),
            # scheduled flights already flying: only the RESIDUAL space is ours
            "air_flights": {
                "F1": {"cap_w": rnd(4500 * sc), "cap_v": round(28 * sc, 1), "var_cost": round(30 * pf, 1)},
                "F2": {"cap_w": rnd(2000 * sc), "cap_v": round(10 * sc, 1), "var_cost": round(30 * pf, 1)}},
            "charter": {"cap_w": 4000, "cap_v": 20, "fixed": p["charter_fixed"], "var_cost": 0, "available": 1},
            "trailers": {"cap_w": 8000, "cap_v": 40, "owned": 1, "owned_trip": p["owned_trip"],
                         "rentable": 3, "rent_trip": int(round(p["owned_trip"] * 1.13, -2)),
                         "handling": round(3 * pf, 2)}}
    return {
        "states": STATES,
        "categories": {k: {"name": c["name"], "rho": c["rho"], "mode": c["mode"], "elasticity": c["elast"],
                           "unmet_penalty": c["unmet_penalty"], "downgrade_eligible": c["downgrade_eligible"]}
                       for k, c in CATEGORIES.items()},
        # a downgraded air parcel travels on trailers and is paid price_ratio x today's price
        "downgrade": {"category": "ELEC", "to_mode": "ground", "price_ratio": 0.51, "rho": 0.005, "max_share": 0.30},
        "defaults": {"price_grid": [0.90, 0.95, 1.00, 1.05, 1.10], "min_fill": 0.60, "min_margin_segment": 0.25,
                     "min_margin_lane": 0.25, "allow_downgrade": True, "allow_unmet": False},
        "routes": routes,
        "note": "All costs, capacities and elasticities are illustrative assumptions."}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=(date.today() - timedelta(days=1)).isoformat(), help="last history date (YYYY-MM-DD)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs("data", exist_ok=True)
    df = build_demand(date.fromisoformat(a.end), a.seed)
    df.to_csv("data/demand_data.csv", index=False)
    with open("data/network.json", "w") as f:
        json.dump(build_network(), f, indent=2)
    hist = df[df.weight_kg.notna()]
    print(f"demand_data.csv: {len(df):,} rows ({len(hist):,} history + {len(df) - len(hist)} future), "
          f"{df.groupby(['route_id', 'category']).ngroups} series, history ends {a.end}")
    print("network.json written.")
    print(hist.groupby("category").weight_kg.agg(["mean", "std"]).round(0).to_string())
