"""12-month ARIMA forecasts of core CPI inflation and the unemployment rate.

These two series get an ARIMA forecast so that, like CPI, PCE and interest rates,
they can be averaged with the VAR forecast in combine_forecast.py.

- Core CPI (all items less food and energy) is turned into "% change vs. a year
  ago", the same way as headline CPI and PCE. The ARIMA order is chosen by AIC,
  using the same candidates as the PCE forecast.
- Unemployment is used as it is (already in %), like the interest rates. The
  ARIMA order is chosen by AIC, using the same candidates as the rate forecasts.

October 2025 was never published for either series (government shutdown).
A missing month inside a series is filled with the average of its neighbours.

Output
    data/extra_forecast.csv  series, model, date, mean, lower, upper
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from data_utils import load_extra_yoy
from horizon_backtest import HORIZON
import pce_forecast
import rates_forecast

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_PATH = DATA_DIR / "extra_series.csv"
OUTPUT_PATH = DATA_DIR / "extra_forecast.csv"
START = "1990-01-01"


def load_core_cpi() -> pd.Series:
    """Core CPI inflation, % vs. a year ago, one row per month."""
    yoy = load_extra_yoy(DATA_PATH, ["core"])["core"]
    return yoy[yoy.index >= START].dropna().asfreq("MS")


def load_unemployment() -> pd.Series:
    """Unemployment rate in %, one row per month, interior gaps filled."""
    data = pd.read_csv(DATA_PATH, parse_dates=["date"]).set_index("date").sort_index().asfreq("MS")
    level = data["unemployment"].interpolate(limit_area="inside")
    return level[level.index >= START].dropna().asfreq("MS")


def forecast(series: pd.Series, order: tuple[int, int, int], floor_zero: bool) -> pd.DataFrame:
    frame = ARIMA(series, order=order).fit().get_forecast(steps=HORIZON).summary_frame(alpha=0.05)
    output = pd.DataFrame(
        {
            "date": frame.index,
            "mean": frame["mean"].to_numpy(),
            "lower": frame["mean_ci_lower"].to_numpy(),
            "upper": frame["mean_ci_upper"].to_numpy(),
        }
    )
    if floor_zero:  # an unemployment rate cannot go below zero
        output[["mean", "lower"]] = output[["mean", "lower"]].clip(lower=0)
    return output


def run() -> None:
    warnings.filterwarnings("ignore")
    rows = []

    jobs = [
        # name, label, data, how to pick the order, floor at zero
        ("core_cpi", "Core CPI", load_core_cpi(), pce_forecast.select_order, False),
        ("unemployment", "Unemployment", load_unemployment(), rates_forecast.select_order, True),
    ]
    for name, label, series, select_order, floor_zero in jobs:
        order = select_order(series)
        model = f"ARIMA{order}".replace(" ", "")
        frame = forecast(series, order, floor_zero).assign(series=name, model=model)
        rows.append(frame)
        print(f"{label}: {model}, latest {series.iloc[-1]:.2f}%, in 12 months {frame['mean'].iloc[-1]:.2f}%")

    cols = ["series", "model", "date", "mean", "lower", "upper"]
    pd.concat(rows)[cols].round(4).to_csv(OUTPUT_PATH, index=False, date_format="%Y-%m-%d")
    print(f"Saved {OUTPUT_PATH.name}")


if __name__ == "__main__":
    run()
