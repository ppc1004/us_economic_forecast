"""12-month VAR forecast of interest rates, inflation and unemployment.

A VAR (vector autoregression) forecasts several series together. Each series is
predicted from the recent past of every series in the model, so the interest-rate
forecast can react to what inflation, jobs, oil and wages are doing.

Nine monthly inputs:
    fed_funds, gs2, gs10, unemployment      used as they are (already in %)
    cpi, core_pce, industrial_production,   turned into "% change vs. a year ago"
    oil_wti, wages

The number of lags (how many past months the model looks at) is chosen by BIC.
The model is fit on data since 1990 and stops at the last month where every
input has been published.

Output
    data/var_forecast.csv  series, model, date, mean, lower, upper
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
from statsmodels.tsa.api import VAR

from data_utils import load_column_yoy, load_extra_yoy, load_yoy

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
EXTRA_PATH = DATA_DIR / "extra_series.csv"
CPI_PATH = DATA_DIR / "cpi_data.csv"
PCE_PATH = DATA_DIR / "pce_data.csv"
OUTPUT_PATH = DATA_DIR / "var_forecast.csv"

START = "1990-01-01"
MAX_LAGS = 6  # look back at most 6 months
HORIZON = 12  # forecast 12 months past the latest Fed funds reading
LEVEL_COLUMNS = ["fed_funds", "gs2", "gs10", "unemployment"]
YOY_COLUMNS = ["industrial_production", "oil_wti", "wages"]
RATE_SERIES = ["fed_funds", "gs2", "gs10"]  # interest rates cannot go below zero
OUTPUT_SERIES = ["fed_funds", "gs2", "gs10", "cpi", "core_pce", "unemployment"]


def load_levels() -> pd.DataFrame:
    """Rates and unemployment as they are, one row per calendar month.

    Unemployment for October 2025 was never published (government shutdown).
    A missing month inside a series is filled with the average of its neighbours.
    """
    data = pd.read_csv(EXTRA_PATH, parse_dates=["date"]).set_index("date").sort_index().asfreq("MS")
    return data[LEVEL_COLUMNS].interpolate(limit_area="inside")


def load_inputs() -> pd.DataFrame:
    """Put all nine inputs side by side, one row per month."""
    columns = dict(load_levels().items())
    columns["cpi"] = load_yoy(CPI_PATH)
    columns["core_pce"] = load_column_yoy(PCE_PATH, "core_pce")

    growth = load_extra_yoy(EXTRA_PATH, YOY_COLUMNS)
    for name in YOY_COLUMNS:
        columns[name] = growth[name]

    data = pd.DataFrame(columns)
    data = data[data.index >= START]

    # Oil is a daily series, so the current month is only a partial average.
    # Drop any month that has not finished yet.
    this_month = pd.Timestamp.today().to_period("M").to_timestamp()
    data = data[data.index < this_month]

    # Keep only months where every input is available. The newest month for
    # some series (e.g. industrial production) comes out later than for others.
    return data.dropna().asfreq("MS")


def run() -> None:
    warnings.filterwarnings("ignore")
    data = load_inputs()

    # Choose how many past months to use: try 1 to MAX_LAGS, keep the best BIC.
    lags = max(1, VAR(data).select_order(maxlags=MAX_LAGS).bic)
    fitted = VAR(data).fit(lags)

    # Forecast far enough ahead to reach 12 months past the latest Fed funds data.
    latest_rate = load_levels()["fed_funds"].last_valid_index()
    last_common = data.index.max()
    gap = (latest_rate.year - last_common.year) * 12 + (latest_rate.month - last_common.month)
    steps = gap + HORIZON

    history = data.to_numpy()[-lags:]
    mean, lower, upper = fitted.forecast_interval(history, steps=steps, alpha=0.05)
    dates = pd.date_range(last_common, periods=steps + 1, freq="MS")[1:]

    rows = []
    for i, name in enumerate(data.columns):
        if name not in OUTPUT_SERIES:
            continue
        frame = pd.DataFrame(
            {
                "series": name,
                "model": f"VAR({lags})",
                "date": dates,
                "mean": mean[:, i],
                "lower": lower[:, i],
                "upper": upper[:, i],
            }
        )
        if name in RATE_SERIES:
            frame[["mean", "lower"]] = frame[["mean", "lower"]].clip(lower=0)
        rows.append(frame.tail(HORIZON))  # keep the same 12 months as the ARIMA forecasts

    output = pd.concat(rows)
    output.round(4).to_csv(OUTPUT_PATH, index=False, date_format="%Y-%m-%d")

    print(f"VAR({lags}) on {len(data)} months, {data.index.min():%Y-%m} to {last_common:%Y-%m}")
    for name in OUTPUT_SERIES:
        last = data[name].iloc[-1]
        end = output[output["series"] == name].iloc[-1]
        print(f"{name:>13}: now {last:5.2f}%  ->  {end['date']:%Y-%m} {end['mean']:5.2f}%")
    print(f"Saved {OUTPUT_PATH.name}")


if __name__ == "__main__":
    run()
