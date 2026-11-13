"""Forecast U.S. GDP growth for the next four quarters (one year).

GDP is published once a quarter, while the other data on this site is monthly.
So this script works with quarterly data: each quarter's average of the monthly
series, next to that quarter's GDP.

Two models, then their average (same idea as combine_forecast.py):
    ARIMA  GDP growth predicted from its own past only.
    VAR    GDP growth, unemployment, CPI inflation and the Fed funds rate,
           each predicted from the recent past of all four.

Growth is "% change vs. the same quarter a year ago", the way GDP growth is
usually quoted in the news. The number of VAR lags is chosen by BIC.

Output
    data/gdp_forecast.csv  date, arima, var, mean, lower, upper
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
EXTRA_PATH = DATA_DIR / "extra_series.csv"
CPI_PATH = DATA_DIR / "cpi_data.csv"
OUTPUT_PATH = DATA_DIR / "gdp_forecast.csv"

START = "1990-01-01"
HORIZON = 4  # quarters ahead
MAX_LAGS = 4  # look back at most 4 quarters
ARIMA_ORDER = (1, 0, 0)  # this quarter's growth depends on last quarter's
ARIMA_WEIGHT = 0.5  # VAR gets the other half


def monthly(path: Path) -> pd.DataFrame:
    """A CSV on a complete monthly calendar, with one-month gaps filled in."""
    data = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index().asfreq("MS")
    return data.interpolate(limit_area="inside")


def load_quarterly() -> pd.DataFrame:
    """One row per quarter: GDP growth, unemployment, CPI inflation, Fed funds rate."""
    extra = monthly(EXTRA_PATH)
    cpi = monthly(CPI_PATH)["cpi"]

    # GDP sits in the first month of each quarter; the others become quarterly averages.
    gdp = extra["real_gdp"].dropna().resample("QS").first()
    averages = (
        pd.DataFrame(
            {
                "unemployment": extra["unemployment"],
                "fed_funds": extra["fed_funds"],
                "cpi_level": cpi,
            }
        )
        .resample("QS")
        .mean()
    )

    data = pd.DataFrame(
        {
            "gdp": gdp.pct_change(4, fill_method=None) * 100,
            "unemployment": averages["unemployment"],
            "cpi": averages["cpi_level"].pct_change(4, fill_method=None) * 100,
            "fed_funds": averages["fed_funds"],
        }
    )
    data = data[data.index >= START]

    # Keep quarters where everything is available (GDP comes out about a month after a quarter ends).
    return data.dropna().asfreq("QS")


def run() -> None:
    warnings.filterwarnings("ignore")
    data = load_quarterly()
    last = data.index.max()
    dates = pd.date_range(last, periods=HORIZON + 1, freq="QS")[1:]

    # ARIMA on GDP growth alone.
    arima = ARIMA(data["gdp"], order=ARIMA_ORDER).fit().get_forecast(steps=HORIZON).summary_frame(alpha=0.05)

    # VAR on all four series.
    lags = max(1, VAR(data).select_order(maxlags=MAX_LAGS).bic)
    var_mean, var_lower, var_upper = (
        VAR(data).fit(lags).forecast_interval(data.to_numpy()[-lags:], steps=HORIZON, alpha=0.05)
    )
    g = list(data.columns).index("gdp")

    w = ARIMA_WEIGHT
    output = pd.DataFrame(
        {
            "date": dates,
            "arima": arima["mean"].to_numpy(),
            "var": var_mean[:, g],
            "lower": w * arima["mean_ci_lower"].to_numpy() + (1 - w) * var_lower[:, g],
            "upper": w * arima["mean_ci_upper"].to_numpy() + (1 - w) * var_upper[:, g],
        }
    )
    output.insert(3, "mean", w * output["arima"] + (1 - w) * output["var"])

    numbers = ["arima", "var", "mean", "lower", "upper"]
    output[numbers] = output[numbers].round(4)
    output.to_csv(OUTPUT_PATH, index=False, date_format="%Y-%m-%d")

    print(f"Quarterly data {data.index.min():%Y-%m} to {last:%Y-%m}; VAR({lags})")
    print(f"GDP growth now: {data['gdp'].iloc[-1]:.2f}% vs. a year ago")
    for _, row in output.iterrows():
        q = f"{row['date'].year} Q{(row['date'].month - 1) // 3 + 1}"
        print(f"  {q}: ARIMA {row['arima']:.2f}%  VAR {row['var']:.2f}%  -> average {row['mean']:.2f}%")
    print(f"Saved {OUTPUT_PATH.name}")


if __name__ == "__main__":
    run()
