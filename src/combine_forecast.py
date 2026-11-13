"""Average the ARIMA and VAR forecasts, half and half.

Different models make different mistakes, so an average of the two is usually
steadier than either one alone. For every series and month that both models
forecast, this script takes the simple average of the two forecasts and of their
95% ranges.

Inputs
    data/rates_forecast.csv  ARIMA: Fed funds, 2-, 10- and 30-year Treasury
    data/yoy_forecast.csv    ARIMA: CPI inflation
    data/pce_forecast.csv    ARIMA: PCE and core PCE inflation
    data/var_forecast.csv    VAR:   rates, CPI, core PCE, unemployment

Output
    data/combined_forecast.csv  series, date, arima, var, mean, lower, upper
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT_PATH = DATA_DIR / "combined_forecast.csv"
ARIMA_WEIGHT = 0.5  # VAR gets the other half


def load_arima() -> pd.DataFrame:
    """All ARIMA forecasts in one table: series, date, mean, lower, upper."""
    rates = pd.read_csv(DATA_DIR / "rates_forecast.csv", parse_dates=["date"])
    pce = pd.read_csv(DATA_DIR / "pce_forecast.csv", parse_dates=["date"])

    cpi = pd.read_csv(DATA_DIR / "yoy_forecast.csv", parse_dates=["date"])
    cpi = cpi.rename(columns={"mean_ci_lower": "lower", "mean_ci_upper": "upper"}).assign(series="cpi")

    columns = ["series", "date", "mean", "lower", "upper"]
    return pd.concat([rates[columns], cpi[columns], pce[columns]])


def run() -> None:
    columns = ["series", "date", "mean", "lower", "upper"]
    arima = load_arima()
    var = pd.read_csv(DATA_DIR / "var_forecast.csv", parse_dates=["date"])[columns]

    # Line the two models up: one row per series and month that both forecast.
    both = arima.merge(var, on=["series", "date"], suffixes=("_arima", "_var"))

    w = ARIMA_WEIGHT
    output = pd.DataFrame(
        {
            "series": both["series"],
            "date": both["date"],
            "arima": both["mean_arima"],
            "var": both["mean_var"],
            "mean": w * both["mean_arima"] + (1 - w) * both["mean_var"],
            "lower": w * both["lower_arima"] + (1 - w) * both["lower_var"],
            "upper": w * both["upper_arima"] + (1 - w) * both["upper_var"],
        }
    ).sort_values(["series", "date"])

    numbers = ["arima", "var", "mean", "lower", "upper"]
    output[numbers] = output[numbers].round(4)
    output.to_csv(OUTPUT_PATH, index=False, date_format="%Y-%m-%d")

    for name, rows in output.groupby("series", sort=False):
        end = rows.iloc[-1]
        print(
            f"{name:>9} {end['date']:%Y-%m}: ARIMA {end['arima']:.2f}%  VAR {end['var']:.2f}%  "
            f"-> average {end['mean']:.2f}%"
        )
    print(f"Saved {OUTPUT_PATH.name}")


if __name__ == "__main__":
    run()
