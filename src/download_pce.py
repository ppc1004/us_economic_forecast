"""Download PCE prices and spending by category from the BEA.

The PCE price index (the Fed's preferred inflation gauge) and its components
come from BEA's monthly NIPA flat file:

    https://apps.bea.gov/national/Release/TXT/NipaDataM.txt

Every category in data/pce_tree.csv is stored twice: its chain-type price
index (BEA code suffix "RG", NIPA table 2.4.4U) and its nominal spending
(suffix "RC", table 2.4.5U). Spending shares are the PCE weights: unlike CPI
weights they change every month.

Outputs
    data/pce_data.csv        headline and core PCE price index (date, pce, core_pce)
    data/pce_components.csv  price index per category, monthly since 1979
    data/pce_weights.csv     latest nominal spending per category and its share of total PCE

If BEA cannot be reached, headline and core PCE fall back to FRED (PCEPI,
PCEPILFE) and the component files are left unchanged.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TREE_PATH = DATA_DIR / "pce_tree.csv"
HEADLINE_PATH = DATA_DIR / "pce_data.csv"
COMPONENTS_PATH = DATA_DIR / "pce_components.csv"
WEIGHTS_PATH = DATA_DIR / "pce_weights.csv"

BEA_URL = "https://apps.bea.gov/national/Release/TXT/NipaDataM.txt"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
START = "1979-01-01"
CORE = "DPCC"  # PCE excluding food and energy
HEADERS = {"User-Agent": "us_economic_forecast research dashboard (github.com/ppc1004/us_economic_forecast)"}


def parse_nipa(text: str, codes: set[str]) -> pd.DataFrame:
    """Long NIPA rows (%SeriesCode, Period, Value) -> wide frame of the requested codes."""
    df = pd.read_csv(io.StringIO(text), dtype=str)
    df.columns = ["code", "period", "value"]
    df = df[df["code"].isin(codes) & df["period"].str.match(r"^\d{4}M\d{2}$")]
    df = df.assign(
        date=pd.to_datetime(df["period"].str[:4] + "-" + df["period"].str[5:] + "-01"),
        value=pd.to_numeric(df["value"].str.replace(",", "", regex=False), errors="coerce"),
    )
    wide = df.pivot_table(index="date", columns="code", values="value", aggfunc="last").sort_index()
    wide.index.name = "date"
    return wide


def spending_weights(tree: pd.DataFrame, nominal: pd.DataFrame) -> pd.DataFrame:
    """Latest nominal spending per category and its share of total PCE (percent)."""
    last = nominal.dropna(how="all").index.max()
    row = nominal.loc[last]
    total = row["DPCERC"]
    out = tree.copy()
    out["spending_musd"] = [row.get(c + "RC") for c in tree["code"]]
    out["weight"] = out["spending_musd"] / total * 100
    out["date"] = last.strftime("%Y-%m-%d")
    return out


def fred_headline() -> pd.DataFrame:
    frames = {}
    for column, series_id in {"pce": "PCEPI", "core_pce": "PCEPILFE"}.items():
        data = pd.read_csv(FRED_CSV_URL.format(series_id))
        data.columns = ["date", column]
        data["date"] = pd.to_datetime(data["date"])
        frames[column] = pd.to_numeric(data.set_index("date")[column], errors="coerce")
    return pd.DataFrame(frames).dropna(how="all")


def download_pce() -> None:
    tree = pd.read_csv(TREE_PATH)
    price_codes = {c + "RG" for c in tree["code"]} | {CORE + "RG"}
    spend_codes = {c + "RC" for c in tree["code"]}

    try:
        response = requests.get(BEA_URL, headers=HEADERS, timeout=180)
        response.raise_for_status()
        wide = parse_nipa(response.text, price_codes | spend_codes)
    except Exception as error:
        print(f"BEA download failed ({error}); using FRED for headline and core PCE only.")
        fred_headline().round(3).to_csv(HEADLINE_PATH, date_format="%Y-%m-%d")
        return

    if "DPCERG" not in wide or "DPCERC" not in wide:
        raise SystemExit("BEA file is missing total PCE; keeping the previous files.")

    headline = pd.DataFrame({"pce": wide["DPCERG"], "core_pce": wide.get(CORE + "RG")}).dropna(how="all")
    headline.round(3).to_csv(HEADLINE_PATH, date_format="%Y-%m-%d")

    prices = wide[[c for c in wide.columns if c.endswith("RG") and c != CORE + "RG"]]
    prices = prices.rename(columns=lambda c: c[:-2])
    prices = prices[[c for c in tree["code"] if c in prices.columns]]
    prices = prices[prices.index >= START]
    prices.round(3).to_csv(COMPONENTS_PATH, date_format="%Y-%m-%d")

    nominal = wide[[c for c in wide.columns if c.endswith("RC")]]
    weights = spending_weights(tree, nominal)
    weights.round({"spending_musd": 0, "weight": 4}).to_csv(WEIGHTS_PATH, index=False)

    missing = sorted(set(tree["code"]) - set(prices.columns))
    if missing:
        print(f"  no price index for: {', '.join(missing)}")
    print(f"Saved PCE through {headline.index.max():%Y-%m}: {len(prices.columns)} categories")


if __name__ == "__main__":
    download_pce()
