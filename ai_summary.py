"""Ask an AI model to write a short, plain-English summary of the latest numbers.

How it works
1. Collect the latest numbers the website shows (inflation, interest rates,
   jobs, GDP) and the 12-month forecasts, from the CSVs in data/.
2. If those numbers are the same as last time, stop: nothing new to say.
3. Send the numbers to an AI model through GitHub Models (free for personal
   accounts; GitHub Actions can use it with its built-in GITHUB_TOKEN, so no
   extra key is needed).
4. Check every number in the AI's answer against the data. If the AI wrote a
   number that is not in the data, ask once more; if it still fails, keep the
   old summary and publish nothing new.
5. Save the result to data/ai_summary.json, which the website reads.

If there is no token (for example, when run on your own computer), the script
says so and stops without changing anything.

Output
    data/ai_summary.json  {"generated", "model", "points": [{"title", "text"}], "facts_hash"}
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from data_utils import load_column_yoy, load_extra_yoy, load_yoy

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT_PATH = DATA_DIR / "ai_summary.json"

API_URL = "https://models.github.ai/inference/chat/completions"
MODELS = ["openai/gpt-4.1-mini", "openai/gpt-4o-mini"]  # try in this order
TIMEOUT = 60  # seconds


# ---------------------------------------------------------------------------
# 1. Collect the facts
# ---------------------------------------------------------------------------

def month(d: pd.Timestamp) -> str:
    return d.strftime("%B %Y")  # e.g. "October 2026"


def latest(series: pd.Series) -> tuple[pd.Timestamp, float]:
    s = series.dropna()
    return s.index[-1], float(s.iloc[-1])


def forecast_end(series_name: str) -> tuple[pd.Timestamp, float] | None:
    """Last month of the 12-month forecast: the ARIMA + VAR average if there is one."""
    combined = pd.read_csv(DATA_DIR / "combined_forecast.csv", parse_dates=["date"])
    rows = combined[combined["series"] == series_name].sort_values("date")
    if rows.empty:
        return None
    end = rows.iloc[-1]
    return end["date"], float(end["mean"])


def collect_facts() -> list[dict]:
    """Every number the AI may use, already rounded the way the website shows it."""
    facts: list[dict] = []

    def add(name: str, value: float, when: str, decimals: int = 1) -> None:
        facts.append({"name": name, "value": round(value, decimals), "when": when, "decimals": decimals})

    extra = DATA_DIR / "extra_series.csv"
    inflation = {
        "cpi": ("CPI inflation (prices of everything consumers buy), % vs. a year ago", load_yoy(DATA_DIR / "cpi_data.csv")),
        "core_cpi": ("Core CPI inflation (without food and energy), % vs. a year ago", load_extra_yoy(extra, ["core"])["core"]),
        "pce": ("PCE inflation (the measure the Fed's 2% goal uses), % vs. a year ago", load_column_yoy(DATA_DIR / "pce_data.csv", "pce")),
        "core_pce": ("Core PCE inflation (without food and energy), % vs. a year ago", load_column_yoy(DATA_DIR / "pce_data.csv", "core_pce")),
    }
    for key, (label, series) in inflation.items():
        d, v = latest(series)
        add(label, v, month(d))
        end = forecast_end(key)
        if end:
            add(f"Forecast: {label}", end[1], month(end[0]))

    levels = pd.read_csv(extra, parse_dates=["date"]).set_index("date").sort_index()
    rates = {
        "fed_funds": "The Fed's interest rate (federal funds rate), %",
        "gs10": "10-year U.S. Treasury yield (affects mortgage rates), %",
    }
    for key, label in rates.items():
        d, v = latest(levels[key])
        add(label, v, month(d), decimals=2)
        end = forecast_end(key)
        if end:
            add(f"Forecast: {label}", end[1], month(end[0]), decimals=2)

    d, v = latest(levels["unemployment"])
    add("Unemployment rate, %", v, month(d))
    end = forecast_end("unemployment")
    if end:
        add("Forecast: unemployment rate, %", end[1], month(end[0]))

    gdp = levels["real_gdp"].dropna()
    growth = (gdp / gdp.shift(4) - 1).dropna() * 100  # quarterly: vs. the same quarter a year ago
    if len(growth):
        d = growth.index[-1]
        add("GDP growth (size of the economy), % vs. a year ago", float(growth.iloc[-1]), f"Q{(d.month - 1) // 3 + 1} {d.year}")
    gdp_fc = pd.read_csv(DATA_DIR / "gdp_forecast.csv", parse_dates=["date"]).sort_values("date")
    if len(gdp_fc):
        end = gdp_fc.iloc[-1]
        d = end["date"]
        add("Forecast: GDP growth, % vs. a year ago", float(end["mean"]), f"Q{(d.month - 1) // 3 + 1} {d.year}")

    return facts


def facts_hash(facts: list[dict]) -> str:
    return hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 2. Ask the AI
# ---------------------------------------------------------------------------

INSTRUCTIONS = """You write a short summary of the U.S. economy for a public website.
Readers have zero background in economics.

Rules:
- Use ONLY the numbers in the data below. Copy them exactly as given. Do not
  calculate new numbers (no differences, no rounding, no percent changes).
  You may also mention the Fed's 2% goal.
- Plain, simple English. No jargon. If you must name a measure, explain it in
  a few words. Short sentences.
- Neutral and factual. No politics, no blame, no advice to buy or sell anything.
- Say "forecast" or "our models expect" for forecasts; never present a
  forecast as certain.
- Write 4 or 5 points: prices (inflation), interest rates, jobs, the economy
  (GDP), and optionally what to watch next. Each point has a title of 1-3 words
  and a text of 1-2 sentences.

Answer with JSON only, in this shape:
{"points": [{"title": "...", "text": "..."}]}
"""


def ask(model: str, facts: list[dict], token: str, feedback: str | None = None) -> dict:
    lines = [f"- {f['name']}: {f['value']:.{f['decimals']}f} ({f['when']})" for f in facts]
    user = "Data:\n" + "\n".join(lines)
    if feedback:
        user += f"\n\nYour last answer had a problem: {feedback} Please fix it."

    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        },
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"]
    return json.loads(text)


# ---------------------------------------------------------------------------
# 3. Check the answer
# ---------------------------------------------------------------------------

def check(answer: dict, facts: list[dict]) -> str | None:
    """Return a problem description, or None if the answer is fine."""
    points = answer.get("points")
    if not isinstance(points, list) or not 3 <= len(points) <= 6:
        return "It must have 3 to 6 points."

    allowed = {"2", "10", "12"}  # the Fed's 2% goal, "10-year" Treasury, "12 months"
    for f in facts:
        allowed.add(f"{f['value']:.{f['decimals']}f}")
        allowed.update(re.findall(r"\d+(?:\.\d+)?", f["when"]))  # years and quarter numbers in dates

    for p in points:
        if not isinstance(p, dict) or not str(p.get("title", "")).strip() or not str(p.get("text", "")).strip():
            return "Every point needs a title and a text."
        for number in re.findall(r"\d+(?:\.\d+)?", f"{p['title']} {p['text']}"):
            if number not in allowed:
                return f"The number {number} is not in the data. Use only the numbers given, copied exactly."
    return None


# ---------------------------------------------------------------------------

def run() -> None:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("No GITHUB_TOKEN set, so no AI summary was written (this is normal on your own computer).")
        return

    facts = collect_facts()
    fingerprint = facts_hash(facts)
    if OUTPUT_PATH.exists():
        try:
            if json.loads(OUTPUT_PATH.read_text()).get("facts_hash") == fingerprint:
                print("Numbers have not changed since the last summary; nothing to do.")
                return
        except ValueError:
            pass  # a broken old file is simply replaced

    for model in MODELS:
        feedback = None
        for attempt in range(2):  # first try, then one retry with feedback
            try:
                answer = ask(model, facts, token, feedback)
            except Exception as error:
                print(f"{model}: request failed ({error})")
                break  # try the next model
            feedback = check(answer, facts)
            if feedback is None:
                output = {
                    "generated": date.today().isoformat(),
                    "model": model,
                    "points": [{"title": p["title"].strip(), "text": p["text"].strip()} for p in answer["points"]],
                    "facts_hash": fingerprint,
                }
                OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
                print(f"Saved {OUTPUT_PATH.name} ({model}, {len(output['points'])} points)")
                return
            print(f"{model}, try {attempt + 1}: {feedback}")

    print("No summary passed the number check; the old summary (if any) was kept.")


if __name__ == "__main__":
    run()
