"""Generate examples/survey.csv - a small, deliberately messy teaching dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 400

countries = ["Bangladesh", "India", "Nepal", "Pakistan", "Sri Lanka", "Bhutan", "Maldives",
             "Indonesia", "Malaysia", "Thailand", "Vietnam", "Philippines"]
group = rng.choice(["control", "treatment"], size=n)
score = rng.normal(70, 10, n) + np.where(group == "treatment", 4, 0)
score[rng.choice(n, 30, replace=False)] = np.nan

df = pd.DataFrame({
    "respondent_id": np.arange(1, n + 1),
    "age": rng.integers(18, 70, n),
    "income": np.round(rng.lognormal(10, 0.8, n), 2),  # right-skewed
    "gender": rng.choice(["Male", "Female", "male ", "Non-binary"], size=n, p=[0.45, 0.45, 0.05, 0.05]),
    "country": rng.choice(countries, size=n),
    "group": group,
    "score": np.round(score, 1),
    "hours_online": np.round(rng.gamma(2, 2, n), 1),
    "signup_date": pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
})
df["spend"] = [f"{x:,.0f}" for x in df["income"] * rng.uniform(0.05, 0.2, n)]  # numbers stored as text
df.loc[rng.choice(n, 15, replace=False), "income"] = np.nan

out = Path(__file__).with_name("survey.csv")
df.to_csv(out, index=False)
print(f"wrote {out} ({len(df)} rows)")
