import numpy as np
import pandas as pd
import pytest

from learnstat import Session


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 200
    out = pd.DataFrame({
        "income": rng.lognormal(10, 0.9, n),
        "age": rng.integers(18, 70, n),
        "country": rng.choice([f"C{i}" for i in range(12)], n),
        "group": rng.choice(["a", "b"], n),
        "score": rng.normal(50, 10, n),
        "gender": rng.choice(["Male", "Female", "male "], n),
        "spend": [f"{x:,.0f}" for x in rng.uniform(1000, 5000, n)],
    })
    out.loc[:29, "score"] = np.nan
    return out


@pytest.fixture
def session(tmp_path, df):
    s = Session(memory_dir=tmp_path / "mem", ai=False, verbose=False, out=lambda *_: None)
    s.ns["df"] = s.track(df, name="survey")
    return s
