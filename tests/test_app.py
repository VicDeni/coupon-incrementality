"""The simulator runs end to end and agrees with the pipeline's own arithmetic."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from coupon_incrementality import policy

APP = Path(__file__).parents[1] / "app" / "streamlit_app.py"
DATA = APP.parent / "data" / "household_scores.csv"

pytestmark = pytest.mark.skipif(not DATA.exists(), reason="simulator data not built")


def test_app_runs_without_errors():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert len(app.metric) == 4


def test_app_reacts_to_the_controls():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    before = app.metric[0].value
    app.slider[1].set_value(0.35).run()
    assert not app.exception and app.metric[0].value != before


def test_app_formulas_match_the_pipeline():
    # The app repeats two short formulas so that it can be deployed on its own;
    # this guards against the copies drifting apart.
    scores = pd.read_csv(DATA)
    namespace = {"__file__": str(APP)}
    exec(APP.read_text().split("@st.cache_data")[0], namespace)  # pure functions only
    for name in ("profit_score", "predicted_profit"):
        ours = getattr(policy, name)(scores, 0.3, 0.75)
        theirs = namespace[name](scores, 0.3, 0.75)
        assert np.allclose(ours, theirs), name
