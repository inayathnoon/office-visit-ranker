from __future__ import annotations

import pandas as pd
import pytest

from visit_ranker.config import GROUND_TRUTH, OUT_DIR, load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config(profile="demo")


@pytest.fixture(scope="session")
def truth():
    import json

    if not GROUND_TRUTH.exists():
        pytest.skip("no generated data; run `make data` first")
    return json.loads(GROUND_TRUTH.read_text())


@pytest.fixture(scope="session")
def candidates():
    path = OUT_DIR / "candidates.parquet"
    if not path.exists():
        pytest.skip("no feature frame; run `make pipeline` first")
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def scored():
    path = OUT_DIR / "scored_test.parquet"
    if not path.exists():
        pytest.skip("no scored test set; run `make pipeline` first")
    return pd.read_parquet(path)
