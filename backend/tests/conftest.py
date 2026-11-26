import os

import pytest

import meltplan.act.plans as plans

# set before any test module imports meltplan.api.main, whose module-level World would otherwise open data/app.sqlite
os.environ["MELTPLAN_APP_DB"] = ":memory:"


# Session fixture isolating tests from the operational environment.
# Points the plan store's ROOT at a temp directory so data/app.sqlite is never touched.
# Disables the LLM so no test calls a live model.
@pytest.fixture(autouse=True, scope="session")
def _isolated_app_db(tmp_path_factory):
    """Tests must not wipe the operational store of a running API (`data/app.sqlite`), nor call a live LLM."""
    root = tmp_path_factory.mktemp("app")
    (root / "data").mkdir()
    mp = pytest.MonkeyPatch()
    mp.setattr(plans, "ROOT", root)
    mp.setenv("MELTPLAN_LLM", "off")   # never call a live LLM from tests, even when backend/.env holds a key
    yield
    mp.undo()
