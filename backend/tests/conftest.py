import pytest

import scrapchef.act.plans as plans


@pytest.fixture(autouse=True, scope="session")
def _isolated_app_db(tmp_path_factory):
    """Tests must not wipe the operational store of a running API (`data/app.sqlite`), nor call a live LLM."""
    root = tmp_path_factory.mktemp("app")
    (root / "data").mkdir()
    mp = pytest.MonkeyPatch()
    mp.setattr(plans, "ROOT", root)
    mp.setenv("SCRAPCHEF_LLM", "off")   # never call a live LLM from tests, even when backend/.env holds a key
    yield
    mp.undo()
