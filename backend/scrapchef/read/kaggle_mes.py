"""REAL READ adapter over the Kaggle MES / Level-2 / lab data (through the DATA-layer store)."""
from __future__ import annotations

from scrapchef.data.store import connect


class KaggleMES:
    provenance = "REAL"

    def charges(self, heatid: str):
        return connect().execute("SELECT heatid, ts, basket_no, mat_code, mat_name, t FROM heat_charge_basket WHERE heatid=? ORDER BY ts", [heatid]).df()

    def lab(self, heatid: str):
        return connect().execute("SELECT * FROM heat_chem WHERE heatid=?", [heatid]).df()

    def energy(self, heatid: str):
        return connect().execute("SELECT * FROM heat_energy WHERE heatid=?", [heatid]).df()
