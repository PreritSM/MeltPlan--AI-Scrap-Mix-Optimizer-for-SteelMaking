"""REAL READ adapter over the Kaggle MES / Level-2 / lab data (through the DATA-layer store)."""
from __future__ import annotations

from meltplan.data.store import connect


class KaggleMES:
    provenance = "REAL"

    # Fetches the basket charge records for one heat from the store.
    # Rows are ordered by timestamp and include material code, name and tonnes.
    # Returns the result as a pandas DataFrame.
    def charges(self, heatid: str):
        return connect().execute("SELECT heatid, ts, basket_no, mat_code, mat_name, t FROM heat_charge_basket WHERE heatid=? ORDER BY ts", [heatid]).df()

    # Fetches the lab chemistry measurements for one heat.
    # Reads every column of the heat_chem table for the given heat id.
    # Returns the result as a pandas DataFrame.
    def lab(self, heatid: str):
        return connect().execute("SELECT * FROM heat_chem WHERE heatid=?", [heatid]).df()

    # Fetches the energy consumption records for one heat.
    # Reads every column of the heat_energy table for the given heat id.
    # Returns the result as a pandas DataFrame.
    def energy(self, heatid: str):
        return connect().execute("SELECT * FROM heat_energy WHERE heatid=?", [heatid]).df()
