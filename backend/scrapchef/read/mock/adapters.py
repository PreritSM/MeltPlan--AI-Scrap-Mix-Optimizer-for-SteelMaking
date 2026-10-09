"""MOCK adapters: every record carries provenance=MOCK/ASSUMED. Placeholders awaiting real plant feeds."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from scrapchef.data.store import GROUPS, cfg
from ..base import DeliveryEvent, HandlingEvent, InventoryItem, PriceQuote, ScheduledHeat


class MockInventory:
    """Yard stock per group: expected next-day usage x 1.15 headroom (so a feasible plan exists). MOCK."""
    def __init__(self, usage_t: dict[str, float], headroom: float = 1.15):
        self.usage, self.headroom = usage_t, headroom

    def snapshot(self, ts: str) -> list[InventoryItem]:
        return [InventoryItem(group=g, tonnes=round(self.usage.get(g, 0.0) * self.headroom + 2.0, 1)) for g in GROUPS]


class AssumedPrices:
    def quotes(self, ts: str) -> list[PriceQuote]:
        return [PriceQuote(group=g, price_per_t=p) for g, p in cfg("prices")["groups"].items()]


class MockDeliveries:
    def events(self, since: str) -> list[DeliveryEvent]:
        return []


class MockSchedule:
    def __init__(self, n_heats: int, tap: float | None = None):
        self.n, self.tap = n_heats, tap or cfg("mill")["rated_tap_t"]

    def horizon(self, ts: str) -> list[ScheduledHeat]:
        t0 = datetime.fromisoformat(ts)
        return [ScheduledHeat(slot=i + 1, planned_start=(t0 + timedelta(minutes=90 * i)).isoformat(), spec_class=cfg("specs")["default"],
                              target_tap_t=self.tap) for i in range(self.n)]


class MockHandling:
    """Crane 'grab done' confirmations created by the UI; confidence is a placeholder for the vision system."""
    def __init__(self):
        self.log: list[HandlingEvent] = []

    def record(self, ts, crane, code, t, slot, basket) -> HandlingEvent:
        e = HandlingEvent(ts=ts, crane_id=crane, mat_code=code, est_tonnes=t, heat_slot=slot, basket_no=basket,
                          confidence=float(np.clip(np.random.default_rng(slot * 31 + basket).normal(0.9, 0.04), 0.7, 0.99)))
        self.log.append(e)
        return e

    def events(self, since: str) -> list[HandlingEvent]:
        return [e for e in self.log if e.ts >= since]
