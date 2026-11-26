"""MOCK adapters: every record carries provenance=MOCK/ASSUMED. Placeholders awaiting real plant feeds."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from meltplan.data.store import GROUPS, cfg
from ..base import DeliveryEvent, HandlingEvent, InventoryItem, PriceQuote, ScheduledHeat


class MockInventory:
    """Yard stock per group: expected next-day usage x 1.15 headroom (so a feasible plan exists). MOCK."""
    # Creates a mock yard inventory from expected usage per group.
    # headroom scales usage so a feasible plan always exists.
    # Values are MOCK placeholders until a real feed is available.
    def __init__(self, usage_t: dict[str, float], headroom: float = 1.15):
        self.usage, self.headroom = usage_t, headroom

    # Returns the mock yard stock for every grade group.
    # Each group gets usage times headroom plus a small 2 t buffer.
    # The timestamp is accepted for interface compatibility only.
    def snapshot(self, ts: str) -> list[InventoryItem]:
        return [InventoryItem(group=g, tonnes=round(self.usage.get(g, 0.0) * self.headroom + 2.0, 1)) for g in GROUPS]


class AssumedPrices:
    # Returns the assumed price quote for each grade group.
    # Prices come from the prices config file.
    # The timestamp is accepted for interface compatibility only.
    def quotes(self, ts: str) -> list[PriceQuote]:
        return [PriceQuote(group=g, price_per_t=p) for g, p in cfg("prices")["groups"].items()]


class MockDeliveries:
    # Returns delivery events since a given time.
    # The mock has no deliveries, so the list is always empty.
    # Placeholder for a real delivery feed.
    def events(self, since: str) -> list[DeliveryEvent]:
        return []


class MockSchedule:
    # Creates a mock heat schedule with a fixed number of heats.
    # tap defaults to the mill's configured target tap tonnage.
    # Values are MOCK placeholders until a real feed is available.
    def __init__(self, n_heats: int, tap: float | None = None):
        self.n, self.tap = n_heats, tap or cfg("mill")["target_tap_t"]

    # Generates the scheduled heats starting from a given time.
    # Heats are spaced 90 minutes apart with the default spec class.
    # Returns a list of ScheduledHeat records.
    def horizon(self, ts: str) -> list[ScheduledHeat]:
        t0 = datetime.fromisoformat(ts)
        return [ScheduledHeat(slot=i + 1, planned_start=(t0 + timedelta(minutes=90 * i)).isoformat(), spec_class=cfg("specs")["default"],
                              target_tap_t=self.tap) for i in range(self.n)]


class MockHandling:
    """Crane 'grab done' confirmations created by the UI; confidence is a placeholder for the vision system."""
    # Creates an empty log of crane handling events.
    # Events are added when the UI confirms a grab.
    # Stands in for a future vision-based handling system.
    def __init__(self):
        self.log: list[HandlingEvent] = []

    # Records a crane 'grab done' confirmation as a handling event.
    # Assigns a deterministic placeholder confidence based on slot and basket.
    # Appends the event to the log and returns it.
    def record(self, ts, crane, code, t, slot, basket, plan_id=None) -> HandlingEvent:
        e = HandlingEvent(ts=ts, plan_id=plan_id, crane_id=crane, mat_code=code, est_tonnes=t, heat_slot=slot, basket_no=basket,
                          confidence=float(np.clip(np.random.default_rng(slot * 31 + basket).normal(0.9, 0.04), 0.7, 0.99)))
        self.log.append(e)
        return e

    # Looks up a logged handling event by plan, slot, basket and material code.
    # Matches all four fields exactly.
    # Returns the event, or None if it was never recorded.
    def find(self, plan_id, slot, basket, code) -> HandlingEvent | None:
        return next((e for e in self.log if (e.plan_id, e.heat_slot, e.basket_no, e.mat_code) == (plan_id, slot, basket, code)), None)

    # Returns handling events recorded at or after a given time.
    # Filters the in-memory log by timestamp.
    # Implements the HandlingEventSource protocol.
    def events(self, since: str) -> list[HandlingEvent]:
        return [e for e in self.log if e.ts >= since]
