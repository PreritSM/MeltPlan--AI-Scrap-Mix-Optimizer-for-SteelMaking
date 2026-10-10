"""READ-layer adapter protocols. KaggleMES is REAL; everything else ships as a labelled MOCK until the plant supplies a feed."""
from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from scrapchef.contracts.models import Provenance


class InventoryItem(BaseModel):
    group: str
    tonnes: float
    provenance: Provenance = Provenance.MOCK


class PriceQuote(BaseModel):
    group: str
    price_per_t: float
    currency: str = "USD"
    provenance: Provenance = Provenance.ASSUMED


class DeliveryEvent(BaseModel):
    ts: str
    group: str
    tonnes: float
    status: str
    provenance: Provenance = Provenance.MOCK


class ScheduledHeat(BaseModel):
    slot: int
    planned_start: str
    spec_class: str
    target_tap_t: float
    provenance: Provenance = Provenance.MOCK


class HandlingEvent(BaseModel):
    ts: str
    plan_id: str | None = None
    crane_id: str
    mat_code: int
    est_tonnes: float
    heat_slot: int
    basket_no: int
    confidence: float
    provenance: Provenance = Provenance.MOCK


class InventorySource(Protocol):
    def snapshot(self, ts: str) -> list[InventoryItem]: ...


class PriceSource(Protocol):
    def quotes(self, ts: str) -> list[PriceQuote]: ...


class DeliverySource(Protocol):
    def events(self, since: str) -> list[DeliveryEvent]: ...


class ScheduleSource(Protocol):
    def horizon(self, ts: str) -> list[ScheduledHeat]: ...


class HandlingEventSource(Protocol):
    def events(self, since: str) -> list[HandlingEvent]: ...
