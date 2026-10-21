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
    # Protocol: return yard inventory per grade group at a time.
    # Implemented by real or mock inventory adapters.
    # Returns a list of InventoryItem records.
    def snapshot(self, ts: str) -> list[InventoryItem]: ...


class PriceSource(Protocol):
    # Protocol: return price quotes per grade group at a time.
    # Implemented by real or assumed price adapters.
    # Returns a list of PriceQuote records.
    def quotes(self, ts: str) -> list[PriceQuote]: ...


class DeliverySource(Protocol):
    # Protocol: return scrap delivery events since a time.
    # Implemented by real or mock delivery adapters.
    # Returns a list of DeliveryEvent records.
    def events(self, since: str) -> list[DeliveryEvent]: ...


class ScheduleSource(Protocol):
    # Protocol: return the scheduled heats from a start time.
    # Implemented by real or mock schedule adapters.
    # Returns a list of ScheduledHeat records.
    def horizon(self, ts: str) -> list[ScheduledHeat]: ...


class HandlingEventSource(Protocol):
    # Protocol: return crane handling events since a time.
    # Implemented by real or mock handling adapters.
    # Returns a list of HandlingEvent records.
    def events(self, since: str) -> list[HandlingEvent]: ...
