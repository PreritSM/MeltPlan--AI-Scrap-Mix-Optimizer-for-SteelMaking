"""Layer-boundary contracts. Single source of truth (pydantic); every record carries provenance."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Provenance(str, Enum):
    REAL = "REAL"
    DERIVED = "DERIVED"
    ESTIMATED = "ESTIMATED"
    ASSUMED = "ASSUMED"
    MOCK = "MOCK"


class CompositionEstimate(BaseModel):
    group: str
    element: str
    mean: float
    sd: float
    ci90: tuple[float, float]
    n_heats: int
    method: str = "bayes_nnls_bootstrap"
    provenance: Provenance = Provenance.ESTIMATED


class LoadItem(BaseModel):
    grab_order: int
    mat_code: int
    mat_name: str
    group: str
    tonnes: float
    grabs: int


class Basket(BaseModel):
    basket_no: int
    target_t: float
    items: list[LoadItem]


class ElementPrediction(BaseModel):
    mean: float
    p95: float
    limit: float
    p_exceed: float


class CostBreakdown(BaseModel):
    material: float
    energy: float
    total: float
    per_t: float


class Constraint(BaseModel):
    name: str
    kind: Literal["chance", "inventory", "tap", "recipe"]
    dual: float
    dual_units: str
    plain: str


class HeatPlan(BaseModel):
    slot: int
    spec_class: str
    target_tap_t: float
    frozen: bool = False
    group_t: dict[str, float]
    baskets: list[Basket]
    predicted: dict[str, ElementPrediction]
    cost: CostBreakdown


PlanStatus = Literal["PROPOSED", "EMAILED", "SUPERVISOR_APPROVED", "RELEASED", "LOADING", "DONE", "SUPERSEDED", "REJECTED"]


class PlanVersion(BaseModel):
    plan_id: str
    version: int
    parent_version: int | None = None
    created_ts: str
    created_by: Literal["agent", "user"] = "agent"
    mode: Literal["aim", "margin", "chance"]
    alpha: float
    status: PlanStatus = "PROPOSED"
    heats: list[HeatPlan]
    binding: list[Constraint] = Field(default_factory=list)
    provenance: Provenance = Provenance.ESTIMATED
