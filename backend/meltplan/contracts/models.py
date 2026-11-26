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
    tap_t: float | None = None      # tap the planned charge makes (yields . x); chemistry and cost per t use it
    frozen: bool = False
    group_t: dict[str, float]
    baskets: list[Basket]
    predicted: dict[str, ElementPrediction]
    cost: CostBreakdown

    # Gives the tonnes of steel this heat is expected to tap.
    # Uses the charge's computed tap when known, else falls back to the target tap.
    # Older plans saved without tap_t therefore still report a sensible value.
    @property
    def steel_t(self) -> float:
        """Steel this heat makes: the charge's tap when known (plans saved before D19 fall back to the target)."""
        return self.tap_t if self.tap_t is not None else self.target_tap_t


PlanStatus = Literal["PROPOSED", "EMAILED", "SUPERVISOR_APPROVED", "RELEASED", "LOADING", "DONE", "SUPERSEDED", "REJECTED", "DROPPED"]


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
    hand_edited: list[int] = Field(default_factory=list)   # slots changed in the Mix Editor: no shadow prices from them on
    provenance: Provenance = Provenance.ESTIMATED
