"""Row types for the saved encounter study."""

from __future__ import annotations

from typing import Literal, TypedDict


class MotionInput(TypedDict):
    heading_deg: float
    target_deg: float
    rate_deg_s: float
    lower_mps: float
    upper_mps: float


class QueryInput(TypedDict):
    relative_position_m: list[float]
    a: MotionInput
    b: MotionInput
    horizon_s: float
    threshold_m: float


class CaseInput(TypedDict):
    family: str
    da_nm: int
    db_nm: int
    turn_deg: int
    scale: int
    horizon_s: int
    query: QueryInput


class ReferenceRow(TypedDict):
    case: CaseInput
    status: Literal["separated", "conflict", "unresolved"]
    reason: str
    lower_m: float
    upper_m: float
    precision_agrees: bool


class LibraryDecision(TypedDict):
    status: Literal["certified_safe", "unresolved", "unsupported"]
    reason: str
    profile_min_distance_lower_bound_m: float | None


class ComparedRow(TypedDict):
    case: CaseInput
    reference: Literal["separated", "conflict", "unresolved"]
    library: LibraryDecision
