from typing import List
from dataclasses import dataclass


@dataclass
class CurveAttrs:
    coef: float
    intercept: float
    median_spend: float


@dataclass
class Curves:
    variable: str
    data: CurveAttrs


@dataclass
class SpendConstr:
    variable: str
    spend_bounds: List[float]
