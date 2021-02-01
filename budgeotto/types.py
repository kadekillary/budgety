from typing import TypedDict, List

from numpy import float64


class CurveAttrs(TypedDict):
    coef: float64
    intercept: float64
    median_spend: float64


class Curves(TypedDict):
    variable: str
    data: CurveAttrs


class SpendConstr(TypedDict):
    variable: str
    # [Lower, Upper] bounds
    spend_bounds: List[float]
