from typing import List

from numpy import float64


class CurveAttrs:
    coef: float64
    intercept: float64
    median_spend: float64


class Curves:
    variable: str
    data: CurveAttrs


class SpendConstr:
    variable: str
    # [Lower, Upper] bounds
    spend_bounds: List[float]
