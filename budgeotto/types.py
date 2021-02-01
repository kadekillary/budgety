from typing import List
from collections import namedtuple

from numpy import float64


CurveAttrs = namedtuple("CurveAttrs", ["coef", "intercept", "median_spend"])

#  class CurveAttrs(TypedDict):
#  coef: float64
#  intercept: float64
#  median_spend: float64


Curves = namedtuple("Curves", ["variable", "data"])


#  class Curves(TypedDict):
#  variable: str
#  data: CurveAttrs


SpendConstr = namedtuple("SpendConstr", ["variable", "spend_bounds"])


#  class SpendConstr(TypedDict):
#  variable: str
#  # [Lower, Upper] bounds
#  spend_bounds: List[float]
