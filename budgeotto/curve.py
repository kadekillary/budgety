from typing import Dict, Callable, Optional, Tuple

from numpy import ndarray, float64
from pandas import DataFrame
from scipy.optimize import curve_fit

from .helpers import (
    get_variable_data,
    get_midpoint,
    get_median_spend,
    get_variable,
)

from .types import CurveAttrs


class Curve:
    def __init__(
        self, min_rows: int, std_dev: int, df: DataFrame, curve_func: Callable
    ):
        self.min_rows: int = min_rows
        self.std_dev: int = std_dev
        self.df: DataFrame = df
        self.curve_func: Callable = curve_func

    def fit_curve(self, df: DataFrame) -> ndarray:
        if df.shape[0] >= self.min_rows:
            # @TODO: make sure this shit is going to work if spend and
            # revenue are swtiched positionally in df
            params = curve_fit(self.curve_func, df.iloc[:, 0], df.iloc[:, 1])[0]
        else:
            params = ndarray(None)
        return params

    def build_curve(self, df: DataFrame) -> Optional[Tuple[float64, float64]]:
        params = self.fit_curve(df)
        # @TODO: make more abstract to support various curves / params
        if params.size == 2 and params[1] > 0:
            coef = params[1]
            intercept = params[0]
            return coef, intercept
        else:
            return None, None

    def generate_curves(self) -> Dict[str, CurveAttrs]:
        curves = {}
        midpoint = get_midpoint(self.df)
        for i in range(midpoint):
            _df = get_variable_data(i, midpoint, self.std_dev, self.df)
            coef, intercept = self.build_curve(_df)
            if coef:
                dimension = get_variable(_df)
                median_spend = int(get_median_spend(_df.iloc[:, 0]))
                curves[dimension] = CurveAttrs(
                    coef=coef, intercept=intercept, median_spend=median_spend
                )
                # @TODO: how to handle bad curves? Should user remove them
                # in UI or we strip them out here???
        return curves
