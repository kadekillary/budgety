from typing import Dict, List

from pandas import DataFrame

from .curve import Curve
from .curve_opts import CurveOpts
from .types import Curves, SpendConstr


class Budgeotto:
    def __init__(self, df: DataFrame, pivot_column: str, time_agg: str):
        self.df: DataFrame = df
        self.pivot_column: str = pivot_column
        self.time_agg: str = time_agg

    def _pivot_data(self) -> DataFrame:
        pivot = self.pivot_column
        # Widen data
        pivoted_df = self.df.pivot(
            # @TODO: figure out something with this
            index=self.time_agg,
            columns=pivot,
            values=["spend", "target"],
        )
        # Remove multi-index headers
        pivoted_df.columns = pivoted_df.columns.map("{0[0]}_{0[1]}".format)
        return pivoted_df

    def get_curves(
        self, min_rows: int = 25, std_dev: int = 3, curve_func=CurveOpts.log_curve
    ) -> Curves:
        pivoted_df = self._pivot_data()
        curve = Curve(min_rows, std_dev, pivoted_df, curve_func)
        return curve.generate_curves()

    def get_dimensions(self) -> List[str]:
        df = self.df
        pivot = self.pivot_column
        return df[pivot].unique().tolist()

    def create_dimension_attr(self) -> SpendConstr:
        # Use to map various attributes to each dimension
        return {k: None for k in self.get_dimensions()}
