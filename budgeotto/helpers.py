from numpy import abs, float64
from pandas import DataFrame, Series, concat
from scipy.stats import zscore


def remove_bad_data(df: DataFrame, std_dev: int) -> DataFrame:
    # @TODO: maybe look at alternative method for removig outliers
    # current method is semi-effective
    _df = df.dropna()
    # @TODO: Creates discrepancies with the median
    # _df = _df[(abs(zscore(_df)) < std_dev).all(axis=1)]
    return _df


def get_variable_data(
    index: int, midpoint: int, std_dev: int, df: DataFrame
) -> DataFrame:
    _df = concat([df.iloc[:, index], df.iloc[:, index + midpoint]], axis=1)
    _df = remove_bad_data(_df, std_dev)
    return _df


def get_midpoint(df: DataFrame) -> int:
    return df.shape[1] // 2


def get_median_spend(col: Series) -> float64:
    return col.median()


def get_variable(df: DataFrame) -> str:
    return df.columns[0].split("_")[1]
