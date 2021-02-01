import numpy as np


class CurveOpts:
    def log_curve(x, a, b) -> np.float64:
        return b * np.log(x) + a
