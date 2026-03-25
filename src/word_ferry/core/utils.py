import numpy as np


def is_trend_significant(values: list[float], positive: bool, threshold: float) -> bool:
    """
    对values做线性回归，判断斜率是否在给定方向上显著。

    positive=True: 检验斜率是否显著为正（val上升）
    positive=False: 检验斜率是否显著为负（train下降）
    """
    n = len(values)
    x = np.arange(n, dtype=np.float64)
    y = np.array(values, dtype=np.float64)

    x_mean = x.mean()
    y_mean = y.mean()

    ss_xx = ((x - x_mean) ** 2).sum()
    ss_xy = ((x - x_mean) * (y - y_mean)).sum()

    slope = ss_xy / ss_xx

    # 方向都不对，直接返回
    if positive and slope <= 0:
        return False
    if not positive and slope >= 0:
        return False

    # 残差标准差
    y_pred = y_mean + slope * (x - x_mean)
    residuals = y - y_pred
    residual_std = np.sqrt((residuals ** 2).sum() / (n - 2))

    # 斜率的标准误
    slope_se = residual_std / np.sqrt(ss_xx)

    if slope_se == 0:
        return True  # 完美线性，趋势确定

    t_stat = abs(slope) / slope_se
    return t_stat > threshold
