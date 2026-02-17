import pytest
import pandas as pd
import numpy as np
from strategy_lab.metrics import sharpe_ratio, sortino_ratio, calculate_drawdown


def test_sharpe_ratio():
    returns = pd.Series([0.01, 0.01, 0.01, 0.01])
    # Std dev is 0 → Sharpe 0
    assert sharpe_ratio(returns) == 0.0

    returns_2 = pd.Series([0.1, -0.1, 0.1, -0.1])
    # Mean 0 → Sharpe 0
    assert sharpe_ratio(returns_2) == 0.0


def test_drawdown():
    returns = pd.Series([0.1, -0.1, 0.212121])
    dd_df = calculate_drawdown(returns)
    drawdown = dd_df['Drawdown']
    assert np.isclose(drawdown.iloc[0], 0.0)
    assert np.isclose(drawdown.iloc[1], 1 / np.exp(0.1) - 1)
    assert np.isclose(drawdown.iloc[2], 0.0)
