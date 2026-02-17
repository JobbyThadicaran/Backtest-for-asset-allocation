import pandas as pd
import numpy as np
from strategy_lab.engine import BacktestEngine


def test_backtest_rolling_window():
    dates = pd.date_range(start='2020-01-01', periods=20, freq='ME')
    data = pd.DataFrame(
        np.ones((20, 2)) * 0.01,
        index=dates, columns=['A', 'B']
    )

    def strategy(hist_data):
        return pd.Series([0.5, 0.5], index=hist_data.columns)

    engine = BacktestEngine(data)
    results = engine.run(strategy, train_window_months=12,
                         test_window_months=1, window_type='rolling')

    assert len(results) == 8
    assert 'Strategy' in results.columns
    assert np.allclose(results['Strategy'], 0.01)


def test_backtest_expanding_window():
    dates = pd.date_range(start='2020-01-01', periods=20, freq='ME')
    data = pd.DataFrame(np.random.randn(20, 2), index=dates, columns=['A', 'B'])

    calls = []

    def strategy(hist_data):
        calls.append(len(hist_data))
        return pd.Series([0.5, 0.5], index=hist_data.columns)

    engine = BacktestEngine(data)
    engine.run(strategy, train_window_months=10,
               test_window_months=1, window_type='expanding')

    assert calls[0] == 10  # first call sees 10 rows
    assert calls[1] == 11  # second call sees 11 rows (expanding)
