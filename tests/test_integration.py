import pytest
import pandas as pd
import numpy as np
import os
from strategy_lab.engine import BacktestEngine
from strategy_lab.metrics import sharpe_ratio

reportlab = pytest.importorskip("reportlab")
from strategy_lab.report_builder import generate_pdf_report


def test_full_backtest_flow(tmp_path):
    # 1. Create synthetic data
    dates = pd.date_range(start='2020-01-01', periods=24, freq='ME')
    data = pd.DataFrame(
        np.random.normal(0.01, 0.05, (24, 3)),
        index=dates, columns=['A', 'B', 'C']
    )

    # 2. Define a strategy (Inverse Volatility)
    def inverse_vol_strategy(history):
        vol = history.std()
        inv_vol = 1 / vol
        return inv_vol / inv_vol.sum()

    # 3. Run engine
    engine = BacktestEngine(data)
    results = engine.run(inverse_vol_strategy,
                         train_window_months=12,
                         test_window_months=1)

    assert not results.empty
    assert 'Strategy' in results.columns
    assert len(results.columns) == 4  # Strategy + 3 assets

    # 4. Generate PDF
    metrics_df = pd.DataFrame({'Value': [0.5, 0.1]}, index=['Sharpe', 'Drawdown'])
    report_file = tmp_path / "test_report.pdf"
    generate_pdf_report(results, metrics_df, filename=str(report_file))
    assert os.path.exists(report_file)
    assert os.path.getsize(report_file) > 0
