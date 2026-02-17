import pandas as pd
import numpy as np
from typing import Callable
from .data import validate_data, clean_data


class BacktestEngine:
    """
    Runs a walk-forward backtest on a monthly return series.

    Attributes:
        data: DataFrame with DatetimeIndex and one column per asset.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = validate_data(data, silent=True)

    def run(self,
            strategy_func: Callable[[pd.DataFrame], pd.Series],
            train_window_months: int,
            test_window_months: int,
            window_type: str = 'rolling') -> pd.DataFrame:
        """
        Args:
            strategy_func: function(history_df) → pd.Series of weights.
            train_window_months: months of history to feed the strategy.
            test_window_months: months to hold positions before rebalancing.
            window_type: 'rolling' (fixed-size window) or 'expanding'.

        Returns:
            DataFrame with a 'Strategy' column (returns) and one column
            per asset (the weights used in each period).
        """
        if window_type not in ['rolling', 'expanding']:
            raise ValueError("window_type must be 'rolling' or 'expanding'")

        data = self.data
        n_rows = len(data)

        # Detect frequency
        freq = pd.infer_freq(data.index)
        if freq is None:
            days_diff = (data.index[1] - data.index[0]).days
            if not (28 <= days_diff <= 31):
                print("Warning: Could not infer frequency. "
                      "Assuming data is Monthly.")

        train_steps = train_window_months
        test_steps = test_window_months

        strategy_returns = []
        weights_history = []
        current_step = train_steps

        while current_step + test_steps <= n_rows:
            # ---- Training window ----
            if window_type == 'rolling':
                train_start = current_step - train_steps
            else:  # expanding
                train_start = 0

            train_data = data.iloc[train_start:current_step]

            # ---- Compute weights ----
            try:
                weights = strategy_func(train_data)
            except Exception as e:
                print(f"Error in strategy at step {current_step}: {e}")
                weights = pd.Series(0, index=data.columns)

            weights = weights.reindex(data.columns, fill_value=0)

            # ---- Apply to test window ----
            test_end = min(current_step + test_steps, n_rows)
            test_returns = data.iloc[current_step:test_end]
            portfolio_return = (test_returns * weights).sum(axis=1)

            strategy_returns.append(portfolio_return)

            for _ in range(len(test_returns)):
                weights_history.append(weights)

            current_step += test_steps

        if not strategy_returns:
            return pd.DataFrame()

        full_returns = pd.concat(strategy_returns)
        full_weights = pd.DataFrame(weights_history, index=full_returns.index)
        result = pd.concat([full_returns.rename('Strategy'), full_weights], axis=1)
        return result
