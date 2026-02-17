# Strategy Lab — Backtesting Framework

A Python-based investment strategy backtesting framework built for the **Computational Investing** course. This tool allows users to test trading strategies on historical data using a walk-forward optimization engine and visualize results through an interactive dashboard.

## 🚀 Features

- **Backtesting Engine**: robust walk-forward testing with support for both **Rolling** and **Expanding** windows.
- **Data Handling**:
  - Automatic download from **Yahoo Finance**.
  - Support for **CSV uploads**.
  - Built-in data cleaning and validation (NaN handling, frequency alignment).
- **Strategy Development**:
  - Comes with a default **1/N (Equal Weight)** strategy.
  - Supports **custom strategy uploads** (Python scripts).
- **Performance Metrics**:
  - Returns: Total Return, Annualized Return.
  - Risk: Annualized Volatility, Max Drawdown.
  - Ratios: **Sharpe**, **Sortino**, **Calmar**.
  - Higher moments: Skewness, Kurtosis.
- **Interactive Dashboard**: A **Streamlit** web app to configure parameters and visualize equity curves, drawdowns, and asset allocation.
- **Reporting**: Generates professional **PDF reports** with charts and key metrics.

## 🛠 Installation

This project uses `uv` for dependency management.

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/JobbyThadicaran/Backtest-for-asset-allocation.git
    cd Backtest-for-asset-allocation
    ```

2.  **Install dependencies**:
    ```bash
    uv pip install -e ".[dev]"
    ```
    *(Note: This creates a virtual environment if one doesn't exist.)*

## 🖥 Usage

### Launch the Dashboard
Run the following command to start the interactive web application:

```bash
uv run streamlit run src/strategy_lab/app/main.py
```

### Running Tests
To verify everything is working correctly, run the test suite:

```bash
uv run pytest tests/ -v
```

## 📂 Project Structure

```
project/
├── src/
│   └── strategy_lab/
│       ├── data.py           # Data loading & cleaning
│       ├── engine.py         # Backtesting logic (Walk-forward)
│       ├── metrics.py        # Financial performance metrics
│       ├── report_builder.py # PDF generation
│       └── app/
│           └── main.py       # Streamlit Dashboard
├── tests/                    # Unit and integration tests
└── pyproject.toml            # Dependencies and metadata
```

## 📝 Custom Strategies

To test your own strategy, create a Python file (e.g., `my_strategy.py`) with a `get_weights` function:

```python
import pandas as pd

def get_weights(df: pd.DataFrame) -> pd.Series:
    """
    Args:
        df: Historical data (DataFrame of prices or returns)
    Returns:
        pd.Series: Portfolio weights summing to 1.
    """
    # Example: Inverse Volatility
    vol = df.pct_change().std()
    inv_vol = 1 / vol
    return inv_vol / inv_vol.sum()
```

Upload this file in the dashboard sidebar under **"Upload Script"**.

## 📄 License

MIT License.
