import pandas as pd
import numpy as np


def validate_data(df: pd.DataFrame, silent: bool = False) -> pd.DataFrame:
    """
    Checks the validity of the input DataFrame for backtesting.

    Checks:
    1. Index is a DatetimeIndex and is monotonic increasing.
    2. Checks for NaN values and reports them.
    """
    # Make sure the index contains dates
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            raise ValueError(f"Index could not be converted to DatetimeIndex: {e}")

    # Sort by date if not already sorted
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    # Warn about missing values
    if not silent and df.isnull().values.any():
        print("Warning: Input data contains NaN values. Use clean_data to handle them.")

    return df


def clean_data(df: pd.DataFrame, method='ffill') -> pd.DataFrame:
    """
    Cleans the data by handling missing values.

    Args:
        method: 'ffill' (forward-fill), 'bfill' (backward-fill),
                or 'fill_zero'.
    """
    if method == 'ffill':
        return df.ffill()
    elif method == 'bfill':
        return df.bfill()
    elif method == 'fill_zero':
        return df.fillna(0.0)
    else:
        raise ValueError(f"Unknown cleaning method: {method}")


class DataLoader:
    """Helper class to load data from CSV."""

    @staticmethod
    def load_csv(filepath: str, date_col: str = 'Date') -> pd.DataFrame:
        df = pd.read_csv(filepath, parse_dates=[date_col], index_col=date_col)
        return validate_data(df, silent=True)


def download_data(tickers: list[str], start_date: str, end_date: str = None) -> pd.DataFrame:
    """Downloads historical price data using yfinance."""
    import yfinance as yf

    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(',')]

    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        group_by='column'
    )

    if len(tickers) == 1:
        if 'Close' in data.columns:
            data = data[['Close']]
            data.columns = tickers
    else:
        if 'Close' in data.columns:
            data = data['Close']

    return data


def price_to_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Converts prices to log returns: ln(P_t / P_{t-1})."""
    df = df.astype(float)
    return np.log(df / df.shift(1))


def resample_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Resamples daily data to monthly (last price of each month)."""
    return df.resample('ME').last()
