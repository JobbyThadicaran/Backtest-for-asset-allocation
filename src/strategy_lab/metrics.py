import numpy as np
import pandas as pd


def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """Cumulative wealth from log returns: exp(cumsum(log_returns))."""
    return np.exp(returns.cumsum())


def calculate_drawdown(returns: pd.Series) -> pd.DataFrame:
    """Calculates wealth, peaks, and drawdown from log returns."""
    wealth = np.exp(returns.cumsum())
    peaks = wealth.cummax()
    drawdown = (wealth - peaks) / peaks
    return pd.DataFrame({'Wealth': wealth, 'Peaks': peaks, 'Drawdown': drawdown})


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0,
                 periods_per_year: int = 12) -> float:
    """Annualized Sharpe Ratio."""
    excess_returns = (returns - risk_free_rate) / periods_per_year
    if excess_returns.std() == 0:
        return 0.0
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0,
                  periods_per_year: int = 12) -> float:
    """Annualized Sortino Ratio (penalises only downside volatility)."""
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_returns.std()


def calmar_ratio(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Calmar Ratio: annualised return / max drawdown."""
    dd = calculate_drawdown(returns)
    max_drawdown = dd['Drawdown'].min()
    if max_drawdown == 0:
        return 0.0
    ann_ret = annualized_return(returns, periods_per_year)
    return ann_ret / abs(max_drawdown)


def annualized_return(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Annualised return from log returns."""
    if len(returns) == 0:
        return 0.0
    return np.exp(returns.mean() * periods_per_year) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Annualised volatility."""
    return returns.std() * np.sqrt(periods_per_year)


def skewness(returns: pd.Series) -> float:
    """Skewness of the return distribution."""
    return returns.skew()


def excess_kurtosis(returns: pd.Series) -> float:
    """Excess kurtosis (normal distribution = 0)."""
    return returns.kurtosis()
