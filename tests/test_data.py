import pytest
import pandas as pd
import numpy as np
from strategy_lab.data import validate_data, clean_data


def test_validate_data_valid():
    dates = pd.date_range(start='2020-01-01', periods=5, freq='ME')
    df = pd.DataFrame(np.random.randn(5, 2), index=dates, columns=['A', 'B'])
    validated_df = validate_data(df)
    assert validated_df.shape == (5, 2)
    assert validated_df.index.is_monotonic_increasing


def test_validate_data_invalid_index():
    df = pd.DataFrame({'A': [1, 2, 3]}, index=['a', 'b', 'c'])
    with pytest.raises(ValueError):
        validate_data(df)


def test_clean_data_ffill():
    dates = pd.date_range(start='2020-01-01', periods=5, freq='ME')
    df = pd.DataFrame({'A': [1, np.nan, 3, np.nan, 5]}, index=dates)
    cleaned = clean_data(df, method='ffill')
    assert cleaned['A'].isnull().sum() == 0
    assert cleaned.iloc[1]['A'] == 1.0
