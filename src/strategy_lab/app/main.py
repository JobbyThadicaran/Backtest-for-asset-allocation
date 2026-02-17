import streamlit as st
import pandas as pd
import plotly.express as px

from strategy_lab.data import (
    validate_data, clean_data, DataLoader,
    download_data, price_to_log_returns, resample_to_monthly
)
from strategy_lab.engine import BacktestEngine
from strategy_lab.metrics import (
    calculate_cumulative_returns, calculate_drawdown,
    sharpe_ratio, sortino_ratio, calmar_ratio,
    annualized_return, annualized_volatility,
    skewness, excess_kurtosis
)
from strategy_lab.report_builder import generate_pdf_report

st.set_page_config(page_title="Strategy Lab", layout="wide")
st.title("Strategy Lab — Backtesting Dashboard")

# ---- Sidebar: Data Source ----
st.sidebar.header("1. Data Source")
data_source = st.sidebar.radio("Choose source", ["Upload CSV", "Download (yfinance)"])

df = None

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=[0], index_col=0)
            df = validate_data(df)
            st.sidebar.success("CSV Loaded")
        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {e}")

elif data_source == "Download (yfinance)":
    tickers_input = st.sidebar.text_input("Tickers (comma separated)", "SPY, AGG, GLD")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

    if st.sidebar.button("Fetch Data"):
        if tickers_input:
            with st.spinner("Downloading data..."):
                try:
                    df = download_data(tickers_input,
                                       start_date=str(start_date),
                                       end_date=str(end_date))
                    if df.empty:
                        st.sidebar.warning("No data found for tickers.")
                    else:
                        st.sidebar.success("Data Downloaded")
                        st.session_state['data'] = df
                except Exception as e:
                    st.sidebar.error(f"Download Error: {e}")

    if data_source == "Download (yfinance)" and 'data' in st.session_state:
        df = st.session_state['data']

# ---- Sidebar: Strategy & Parameters ----
st.sidebar.header("2. Strategy Configuration")
strategy_source = st.sidebar.radio("Strategy Source", ["Default (1/N)", "Upload Script"])

strategy_func = None

if strategy_source == "Upload Script":
    uploaded_script = st.sidebar.file_uploader("Upload Strategy Script (.py)", type=['py'])
    st.sidebar.info("Script must contain a function `get_weights(df)` returning a Series.")
    if uploaded_script:
        script_content = uploaded_script.read().decode("utf-8")
        local_scope = {}
        try:
            exec(script_content, globals(), local_scope)
            if 'get_weights' in local_scope:
                strategy_func = local_scope['get_weights']
                st.sidebar.success("Strategy loaded: get_weights")
            else:
                st.sidebar.error("Function `get_weights(df)` not found in script.")
        except Exception as e:
            st.sidebar.error(f"Error loading script: {e}")
else:
    def equal_weight_strategy(df):
        n_assets = len(df.columns)
        return pd.Series(1 / n_assets, index=df.columns)

    strategy_func = equal_weight_strategy
    st.sidebar.success("Using Default 1/N Strategy")

st.sidebar.header("3. Backtest Parameters")
train_window = st.sidebar.number_input("Training Window (Months)", min_value=1, value=12)
test_window = st.sidebar.number_input("Test Window (Months)", min_value=1, value=1)
window_type = st.sidebar.selectbox("Window Type", ["rolling", "expanding"])

# ---- Main area: Run backtest & display results ----
if df is not None and strategy_func is not None:
    try:
        st.subheader("Data Preview (Prices)")
        st.dataframe(df.head())

        # Clean data
        if df.isnull().values.any():
            st.warning("Data contains NaNs. Filling with ffill.")
            df = clean_data(df, 'ffill')

        # Resample
        st.markdown("**Data Frequency Processing**")
        resample_monthly = st.checkbox("Resample to Monthly (End of Month)", value=True)
        if resample_monthly:
            df = resample_to_monthly(df)
            st.info("Data resampled to Monthly frequency.")

        # Convert to log returns
        st.markdown("**Converting Prices to Log Returns for Backtest**")
        returns_df = price_to_log_returns(df).dropna()
        st.subheader("Data Preview (Log Returns)")
        st.dataframe(returns_df.head())

        if st.button("Run Backtest"):
            with st.spinner("Running Backtest on Log Returns..."):
                engine = BacktestEngine(returns_df)
                results = engine.run(
                    strategy_func=strategy_func,
                    train_window_months=train_window,
                    test_window_months=test_window,
                    window_type=window_type
                )

                if results.empty:
                    st.error("Backtest returned no results. Check windows and data size.")
                    st.session_state['backtest_results'] = None
                else:
                    st.success("Backtest Complete!")
                    st.session_state['backtest_results'] = results

        # Show results if available
        if 'backtest_results' in st.session_state and st.session_state['backtest_results'] is not None:
            results = st.session_state['backtest_results']
            st.divider()

            col1, col2 = st.columns([2, 1])

            with col1:
                cum_ret = calculate_cumulative_returns(results['Strategy'])
                fig_cum = px.line(cum_ret, title="Cumulative Returns")
                st.plotly_chart(fig_cum, width='stretch')

                dd_df = calculate_drawdown(results['Strategy'])
                fig_dd = px.area(dd_df, x=dd_df.index, y='Drawdown', title="Drawdown")
                st.plotly_chart(fig_dd, width='stretch')

                weights = results.drop(columns=['Strategy'])
                fig_weights = px.area(weights, title="Asset Allocation")
                st.plotly_chart(fig_weights, width='stretch')

            with col2:
                sharpe = sharpe_ratio(results['Strategy'])
                sortino = sortino_ratio(results['Strategy'])
                calmar = calmar_ratio(results['Strategy'])
                total_ret = cum_ret.iloc[-1] - 1
                max_dd = dd_df['Drawdown'].min()
                ann_ret = annualized_return(results['Strategy'])
                ann_vol = annualized_volatility(results['Strategy'])
                skew = skewness(results['Strategy'])
                ex_kurt = excess_kurtosis(results['Strategy'])

                metrics_data = {
                    "Metric": [
                        "Total Return", "Annualized Return",
                        "Annualized Volatility", "Max Drawdown",
                        "Sharpe Ratio", "Sortino Ratio",
                        "Calmar Ratio", "Skewness", "Excess Kurtosis"
                    ],
                    "Value": [
                        f"{total_ret:.1%}", f"{ann_ret:.1%}",
                        f"{ann_vol:.1%}", f"{max_dd:.1%}",
                        f"{sharpe:.1f}", f"{sortino:.1f}",
                        f"{calmar:.1f}", f"{skew:.1f}", f"{ex_kurt:.1f}"
                    ]
                }
                metrics_df = pd.DataFrame(metrics_data).set_index("Metric")
                st.table(metrics_df)

            # PDF report
            st.subheader("Report Export")
            report_path = "backtest_report.pdf"
            if st.button("Generate PDF Report"):
                try:
                    generate_pdf_report(results, metrics_df, filename=report_path)
                    with open(report_path, "rb") as f:
                        pdf_data = f.read()
                    st.download_button(
                        label="Download PDF",
                        data=pdf_data,
                        file_name="strategy_report.pdf",
                        mime="application/pdf"
                    )
                    st.success("PDF Generated!")
                except Exception as e:
                    st.error(f"Error generating PDF: {e}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please select a data source, provide data, and ensure a strategy is selected to proceed.")
