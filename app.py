import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
from src.storage_duckdb import load_ohlcv, upsert_ohlcv
from src.providers.stooq import StooqProvider
from src.indicators import calculate_indicators
from src.models.moving_average import MovingAverageModel
from src.models.random_forest import RandomForestModel
from src.models.lgbm import LightGBMModel
from src.models.automfles import AutoMFLESModel
from src.advisor import get_advice
from src.plotting import plot_forecast, plot_backtest
from src.backtest import backtest_model

st.set_page_config(page_title="Stock Advisor", layout="wide")
st.title("Stock Investment Advisor")

# Sidebar
st.sidebar.header("Configuration")
symbol = st.sidebar.text_input("Symbol", "AAPL").upper()
provider_name = st.sidebar.selectbox("Provider", ["Stooq"])
start_date = st.sidebar.date_input("Start Date", date.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", date.today())
horizon = st.sidebar.number_input("Forecast Horizon (days)", value=30, min_value=1, max_value=90)
model_type = st.sidebar.selectbox("Model", ["ALL", "Moving Average", "Random Forest", "LightGBM", "AutoMFLES"])

# Fetch Data Button
if st.sidebar.button("Fetch / Update Data"):
    with st.spinner("Fetching data from Stooq..."):
        provider = StooqProvider()
        fetch_start = start_date - timedelta(days=300)
        df_new = provider.get_ohlcv(symbol, fetch_start, end_date)
        
        if not df_new.empty:
            upsert_ohlcv(df_new)
            st.sidebar.success(f"Fetched {len(df_new)} rows.")
        else:
            st.sidebar.warning("No data found.")

# Main Data Load
load_start = start_date - timedelta(days=500) # Load more for backtest
df = load_ohlcv(symbol, "stooq", start=load_start, end=end_date)

if df.empty:
    st.info("No data available in cache. Please click 'Fetch / Update Data'.")
else:
    # Indicators
    df = calculate_indicators(df)
    
    display_mask = df['date'].dt.date >= start_date
    display_df = df[display_mask]
    
    col_main, col_advisor = st.columns([3, 1])
    
    # Store forecasts in a dict: {model_name: df}
    forecasts = {}
    advice = None
    metrics_display = {}
    backtest_results = {} # Store cv_df for plotting
    
    # Run Forecast Button
    if st.button("Run Forecast"):
        models_to_run = []
        if model_type == "ALL":
            models_to_run = ["Moving Average", "Random Forest", "LightGBM", "AutoMFLES"]
        else:
            models_to_run = [model_type]
            
        # Dictionary to map name to class/params
        # Params now loaded from config inside model __init__, so we pass minimal kwargs
        model_registry = {
            "Moving Average": MovingAverageModel,
            "Random Forest": RandomForestModel,
            "LightGBM": LightGBMModel,
            "AutoMFLES": AutoMFLESModel
        }
        
        all_pred_returns = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_steps = len(models_to_run) * 2 # Train + Backtest
        current_step = 0
        
        for i, m_name in enumerate(models_to_run):
            # Training
            status_text.text(f"Training {m_name} ({i+1}/{len(models_to_run)})...")
            
            cls = model_registry[m_name]
            model = cls(horizon=horizon) # Params from config
            
            model.train(df)
            f_df = model.predict(df, horizon=horizon)
            forecasts[m_name] = f_df
            
            if not f_df.empty:
                all_pred_returns.append(f_df['predicted_return'].values)
            
            current_step += 1
            progress_bar.progress(current_step / total_steps)

        # Ensemble Advice
        if all_pred_returns:
            min_len = min(len(r) for r in all_pred_returns)
            trimmed_returns = [r[:min_len] for r in all_pred_returns]
            avg_returns = np.mean(trimmed_returns, axis=0)
            
            curr_vol = df['volatility_20'].iloc[-1]
            sma20 = df['sma_20'].iloc[-1]
            sma60 = df['sma_60'].iloc[-1]
            
            advice = get_advice(avg_returns, curr_vol, sma20, sma60)
            
        # Backtest
        for i, m_name in enumerate(models_to_run):
            status_text.text(f"Backtesting {m_name} ({i+1}/{len(models_to_run)})...")
            
            cls = model_registry[m_name]
            # Returns (metrics, cv_df)
            m_metrics, cv_df = backtest_model(cls, df, horizon=horizon, backtest_window=180)
            
            metrics_display[m_name] = m_metrics
            backtest_results[m_name] = cv_df
            
            current_step += 1
            progress_bar.progress(current_step / total_steps)
            
        status_text.empty()
        progress_bar.empty()
        
        # Save to session
        st.session_state['forecasts'] = forecasts
        st.session_state['advice'] = advice
        st.session_state['metrics_display'] = metrics_display
        st.session_state['backtest_results'] = backtest_results

    # Plotting
    with col_main:
        st.subheader(f"Price Chart: {symbol}")
        
        curr_forecasts = st.session_state.get('forecasts', {})
        curr_metrics = st.session_state.get('metrics_display', {})
        curr_backtests = st.session_state.get('backtest_results', {})
        
        fig = plot_forecast(display_df, curr_forecasts, title=f"{symbol} Forecasts")
        st.plotly_chart(fig, width='stretch')
        
        if curr_metrics:
            st.subheader("Backtest Performance (Last 180d)")
            
            # 1. Metrics Table
            metric_rows = []
            for m_name, met in curr_metrics.items():
                row = {'Model': m_name}
                row.update(met)
                metric_rows.append(row)
            
            met_df = pd.DataFrame(metric_rows)
            if not met_df.empty:
                st.dataframe(met_df.style.format({
                    'MAE': '{:.4f}',
                    'RMSE': '{:.4f}',
                }), use_container_width=True)
                
            # 2. Backtest Visualizations
            st.write("### Visual Backtest Analysis")
            # Show tabs for each model
            tabs = st.tabs(list(curr_backtests.keys()))
            for i, m_name in enumerate(curr_backtests.keys()):
                cv_df = curr_backtests[m_name]
                with tabs[i]:
                    if not cv_df.empty:
                        fig_bt = plot_backtest(cv_df, m_name)
                        st.plotly_chart(fig_bt, width='stretch')
                    else:
                        st.write("No backtest data available (insufficient history).")

    # Advisor Panel
    with col_advisor:
        st.subheader("Advisor")
        if 'advice' in st.session_state and st.session_state['advice']:
            adv = st.session_state['advice']
            
            color = "blue"
            if adv['decision'] == "BUY": color = "green"
            elif adv['decision'] == "SELL": color = "red"
            
            st.markdown(f"## :{color}[{adv['decision']}]")
            st.caption(f"Confidence: {adv['confidence']}")
            
            expl = adv['explanation']
            st.write("**Ensemble Analysis:**")
            st.write(f"- Exp. Return (30d): {expl['expected_return_30d']:.2%}")
            st.write(f"- Risk (Vol*SqT): {expl['risk_metric']:.2%}")
            st.write(f"- Score: {expl['score']:.4f}")
            st.write(f"- Trend Bearish: {expl['trend_bearish']}")
            
            st.divider()
            st.caption("Disclaimer: Not financial advice.")
            
    st.dataframe(display_df.tail())
