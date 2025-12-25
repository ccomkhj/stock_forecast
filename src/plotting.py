import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

def plot_forecast(history_df: pd.DataFrame, forecasts: dict[str, pd.DataFrame], title: str = "Stock Price Forecast"):
    fig = go.Figure()
    
    history_df = history_df.sort_values('date')
    
    # Historical Close - WHITE COLOR
    fig.add_trace(go.Scatter(
        x=history_df['date'], 
        y=history_df['close'], 
        mode='lines', 
        name='History',
        line=dict(color='white', width=1.5)
    ))
    
    colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#FFD700']
    
    for i, (name, forecast_df) in enumerate(forecasts.items()):
        if forecast_df.empty:
            continue
            
        color = colors[i % len(colors)]
        
        last_hist = history_df.iloc[-1]
        
        forecast_x = [last_hist['date']] + forecast_df['date'].tolist()
        forecast_y = [last_hist['close']] + forecast_df['predicted_price'].tolist()
        
        # Main Line
        fig.add_trace(go.Scatter(
            x=forecast_x,
            y=forecast_y,
            mode='lines',
            name=f'{name} Forecast',
            line=dict(dash='dash', color=color)
        ))
        
        # Confidence Band
        if 'predicted_lower' in forecast_df.columns and 'predicted_upper' in forecast_df.columns:
            x_band = forecast_df['date'].tolist()
            upper_y = forecast_df['predicted_upper'].tolist()
            lower_y = forecast_df['predicted_lower'].tolist()
            
            fig.add_trace(go.Scatter(
                x=x_band,
                y=upper_y,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                name=f'{name} Upper',
                hoverinfo='skip'
            ))
            
            # Use rgba for transparency
            # Convert hex to rgb for fill
            # Simple hack: just use opacity in fillcolor
            
            fig.add_trace(go.Scatter(
                x=x_band,
                y=lower_y,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                # fillcolor is hard to map from hex list dynamically without conversion
                # Let's use a fixed semi-transparent or just rely on plotly default cycle if we didn't force color
                # But we forced color.
                # Let's simple use same color but low opacity if possible, or just gray?
                # Actually, let's skip complex color conversion and use a generic band color or try to match
                fillcolor=f"rgba(128, 128, 128, 0.2)", 
                showlegend=False,
                name=f'{name} Band',
                hoverinfo='skip'
            ))

    fig.update_layout(
        title=title, 
        xaxis_title="Date", 
        yaxis_title="Price",
        hovermode="x unified",
        template="plotly_dark" # Dark theme to make white line visible
    )
    return fig

def plot_backtest(cv_df: pd.DataFrame, model_name: str):
    """
    Plots Actual vs Predicted from Cross Validation.
    cv_df columns: ds, cutoff, y, <model_name>
    """
    if cv_df.empty:
        return go.Figure()
        
    fig = go.Figure()
    
    # Identify model col
    reserved = ['unique_id', 'ds', 'cutoff', 'y']
    model_col = [c for c in cv_df.columns if c not in reserved][0]
    
    # Plot Actuals (y) - Aggregated?
    # Actuals are the same for all cutoffs for the same date?
    # Yes, 'y' is the ground truth. We can plot it as one line.
    # Group by ds and take first (since y is same)
    actuals = cv_df.groupby('ds')['y'].first().reset_index().sort_values('ds')
    
    fig.add_trace(go.Scatter(
        x=actuals['ds'],
        y=actuals['y'],
        mode='lines',
        name='Actual',
        line=dict(color='white', width=2)
    ))
    
    # Plot Predictions
    # Each cutoff produces a forecast trace.
    # Plotting every single cutoff might be messy if n_windows is large.
    # But usually it's ~5-6 windows.
    # Let's plot them as separate segments.
    
    cutoffs = cv_df['cutoff'].unique()
    
    for cutoff in cutoffs:
        subset = cv_df[cv_df['cutoff'] == cutoff].sort_values('ds')
        fig.add_trace(go.Scatter(
            x=subset['ds'],
            y=subset[model_col],
            mode='lines',
            name=f'Pred (Cutoff {pd.to_datetime(cutoff).date()})',
            line=dict(dash='dot', width=1),
            opacity=0.8
        ))
        
    fig.update_layout(
        title=f"Backtest Analysis: {model_name}",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark"
    )
    
    return fig
