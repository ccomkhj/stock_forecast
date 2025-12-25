import numpy as np

def get_advice(predicted_returns, current_volatility, sma20, sma60, lambda_param=0.5):
    """
    Generates Buy/Hold/Sell advice based on predicted returns and risk.
    """
    # 1. Expected compounded return
    r_30 = np.prod([1 + r for r in predicted_returns]) - 1
    
    # 2. Risk penalty
    # Assumes horizon is len(predicted_returns) usually 30
    horizon = len(predicted_returns)
    risk = current_volatility * np.sqrt(horizon)
    
    # 3. Score
    score = r_30 - (lambda_param * risk)
    
    # 4. Trend gate
    trend_bearish = False
    if sma20 is not None and sma60 is not None and sma20 < sma60:
        trend_bearish = True
        # Reduce score or apply penalty
        # Applying a fixed penalty makes it harder to reach BUY threshold
        score -= 0.02 
        
    # 5. Thresholds
    decision = "HOLD"
    # BUY if score > 0.03 AND not bearish (trend gate)
    if score > 0.03 and not trend_bearish:
        decision = "BUY"
    elif score < -0.03:
        decision = "SELL"
    
    # 6. Confidence
    abs_score = abs(score)
    confidence = "Low"
    
    # Threshold for "extreme volatility" - arbitrary but let's say 3% daily
    if abs_score > 0.06 and current_volatility < 0.03:
        confidence = "High"
    elif abs_score >= 0.03:
        confidence = "Medium"
        
    explanation = {
        'expected_return_30d': r_30,
        'risk_metric': risk,
        'score': score,
        'trend_bearish': trend_bearish,
        'volatility': current_volatility
    }
        
    return {
        'decision': decision,
        'confidence': confidence,
        'explanation': explanation
    }
