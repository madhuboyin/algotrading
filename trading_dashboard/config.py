# config.py - Clean Configuration without STOCK_SYMBOLS
"""
Configuration file for the Algorithmic Trading Dashboard
Trading parameters only - tickers managed in separate tickers.py file
"""

# Trading Parameters
TRADING_CONFIG = {
    # Data settings
    'DATA_PERIOD': '2y',  # Historical data period
    
    # Target returns
    'TARGET_RETURN': 0.275,  # 27.5% target return
    'PARTIAL_TARGETS': {
        'TARGET_1': 0.15,    # 15% - First partial exit
        'TARGET_2': 0.22,    # 22% - Second partial exit  
        'TARGET_3': 0.275    # 27.5% - Final exit
    },
    
    # Stop loss settings
    'ATR_MULTIPLIER': 2.0,           # ATR stop loss multiplier
    'PCT_STOP_CONSERVATIVE': 0.08,    # 8% conservative stop
    'PCT_STOP_AGGRESSIVE': 0.12,      # 12% aggressive stop
    'MIN_STOP_LOSS_PCT': 0.05,       # Minimum 5% stop loss
    'SUPPORT_STOP_BUFFER': 0.03,     # 3% below support
    'EMA50_STOP_BUFFER': 0.02,       # 2% below EMA 50
    'BB_STOP_BUFFER': 0.02,          # 2% below BB lower band
    
    # Signal scoring (Updated for stricter criteria)
    'MIN_ENTRY_SCORE': 8,            # Increased from 5 to 8 for stricter entry
    'MAX_SIGNAL_SCORE': 20,          # Maximum possible score - strictly enforced
    
    # Score component limits (to prevent exceeding max score)
    'MAX_BASE_SCORE': 12,            # Maximum base signals score
    'MAX_TREND_SCORE': 5,            # Maximum trend score
    'MAX_MOMENTUM_SCORE': 7,         # Maximum momentum score  
    'MAX_VOLUME_SCORE': 4,           # Maximum volume score
    'MIN_VALUATION_PENALTY': -7,     # Most negative valuation penalty
    
    # Valuation thresholds
    'MAX_DISTANCE_SMA200': 20,       # Max % above SMA 200 before penalty
    'MAX_52W_POSITION': 85,          # Max position in 52W range before penalty
    'HIGH_VOLATILITY_THRESHOLD': 40, # Volatility threshold for penalty
    
    # Risk management
    'MAX_PORTFOLIO_RISK': 0.02,      # 2% max risk per trade
    'TARGET_RISK_REWARD': 2.0,       # Minimum 1:2 risk/reward ratio
    
    # Technical indicator parameters
    'RSI_PERIOD': 14,
    'MACD_FAST': 12,
    'MACD_SLOW': 26,
    'MACD_SIGNAL': 9,
    'ADX_PERIOD': 14,
    'SMA_LONG': 200,
    'EMA_MEDIUM': 50,
    'SMA_SHORT': 20,
    'BB_PERIOD': 20,
    'BB_STD': 2,
    'STOCH_K': 14,
    'STOCH_D': 3,
    'VOLUME_SMA': 20,
    'ATR_PERIOD': 14,
    
    # Analysis parameters
    'RECENT_DAYS': 5,                # Days for recent trend analysis
    'SUPPORT_RESISTANCE_LOOKBACK': 20, # Days for S&R calculation
    'MOMENTUM_THRESHOLD': 0.02,      # 2% momentum threshold
    
    # Holding period
    'TARGET_HOLDING_PERIOD': '3-4 months',
    'REVIEW_FREQUENCY': 'Weekly review recommended'
}

# Signal weights and thresholds
SIGNAL_WEIGHTS = {
    'MACD_BULLISH_CROSSOVER': 2,
    'MACD_ABOVE_SIGNAL': 1,
    'RSI_OVERSOLD_RECOVERY': 2,
    'RSI_BULLISH_ZONE': 1,
    'STRONG_UPTREND_ADX': 2,
    'MODERATE_UPTREND_ADX': 1,
    'MA_GOLDEN_ALIGNMENT': 2,
    'ABOVE_EMA_50': 1,
    'ABOVE_SMA_200': 1,
    'NEAR_BB_LOWER': 1,
    'BB_MIDDLE_ZONE': 1,
    'STOCHASTIC_BULLISH': 1,
    'ABOVE_AVG_VOLUME': 1,
    'POSITIVE_MOMENTUM': 1
}

# RSI thresholds
RSI_THRESHOLDS = {
    'OVERSOLD_MIN': 30,
    'OVERSOLD_MAX': 50,
    'BULLISH_MIN': 50,
    'BULLISH_MAX': 70,
    'OVERBOUGHT': 80
}

# ADX thresholds
ADX_THRESHOLDS = {
    'STRONG_TREND': 25,
    'MODERATE_TREND': 20
}

# Bollinger Band position thresholds
BB_THRESHOLDS = {
    'LOWER_ZONE': 0.2,     # Below 20% = near lower band
    'MIDDLE_MIN': 0.4,     # 40-60% = middle zone
    'MIDDLE_MAX': 0.6
}

# Flask configuration
FLASK_CONFIG = {
    'DEBUG': True,
    'HOST': '0.0.0.0',
    'PORT': 5001
}