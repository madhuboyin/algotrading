# indicators_config.py - Configuration for selectable technical indicators

# Available indicators grouped by category
INDICATOR_CATEGORIES = {
    'Trend Indicators': {
        'SMA_20': {
            'name': 'Simple Moving Average (20)',
            'description': 'Short-term trend indicator',
            'default': True,
            'required': False
        },
        'SMA_50': {
            'name': 'Simple Moving Average (50)',
            'description': 'Medium-term trend indicator',
            'default': True,
            'required': False
        },
        'SMA_100': {
            'name': 'Simple Moving Average (100)',
            'description': 'Long-term trend indicator',
            'default': True,
            'required': False
        },
        'SMA_200': {
            'name': 'Simple Moving Average (200)',
            'description': 'Major trend indicator',
            'default': True,
            'required': True  # Required for trend analysis
        },
        'EMA_20': {
            'name': 'Exponential Moving Average (20)',
            'description': 'Fast-responding trend indicator',
            'default': True,
            'required': False
        },
        'EMA_50': {
            'name': 'Exponential Moving Average (50)',
            'description': 'Medium-term exponential trend',
            'default': True,
            'required': True  # Required for trend analysis
        },
        'ADX': {
            'name': 'Average Directional Index',
            'description': 'Trend strength indicator',
            'default': True,
            'required': False
        }
    },
    'Momentum Indicators': {
        'RSI': {
            'name': 'Relative Strength Index',
            'description': 'Overbought/oversold momentum',
            'default': True,
            'required': True  # Core momentum indicator
        },
        'MACD': {
            'name': 'MACD',
            'description': 'Moving average convergence divergence',
            'default': True,
            'required': True  # Core momentum indicator
        },
        'Stochastic': {
            'name': 'Stochastic Oscillator',
            'description': 'Momentum oscillator',
            'default': True,
            'required': False
        },
        'Williams_R': {
            'name': 'Williams %R',
            'description': 'Momentum oscillator',
            'default': True,
            'required': False
        },
        'ROC': {
            'name': 'Rate of Change',
            'description': 'Price momentum indicator',
            'default': False,
            'required': False
        },
        'CCI': {
            'name': 'Commodity Channel Index',
            'description': 'Momentum indicator',
            'default': False,
            'required': False
        },
        'UO': {
            'name': 'Ultimate Oscillator',
            'description': 'Multi-timeframe momentum',
            'default': False,
            'required': False
        },
        'TRIX': {
            'name': 'TRIX',
            'description': 'Triple exponential momentum',
            'default': False,
            'required': False
        }
    },
    'Volume Indicators': {
        'Volume_SMA': {
            'name': 'Volume Moving Average',
            'description': 'Average volume trend',
            'default': True,
            'required': True  # Required for volume analysis
        },
        'MFI': {
            'name': 'Money Flow Index',
            'description': 'Volume-weighted momentum',
            'default': True,
            'required': False
        },
        'OBV': {
            'name': 'On-Balance Volume',
            'description': 'Volume accumulation indicator',
            'default': True,
            'required': False
        },
        'VWAP': {
            'name': 'Volume Weighted Average Price',
            'description': 'Volume-weighted fair value',
            'default': False,
            'required': False
        }
    },
    'Volatility Indicators': {
        'BB': {
            'name': 'Bollinger Bands',
            'description': 'Price volatility bands',
            'default': True,
            'required': False
        },
        'ATR': {
            'name': 'Average True Range',
            'description': 'Volatility measure',
            'default': True,
            'required': False
        },
        'PC': {
            'name': 'Price Channels',
            'description': 'Donchian channels',
            'default': False,
            'required': False
        }
    }
}

# Default indicator selection (all required + some defaults)
DEFAULT_INDICATORS = {}
for category, indicators in INDICATOR_CATEGORIES.items():
    for indicator_id, config in indicators.items():
        if config['required'] or config['default']:
            DEFAULT_INDICATORS[indicator_id] = True

def get_available_indicators():
    """Get all available indicators with their configurations"""
    return INDICATOR_CATEGORIES

def get_default_indicators():
    """Get default indicator selection"""
    return DEFAULT_INDICATORS.copy()

def get_required_indicators():
    """Get list of required indicators that cannot be disabled"""
    required = []
    for category, indicators in INDICATOR_CATEGORIES.items():
        for indicator_id, config in indicators.items():
            if config['required']:
                required.append(indicator_id)
    return required

def validate_indicator_selection(selected_indicators):
    """Validate that required indicators are selected"""
    required = get_required_indicators()
    missing_required = []
    
    for req_indicator in required:
        if req_indicator not in selected_indicators or not selected_indicators[req_indicator]:
            missing_required.append(req_indicator)
    
    return len(missing_required) == 0, missing_required

def get_indicator_info(indicator_id):
    """Get information about a specific indicator"""
    for category, indicators in INDICATOR_CATEGORIES.items():
        if indicator_id in indicators:
            return {
                'category': category,
                **indicators[indicator_id]
            }
    return None