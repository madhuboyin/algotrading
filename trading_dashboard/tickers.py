# tickers.py - Simple 10 Ticker Configuration

# Quick Select Tickers - Only 10 entries for the dashboard
QUICK_SELECT_TICKERS = [
    'AAPL',   # Apple Inc.
    'GOOGL',  # Alphabet Inc.
    'MSFT',   # Microsoft Corp.
    'TSLA',   # Tesla Inc.
    'NVDA',   # NVIDIA Corp.
    'AMZN',   # Amazon.com Inc.
    'META',   # Meta Platforms
    'AMD',    # Advanced Micro Devices
    'NFLX',   # Netflix Inc.
    'SPY'     # SPDR S&P 500 ETF
]

def get_quick_select_tickers():
    """Get the list of quick select tickers"""
    return QUICK_SELECT_TICKERS

def get_ticker_count():
    """Get the number of quick select tickers"""
    return len(QUICK_SELECT_TICKERS)