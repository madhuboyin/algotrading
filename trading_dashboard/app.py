# app.py - Clean Flask Application without STOCK_SYMBOLS
from flask import Flask, render_template, jsonify, request
import warnings
from technical_analyzer import TechnicalAnalyzer
from tickers import get_quick_select_tickers, get_ticker_count  # Direct import from tickers
import traceback

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Flask Routes
@app.route('/')
def index():
    """Main dashboard page"""
    tickers = get_quick_select_tickers()
    print(f"Loading dashboard with {len(tickers)} quick select tickers: {tickers}")
    return render_template('index.html', symbols=tickers)

@app.route('/analyze/<symbol>')
def analyze_symbol(symbol):
    """Analyze a specific stock symbol"""
    try:
        # Convert to uppercase and validate basic format
        symbol = symbol.upper().strip()
        
        # Basic symbol validation
        if not symbol:
            return jsonify({'error': 'Symbol cannot be empty'}), 400
        
        if len(symbol) > 15:  # Handle crypto symbols like BTC-USD
            return jsonify({'error': 'Symbol too long. Max 15 characters.'}), 400
        
        # Allow alphanumeric and common trading symbols
        allowed_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-=')
        if not all(c in allowed_chars for c in symbol):
            return jsonify({'error': 'Invalid symbol format. Use only letters, numbers, dots, and hyphens.'}), 400
        
        print(f"Analyzing symbol: {symbol}")
        
        # Create analyzer instance
        analyzer = TechnicalAnalyzer(symbol)
        
        # Check if data was fetched successfully
        if analyzer.data is None:
            return jsonify({'error': f'No data available for symbol "{symbol}". Please check if it\'s a valid stock symbol.'}), 404
        
        if analyzer.data.empty:
            return jsonify({'error': f'No historical data found for "{symbol}". Symbol may be delisted or invalid.'}), 404
        
        # Check if we have enough data for analysis
        if len(analyzer.data) < 50:
            return jsonify({'error': f'Insufficient data for "{symbol}". Need at least 50 days of historical data for analysis.'}), 400
        
        # Perform analysis
        result = analyzer.identify_entry_signals()
        
        # Validate result structure
        if not result or 'symbol' not in result:
            return jsonify({'error': f'Analysis failed for "{symbol}". Unable to calculate technical indicators.'}), 500
        
        print(f"Analysis completed successfully for {symbol} - Score: {result.get('signal_score', 'N/A')}")
        return jsonify(result)
    
    except Exception as e:
        print(f"Error analyzing {symbol}: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        
        # Handle specific error types
        error_str = str(e).lower()
        
        if 'no data' in error_str or 'not found' in error_str:
            return jsonify({'error': f'Symbol "{symbol}" not found. Please verify it\'s a valid stock symbol.'}), 404
        elif 'insufficient' in error_str:
            return jsonify({'error': f'Insufficient data for "{symbol}". Symbol may be too new or inactive.'}), 400
        elif 'connection' in error_str or 'network' in error_str:
            return jsonify({'error': 'Network error. Please check your internet connection and try again.'}), 503
        else:
            return jsonify({'error': f'Analysis error for "{symbol}": {str(e)}'}), 500

@app.route('/api/tickers')
def get_tickers():
    """Get list of quick select tickers"""
    return jsonify({
        'tickers': get_quick_select_tickers(),
        'count': get_ticker_count()
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    tickers = get_quick_select_tickers()
    return jsonify({
        'status': 'healthy', 
        'message': 'Trading dashboard is running',
        'tickers_loaded': len(tickers),
        'tickers': tickers
    })

if __name__ == '__main__':
    print("Starting Clean Flask Algorithmic Trading Dashboard...")
    tickers = get_quick_select_tickers()
    print(f"Loaded {len(tickers)} quick select tickers: {', '.join(tickers)}")
    print("\nAccess the dashboard at: http://localhost:5001")
    print("Health check at: http://localhost:5001/health")
    print("API endpoints:")
    print("  /api/tickers - Get ticker list")
    print("\nFeatures:")
    print("• 10 Quick Select tickers for instant analysis")
    print("• Enter any stock symbol to analyze")
    print("• Real-time technical indicator calculations")
    print("• Entry signal scoring system")
    print("• Complete trading plans with stop loss & targets")
    print("• Entry recommendations for all signal types")
    print("• Mobile-responsive design")
    
    app.run(debug=True, host='0.0.0.0', port=5001)