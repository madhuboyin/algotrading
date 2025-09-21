# technical_analyzer.py - Complete Enhanced Technical Analysis Engine with Fundamental Valuation
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from config import TRADING_CONFIG, SIGNAL_WEIGHTS, RSI_THRESHOLDS, ADX_THRESHOLDS, BB_THRESHOLDS

class TechnicalAnalyzer:
    def __init__(self, symbol, period=None):
        """
        Initialize the technical analyzer with stock data
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'GOOGL')
            period (str): Data period ('1y', '2y', '5y', etc.)
        """
        self.symbol = symbol
        self.period = period or TRADING_CONFIG['DATA_PERIOD']
        self.ticker = yf.Ticker(self.symbol)
        self.data = self.fetch_data(self.period)
        self.info = self.fetch_company_info()
        
    def fetch_data(self, period):
        """Fetch stock data from Yahoo Finance with improved error handling"""
        try:
            print(f"Fetching data for {self.symbol} with period {period}")
            
            # Fetch historical data
            data = self.ticker.history(period=period)
            
            if data is None or data.empty:
                print(f"No historical data returned for {self.symbol}")
                return None
            
            print(f"Successfully fetched {len(data)} days of data for {self.symbol}")
            print(f"Date range: {data.index[0]} to {data.index[-1]}")
            
            # Validate data quality
            if len(data) < 50:
                print(f"Warning: Only {len(data)} days of data available for {self.symbol}")
                return None
            
            # Check for required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                print(f"Missing required columns for {self.symbol}: {missing_columns}")
                return None
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {self.symbol}: {e}")
            return None
    
    def fetch_company_info(self):
        """Fetch company fundamental information"""
        try:
            info = self.ticker.info
            print(f"Successfully fetched company info for {self.symbol}")
            return info
        except Exception as e:
            print(f"Warning: Could not fetch company info for {self.symbol}: {e}")
            return {}
    
    def calculate_fundamental_valuation(self):
        """Calculate comprehensive fundamental valuation metrics"""
        current_price = self.data['Close'].iloc[-1] if self.data is not None else 0
        
        # Initialize valuation metrics with defaults
        valuation_metrics = {
            'current_price': current_price,
            'market_cap': None,
            'pe_ratio': None,
            'forward_pe': None,
            'peg_ratio': None,
            'price_to_book': None,
            'price_to_sales': None,
            'enterprise_value': None,
            'ev_to_revenue': None,
            'ev_to_ebitda': None,
            'dividend_yield': None,
            'return_on_equity': None,
            'debt_to_equity': None,
            'current_ratio': None,
            'revenue_growth': None,
            'earnings_growth': None,
            'profit_margin': None,
            'operating_margin': None,
            'gross_margin': None,
            'free_cash_flow': None,
            'book_value_per_share': None,
            'earnings_per_share': None,
            'revenue_per_share': None,
            'sector': None,
            'industry': None,
            'valuation_score': 0,
            'valuation_category': 'Unknown',
            'valuation_warnings': [],
            'valuation_strengths': []
        }
        
        if not self.info:
            valuation_metrics['valuation_warnings'].append("No fundamental data available")
            return valuation_metrics
        
        try:
            # Basic company info
            valuation_metrics['sector'] = self.info.get('sector', 'Unknown')
            valuation_metrics['industry'] = self.info.get('industry', 'Unknown')
            valuation_metrics['market_cap'] = self.info.get('marketCap')
            
            # Valuation ratios
            valuation_metrics['pe_ratio'] = self.info.get('trailingPE')
            valuation_metrics['forward_pe'] = self.info.get('forwardPE')
            valuation_metrics['peg_ratio'] = self.info.get('pegRatio')
            valuation_metrics['price_to_book'] = self.info.get('priceToBook')
            valuation_metrics['price_to_sales'] = self.info.get('priceToSalesTrailing12Months')
            valuation_metrics['enterprise_value'] = self.info.get('enterpriseValue')
            valuation_metrics['ev_to_revenue'] = self.info.get('enterpriseToRevenue')
            valuation_metrics['ev_to_ebitda'] = self.info.get('enterpriseToEbitda')
            
            # Income statement metrics
            valuation_metrics['dividend_yield'] = self.info.get('dividendYield')
            valuation_metrics['profit_margin'] = self.info.get('profitMargins')
            valuation_metrics['operating_margin'] = self.info.get('operatingMargins')
            valuation_metrics['gross_margin'] = self.info.get('grossMargins')
            valuation_metrics['return_on_equity'] = self.info.get('returnOnEquity')
            
            # Balance sheet metrics
            valuation_metrics['debt_to_equity'] = self.info.get('debtToEquity')
            valuation_metrics['current_ratio'] = self.info.get('currentRatio')
            valuation_metrics['book_value_per_share'] = self.info.get('bookValue')
            
            # Per share metrics
            valuation_metrics['earnings_per_share'] = self.info.get('trailingEps')
            valuation_metrics['revenue_per_share'] = self.info.get('revenuePerShare')
            valuation_metrics['free_cash_flow'] = self.info.get('freeCashflow')
            
            # Growth metrics
            valuation_metrics['revenue_growth'] = self.info.get('revenueGrowth')
            valuation_metrics['earnings_growth'] = self.info.get('earningsGrowth')
            
            # Calculate valuation score and category
            self._calculate_valuation_score(valuation_metrics)
            
        except Exception as e:
            print(f"Error calculating fundamental valuation for {self.symbol}: {e}")
            valuation_metrics['valuation_warnings'].append(f"Error processing fundamental data: {str(e)}")
        
        return valuation_metrics
    
    def _calculate_valuation_score(self, metrics):
        """Calculate overall valuation score based on fundamental metrics with sector-aware thresholds"""
        score = 0
        warnings = []
        strengths = []
        
        # Get sector for context-aware scoring
        sector = metrics.get('sector', 'Unknown')
        
        # Sector-specific P/E thresholds
        pe_thresholds = self._get_sector_pe_thresholds(sector)
        
        # P/E Ratio Analysis (Sector-Aware)
        pe_ratio = metrics.get('pe_ratio')
        if pe_ratio and pe_ratio > 0:
            if pe_ratio < pe_thresholds['low']:
                score += 3
                strengths.append(f"Low P/E ratio ({pe_ratio:.1f}) for {sector}")
            elif pe_ratio < pe_thresholds['fair']:
                score += 1
                strengths.append(f"Reasonable P/E ratio ({pe_ratio:.1f})")
            elif pe_ratio > pe_thresholds['high']:
                score -= 2
                warnings.append(f"High P/E ratio ({pe_ratio:.1f}) even for {sector}")
            elif pe_ratio > pe_thresholds['elevated']:
                score -= 1
                warnings.append(f"Elevated P/E ratio ({pe_ratio:.1f})")
        elif pe_ratio and pe_ratio < 0:
            score -= 2
            warnings.append("Negative earnings (loss-making)")
        
        # PEG Ratio Analysis (More Lenient)
        peg_ratio = metrics.get('peg_ratio')
        if peg_ratio and peg_ratio > 0:
            if peg_ratio < 0.8:
                score += 3
                strengths.append(f"Excellent PEG ratio ({peg_ratio:.2f})")
            elif peg_ratio < 1.3:
                score += 1
                strengths.append(f"Good PEG ratio ({peg_ratio:.2f})")
            elif peg_ratio > 3.0:
                score -= 2
                warnings.append(f"High PEG ratio ({peg_ratio:.2f})")
        
        # Price-to-Book Analysis (Sector-Aware)
        pb_ratio = metrics.get('price_to_book')
        if pb_ratio and pb_ratio > 0:
            if sector in ['Technology', 'Software', 'Internet']:
                # Tech companies naturally have higher P/B ratios
                if pb_ratio < 3.0:
                    score += 1
                    strengths.append(f"Low P/B ratio ({pb_ratio:.2f}) for tech")
                elif pb_ratio > 15.0:
                    score -= 1
                    warnings.append(f"Very high P/B ratio ({pb_ratio:.2f})")
            else:
                # Traditional industries
                if pb_ratio < 1.5:
                    score += 2
                    strengths.append(f"Low P/B ratio ({pb_ratio:.2f})")
                elif pb_ratio > 5.0:
                    score -= 1
                    warnings.append(f"High P/B ratio ({pb_ratio:.2f})")
        
        # Price-to-Sales Analysis (Sector-Aware)
        ps_ratio = metrics.get('price_to_sales')
        if ps_ratio and ps_ratio > 0:
            ps_thresholds = self._get_sector_ps_thresholds(sector)
            if ps_ratio < ps_thresholds['low']:
                score += 1
                strengths.append(f"Low P/S ratio ({ps_ratio:.2f})")
            elif ps_ratio > ps_thresholds['high']:
                score -= 1
                warnings.append(f"High P/S ratio ({ps_ratio:.2f}) for {sector}")
        
        # Profitability Analysis (More Realistic)
        profit_margin = metrics.get('profit_margin')
        if profit_margin is not None:
            if profit_margin > 0.25:  # 25%
                score += 2
                strengths.append(f"Excellent profit margin ({profit_margin*100:.1f}%)")
            elif profit_margin > 0.15:  # 15%
                score += 1
                strengths.append(f"Good profit margin ({profit_margin*100:.1f}%)")
            elif profit_margin > 0.05:  # 5%
                # Neutral - no penalty for modest profitability
                pass
            elif profit_margin < 0:
                score -= 2
                warnings.append("Negative profit margin")
        
        # Return on Equity Analysis (More Realistic)
        roe = metrics.get('return_on_equity')
        if roe is not None:
            if roe > 0.25:  # 25%
                score += 2
                strengths.append(f"Excellent ROE ({roe*100:.1f}%)")
            elif roe > 0.15:  # 15%
                score += 1
                strengths.append(f"Good ROE ({roe*100:.1f}%)")
            elif roe > 0.08:  # 8%
                # Neutral - decent ROE, no bonus or penalty
                pass
            elif roe < 0:
                score -= 1
                warnings.append("Negative ROE")
        
        # Debt Analysis (More Nuanced)
        debt_to_equity = metrics.get('debt_to_equity')
        if debt_to_equity is not None:
            if debt_to_equity < 25:  # Very low debt
                score += 1
                strengths.append("Very low debt levels")
            elif debt_to_equity < 50:  # Reasonable debt
                # Neutral - reasonable debt level
                pass
            elif debt_to_equity > 150:  # High debt
                score -= 2
                warnings.append("High debt levels")
            elif debt_to_equity > 100:  # Elevated debt
                score -= 1
                warnings.append("Elevated debt levels")
        
        # Growth Analysis (More Realistic Thresholds)
        revenue_growth = metrics.get('revenue_growth')
        if revenue_growth is not None:
            if revenue_growth > 0.25:  # 25%
                score += 2
                strengths.append(f"Strong revenue growth ({revenue_growth*100:.1f}%)")
            elif revenue_growth > 0.10:  # 10%
                score += 1
                strengths.append(f"Good revenue growth ({revenue_growth*100:.1f}%)")
            elif revenue_growth > 0.03:  # 3%
                # Neutral - modest growth, no penalty
                pass
            elif revenue_growth < -0.05:  # -5%
                score -= 2
                warnings.append("Declining revenue")
        
        # Dividend Analysis (More Balanced)
        dividend_yield = metrics.get('dividend_yield')
        if dividend_yield and dividend_yield > 0:
            if dividend_yield > 0.10:  # 10%
                score -= 1
                warnings.append(f"Very high dividend yield ({dividend_yield*100:.1f}%) - potential distress")
            elif dividend_yield > 0.06:  # 6%
                # Neutral - high but not necessarily bad
                pass
            elif dividend_yield > 0.02:  # 2%
                score += 1
                strengths.append(f"Reasonable dividend yield ({dividend_yield*100:.1f}%)")
        
        # Determine valuation category with more balanced thresholds
        if score >= 6:
            category = "Undervalued"
        elif score >= 2:
            category = "Fair Value"
        elif score >= -2:
            category = "Fairly Valued"
        elif score >= -5:
            category = "Overvalued"
        else:
            category = "Highly Overvalued"
        
        # Cap the score (more balanced range)
        score = max(-8, min(8, score))
        
        metrics['valuation_score'] = score
        metrics['valuation_category'] = category
        metrics['valuation_warnings'] = warnings
        metrics['valuation_strengths'] = strengths
    
    def _get_sector_pe_thresholds(self, sector):
        """Get sector-appropriate P/E ratio thresholds"""
        sector_thresholds = {
            'Technology': {'low': 20, 'fair': 35, 'elevated': 50, 'high': 70},
            'Software': {'low': 25, 'fair': 40, 'elevated': 60, 'high': 80},
            'Internet': {'low': 20, 'fair': 35, 'elevated': 55, 'high': 75},
            'Healthcare': {'low': 15, 'fair': 25, 'elevated': 35, 'high': 50},
            'Financial Services': {'low': 8, 'fair': 15, 'elevated': 20, 'high': 25},
            'Consumer Cyclical': {'low': 12, 'fair': 20, 'elevated': 30, 'high': 40},
            'Consumer Defensive': {'low': 15, 'fair': 22, 'elevated': 30, 'high': 40},
            'Industrials': {'low': 12, 'fair': 18, 'elevated': 25, 'high': 35},
            'Energy': {'low': 8, 'fair': 15, 'elevated': 25, 'high': 40},
            'Utilities': {'low': 12, 'fair': 18, 'elevated': 22, 'high': 28},
            'Real Estate': {'low': 10, 'fair': 20, 'elevated': 30, 'high': 40},
        }
        
        # Default thresholds for unknown sectors
        return sector_thresholds.get(sector, {'low': 15, 'fair': 25, 'elevated': 35, 'high': 50})
    
    def _get_sector_ps_thresholds(self, sector):
        """Get sector-appropriate Price-to-Sales thresholds"""
        sector_thresholds = {
            'Technology': {'low': 5, 'high': 15},
            'Software': {'low': 8, 'high': 20},
            'Internet': {'low': 5, 'high': 15},
            'Healthcare': {'low': 3, 'high': 8},
            'Financial Services': {'low': 2, 'high': 5},
            'Consumer Cyclical': {'low': 1, 'high': 3},
            'Consumer Defensive': {'low': 1, 'high': 4},
            'Industrials': {'low': 1, 'high': 3},
            'Energy': {'low': 1, 'high': 2},
            'Utilities': {'low': 1, 'high': 3},
            'Real Estate': {'low': 2, 'high': 6},
        }
        
        # Default thresholds
        return sector_thresholds.get(sector, {'low': 2, 'high': 8})


    def calculate_sma(self, window):
        """Calculate Simple Moving Average"""
        return self.data['Close'].rolling(window=window).mean()
    
    def calculate_ema(self, window):
        """Calculate Exponential Moving Average"""
        return self.data['Close'].ewm(span=window).mean()
    
    def calculate_rsi(self, window=None):
        """Calculate Relative Strength Index"""
        window = window or TRADING_CONFIG['RSI_PERIOD']
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, fast=None, slow=None, signal=None):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        fast = fast or TRADING_CONFIG['MACD_FAST']
        slow = slow or TRADING_CONFIG['MACD_SLOW'] 
        signal = signal or TRADING_CONFIG['MACD_SIGNAL']
        
        ema_fast = self.data['Close'].ewm(span=fast).mean()
        ema_slow = self.data['Close'].ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_adx(self, window=None):
        """Calculate Average Directional Index (ADX) with proper Wilder's smoothing"""
        window = window or TRADING_CONFIG['ADX_PERIOD']
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']
        
        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement calculation
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        # Set negative values to 0
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Only keep the larger of the two (directional movement rule)
        plus_dm[(plus_dm - minus_dm) < 0] = 0
        minus_dm[(minus_dm - plus_dm) < 0] = 0
        
        # Wilder's smoothing function (exponential with alpha = 1/period)
        def wilders_smoothing(series, period):
            alpha = 1.0 / period
            return series.ewm(alpha=alpha, adjust=False).mean()
        
        # Apply Wilder's smoothing (this is the key fix)
        tr_smooth = wilders_smoothing(tr, window)
        plus_dm_smooth = wilders_smoothing(plus_dm, window)
        minus_dm_smooth = wilders_smoothing(minus_dm, window)
        
        # Calculate Directional Indicators
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        
        # Calculate DX (Directional Index)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Apply Wilder's smoothing to DX to get ADX (this is also important)
        adx = wilders_smoothing(dx, window)
        
        return adx, plus_di, minus_di
    
    def calculate_bollinger_bands(self, window=None, num_std=None):
        """Calculate Bollinger Bands"""
        window = window or TRADING_CONFIG['BB_PERIOD']
        num_std = num_std or TRADING_CONFIG['BB_STD']
        
        sma = self.data['Close'].rolling(window=window).mean()
        std = self.data['Close'].rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, sma, lower_band
    
    def calculate_stochastic(self, k_window=None, d_window=None):
        """Calculate Stochastic Oscillator"""
        k_window = k_window or TRADING_CONFIG['STOCH_K']
        d_window = d_window or TRADING_CONFIG['STOCH_D']
        
        lowest_low = self.data['Low'].rolling(window=k_window).min()
        highest_high = self.data['High'].rolling(window=k_window).max()
        k_percent = 100 * ((self.data['Close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    def calculate_volume_sma(self, window=None):
        """Calculate Volume Simple Moving Average"""
        window = window or TRADING_CONFIG['VOLUME_SMA']
        return self.data['Volume'].rolling(window=window).mean()
    
    def calculate_atr(self, window=None):
        """Calculate Average True Range for volatility-based stops"""
        window = window or TRADING_CONFIG['ATR_PERIOD']
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0
    
    def calculate_williams_r(self, window=14):
        """Calculate Williams %R oscillator"""
        highest_high = self.data['High'].rolling(window=window).max()
        lowest_low = self.data['Low'].rolling(window=window).min()
        williams_r = -100 * ((highest_high - self.data['Close']) / (highest_high - lowest_low))
        return williams_r
    
    def calculate_commodity_channel_index(self, window=20):
        """Calculate Commodity Channel Index (CCI)"""
        typical_price = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        sma_tp = typical_price.rolling(window=window).mean()
        mad = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci
    
    def calculate_money_flow_index(self, window=14):
        """Calculate Money Flow Index (MFI)"""
        typical_price = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        money_flow = typical_price * self.data['Volume']
        
        positive_flow = pd.Series(0.0, index=self.data.index)
        negative_flow = pd.Series(0.0, index=self.data.index)
        
        positive_mask = typical_price > typical_price.shift(1)
        negative_mask = typical_price < typical_price.shift(1)
        
        positive_flow[positive_mask] = money_flow[positive_mask]
        negative_flow[negative_mask] = money_flow[negative_mask]
        
        positive_mf = positive_flow.rolling(window=window).sum()
        negative_mf = negative_flow.rolling(window=window).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
    
    def calculate_on_balance_volume(self):
        """Calculate On-Balance Volume (OBV)"""
        obv = pd.Series(0.0, index=self.data.index)
        obv.iloc[0] = self.data['Volume'].iloc[0]
        
        for i in range(1, len(self.data)):
            if self.data['Close'].iloc[i] > self.data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + self.data['Volume'].iloc[i]
            elif self.data['Close'].iloc[i] < self.data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - self.data['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def calculate_vwap(self, window=20):
        """Calculate Volume Weighted Average Price"""
        typical_price = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        vwap = (typical_price * self.data['Volume']).rolling(window=window).sum() / self.data['Volume'].rolling(window=window).sum()
        return vwap
    
    def calculate_price_channels(self, window=20):
        """Calculate Price Channels (Donchian Channels)"""
        upper_channel = self.data['High'].rolling(window=window).max()
        lower_channel = self.data['Low'].rolling(window=window).min()
        middle_channel = (upper_channel + lower_channel) / 2
        return upper_channel, middle_channel, lower_channel
    
    def calculate_roc(self, window=12):
        """Calculate Rate of Change (ROC)"""
        roc = ((self.data['Close'] - self.data['Close'].shift(window)) / self.data['Close'].shift(window)) * 100
        return roc
    
    def calculate_trix(self, window=14):
        """Calculate TRIX indicator"""
        ema1 = self.data['Close'].ewm(span=window).mean()
        ema2 = ema1.ewm(span=window).mean()
        ema3 = ema2.ewm(span=window).mean()
        trix = ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 10000
        return trix
    
    def calculate_ultimate_oscillator(self, period1=7, period2=14, period3=28):
        """Calculate Ultimate Oscillator"""
        tr1 = self.data['High'] - self.data['Low']
        tr2 = abs(self.data['High'] - self.data['Close'].shift(1))
        tr3 = abs(self.data['Low'] - self.data['Close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        bp = self.data['Close'] - pd.concat([
            self.data['Low'],
            self.data['Close'].shift(1)
        ], axis=1).min(axis=1)
        
        bp_sum1 = bp.rolling(window=period1).sum()
        bp_sum2 = bp.rolling(window=period2).sum()
        bp_sum3 = bp.rolling(window=period3).sum()
        
        tr_sum1 = tr.rolling(window=period1).sum()
        tr_sum2 = tr.rolling(window=period2).sum()
        tr_sum3 = tr.rolling(window=period3).sum()
        
        uo = 100 * ((4 * bp_sum1/tr_sum1) + (2 * bp_sum2/tr_sum2) + (bp_sum3/tr_sum3)) / 7
        return uo
    
    def calculate_support_resistance(self, lookback=None):
        """Calculate support and resistance levels"""
        lookback = lookback or TRADING_CONFIG['SUPPORT_RESISTANCE_LOOKBACK']
        high_prices = self.data['High'].tail(lookback)
        low_prices = self.data['Low'].tail(lookback)
        
        # Support: Recent lows
        support_level = low_prices.min()
        
        # Resistance: Recent highs
        resistance_level = high_prices.max()
        
        return support_level, resistance_level
    
    def calculate_valuation_metrics(self):
        """Calculate technical valuation-based metrics using price action"""
        sma_50 = self.calculate_sma(50)
        sma_100 = self.calculate_sma(100)
        sma_200 = self.calculate_sma(200)
        ema_20 = self.calculate_ema(20)
        
        current_price = self.data['Close'].iloc[-1]
        
        # Distance from key moving averages (valuation indicators)
        distance_from_sma200 = ((current_price - sma_200.iloc[-1]) / sma_200.iloc[-1]) * 100
        distance_from_sma100 = ((current_price - sma_100.iloc[-1]) / sma_100.iloc[-1]) * 100
        distance_from_sma50 = ((current_price - sma_50.iloc[-1]) / sma_50.iloc[-1]) * 100
        distance_from_ema20 = ((current_price - ema_20.iloc[-1]) / ema_20.iloc[-1]) * 100
        
        # Calculate 52-week high/low position
        high_52w = self.data['High'].tail(252).max()  # ~52 weeks of trading days
        low_52w = self.data['Low'].tail(252).min()
        position_52w = ((current_price - low_52w) / (high_52w - low_52w)) * 100
        
        # Recent volatility
        returns = self.data['Close'].pct_change()
        volatility_20d = returns.tail(20).std() * np.sqrt(252) * 100  # Annualized volatility
        
        return {
            'distance_from_sma200': distance_from_sma200,
            'distance_from_sma100': distance_from_sma100,
            'distance_from_sma50': distance_from_sma50,
            'distance_from_ema20': distance_from_ema20,
            'position_52w': position_52w,
            'high_52w': high_52w,
            'low_52w': low_52w,
            'volatility_20d': volatility_20d
        }
    
    def calculate_enhanced_valuation_metrics(self):
        """Enhanced valuation metrics with more balanced scoring"""
        # Get fundamental valuation
        fundamental_metrics = self.calculate_fundamental_valuation()
        
        # Get existing technical valuation metrics
        technical_metrics = self.calculate_valuation_metrics()
        
        # Combine both approaches
        enhanced_metrics = {
            **fundamental_metrics,
            **technical_metrics,
            'combined_valuation_penalty': 0,
            'valuation_analysis': {
                'fundamental_score': fundamental_metrics.get('valuation_score', 0),
                'technical_signals': [],
                'overall_assessment': '',
                'risk_factors': [],
                'opportunities': []
            }
        }
        
        # Calculate combined valuation penalty for scoring system (More Balanced)
        fundamental_score = fundamental_metrics.get('valuation_score', 0)
        
        # Convert fundamental score to penalty (more balanced)
        if fundamental_score >= 4:  # Strong undervalued
            penalty = 2  # Bonus for undervalued stocks
        elif fundamental_score >= 1:  # Slight undervalued to fair
            penalty = 1  # Small bonus
        elif fundamental_score >= -1:  # Fairly valued
            penalty = 0  # No penalty
        elif fundamental_score >= -3:  # Moderately overvalued
            penalty = -1  # Small penalty
        elif fundamental_score >= -5:  # Overvalued
            penalty = -2  # Moderate penalty
        else:  # Highly overvalued
            penalty = -3  # Significant penalty (reduced from -5)
        
        # Adjust penalty based on technical indicators (Keep existing logic but cap it)
        distance_sma200 = technical_metrics.get('distance_from_sma200', 0)
        position_52w = technical_metrics.get('position_52w', 50)
        
        # Technical overextension penalties (Reduced)
        if distance_sma200 > 40:  # Very far from SMA 200
            penalty -= 2
            enhanced_metrics['valuation_analysis']['risk_factors'].append("Price very far above SMA 200")
        elif distance_sma200 > 25:  # Far from SMA 200
            penalty -= 1
            
        if position_52w > 95:  # Very near 52W highs
            penalty -= 1
            enhanced_metrics['valuation_analysis']['risk_factors'].append("At 52-week highs")
        elif position_52w > 85:  # Near 52W highs
            # No penalty - being near highs isn't necessarily bad
            pass
        
        # Technical support opportunities
        if position_52w < 15:  # Very near 52W lows
            penalty += 1
            enhanced_metrics['valuation_analysis']['opportunities'].append("Near 52-week lows - potential value")
        elif position_52w < 30:  # In lower range
            enhanced_metrics['valuation_analysis']['opportunities'].append("In lower price range")
        
        # Cap the combined penalty (more reasonable range)
        enhanced_metrics['combined_valuation_penalty'] = max(-5, min(3, penalty))
        
        # Overall assessment
        if penalty >= 1:
            assessment = "Attractive valuation opportunity"
        elif penalty >= 0:
            assessment = "Fair to reasonable valuation"
        elif penalty >= -2:
            assessment = "Slightly overvalued but acceptable"
        else:
            assessment = "Overvalued - exercise caution"
        
        enhanced_metrics['valuation_analysis']['overall_assessment'] = assessment
        
        return enhanced_metrics
    
    def calculate_trend_strength(self):
        """Calculate overall trend strength"""
        df = self.get_all_indicators()
        latest = df.iloc[-1]
        
        trend_score = 0
        trend_signals = []
        
        # Moving average trend alignment
        if latest['Close'] > latest['EMA_50'] > latest['SMA_200']:
            trend_score += 3
            trend_signals.append("Strong MA Alignment")
        elif latest['Close'] > latest['EMA_50']:
            trend_score += 2
            trend_signals.append("Above EMA 50")
        elif latest['Close'] > latest['SMA_200']:
            trend_score += 1
            trend_signals.append("Above SMA 200")
        else:
            trend_score -= 2
            trend_signals.append("Below Key MAs")
        
        # ADX trend strength
        if latest['ADX'] > 25:
            if latest['Plus_DI'] > latest['Minus_DI']:
                trend_score += 2
                trend_signals.append("Strong Uptrend")
            else:
                trend_score -= 2
                trend_signals.append("Strong Downtrend")
        
        # Price channel position
        if 'PC_Upper' in latest and 'PC_Lower' in latest:
            pc_position = (latest['Close'] - latest['PC_Lower']) / (latest['PC_Upper'] - latest['PC_Lower'])
            if pc_position > 0.8:
                trend_score -= 1  # Near resistance
                trend_signals.append("Near Channel Top")
            elif pc_position < 0.2:
                trend_score += 1  # Near support
                trend_signals.append("Near Channel Bottom")
        
        return trend_score, trend_signals
    
    def calculate_momentum_strength(self):
        """Calculate momentum strength across multiple indicators"""
        df = self.get_all_indicators()
        latest = df.iloc[-1]
        recent = df.tail(5)
        
        momentum_score = 0
        momentum_signals = []
        
        # RSI momentum
        if 30 <= latest['RSI'] <= 50:
            momentum_score += 2
            momentum_signals.append("RSI Oversold Recovery")
        elif 50 < latest['RSI'] <= 65:
            momentum_score += 1
            momentum_signals.append("RSI Bullish")
        elif latest['RSI'] > 75:
            momentum_score -= 2
            momentum_signals.append("RSI Overbought")
        
        # MACD momentum
        if (latest['MACD'] > latest['MACD_Signal'] and 
            recent['MACD'].iloc[-2] <= recent['MACD_Signal'].iloc[-2]):
            momentum_score += 3
            momentum_signals.append("MACD Bullish Cross")
        elif latest['MACD'] > latest['MACD_Signal'] and latest['MACD_Histogram'] > 0:
            momentum_score += 1
            momentum_signals.append("MACD Bullish")
        
        # Williams %R
        if 'Williams_R' in latest:
            if -50 <= latest['Williams_R'] <= -20:
                momentum_score += 1
                momentum_signals.append("Williams %R Bullish")
            elif latest['Williams_R'] > -20:
                momentum_score -= 1
                momentum_signals.append("Williams %R Overbought")
        
        # Stochastic
        if (latest['Stoch_K'] > latest['Stoch_D'] and 
            latest['Stoch_K'] < 80 and latest['Stoch_D'] < 80):
            momentum_score += 1
            momentum_signals.append("Stochastic Bullish")
        
        # Rate of Change
        if 'ROC' in latest and latest['ROC'] > 2:
            momentum_score += 1
            momentum_signals.append("Positive ROC")
        elif 'ROC' in latest and latest['ROC'] < -5:
            momentum_score -= 1
            momentum_signals.append("Negative ROC")
        
        return momentum_score, momentum_signals
    
    def calculate_volume_strength(self):
        """Calculate volume-based strength indicators"""
        df = self.get_all_indicators()
        latest = df.iloc[-1]
        
        volume_score = 0
        volume_signals = []
        
        # Volume vs average
        if latest['Volume'] > latest['Volume_SMA'] * 1.5:
            volume_score += 2
            volume_signals.append("High Volume")
        elif latest['Volume'] > latest['Volume_SMA']:
            volume_score += 1
            volume_signals.append("Above Avg Volume")
        
        # Money Flow Index
        if 'MFI' in latest:
            if 20 <= latest['MFI'] <= 50:
                volume_score += 1
                volume_signals.append("MFI Oversold Recovery")
            elif latest['MFI'] > 80:
                volume_score -= 1
                volume_signals.append("MFI Overbought")
        
        # On-Balance Volume trend
        if 'OBV' in df.columns:
            obv_trend = df['OBV'].tail(5).pct_change().mean()
            if obv_trend > 0.01:
                volume_score += 1
                volume_signals.append("OBV Trending Up")
            elif obv_trend < -0.01:
                volume_score -= 1
                volume_signals.append("OBV Trending Down")
        
        return volume_score, volume_signals
    
    def calculate_fair_value_entry_points(self, current_price):
        """Calculate suggested entry points based on technical levels"""
        df = self.get_all_indicators()
        latest = df.iloc[-1]
        valuation = self.calculate_enhanced_valuation_metrics()
        
        entry_suggestions = {}
        
        # Support levels for entry
        support_levels = []
        
        # EMA 50 as dynamic support
        if latest['EMA_50'] < current_price:
            support_levels.append(('EMA 50', latest['EMA_50']))
        
        # SMA 200 as major support
        if latest['SMA_200'] < current_price:
            support_levels.append(('SMA 200', latest['SMA_200']))
        
        # Bollinger Band lower as support
        if latest['BB_Lower'] < current_price:
            support_levels.append(('BB Lower', latest['BB_Lower']))
        
        # Price channel lower
        if 'PC_Lower' in latest and latest['PC_Lower'] < current_price:
            support_levels.append(('Channel Lower', latest['PC_Lower']))
        
        # VWAP as fair value
        if 'VWAP' in latest:
            support_levels.append(('VWAP', latest['VWAP']))
        
        # Sort by price (descending)
        support_levels.sort(key=lambda x: x[1], reverse=True)
        
        # Determine if stock is overpriced
        is_overpriced = False
        overpriced_reasons = []
        
        # Check distance from key levels
        if valuation['distance_from_sma200'] > 20:
            is_overpriced = True
            overpriced_reasons.append(f">{valuation['distance_from_sma200']:.1f}% above SMA 200")
        
        if valuation['position_52w'] > 85:
            is_overpriced = True
            overpriced_reasons.append(f"Near 52W high ({valuation['position_52w']:.1f}%)")
        
        if latest['RSI'] > 75:
            is_overpriced = True
            overpriced_reasons.append(f"RSI overbought ({latest['RSI']:.1f})")
        
        # Suggest entry points
        if is_overpriced:
            # Suggest better entry levels
            entry_suggestions['is_overpriced'] = True
            entry_suggestions['reasons'] = overpriced_reasons
            entry_suggestions['suggested_entries'] = []
            
            for level_name, level_price in support_levels[:3]:  # Top 3 support levels
                discount = ((current_price - level_price) / current_price) * 100
                if discount > 2:  # At least 2% discount
                    entry_suggestions['suggested_entries'].append({
                        'level': level_name,
                        'price': level_price,
                        'discount': discount
                    })
        else:
            entry_suggestions['is_overpriced'] = False
            entry_suggestions['current_fair'] = True
        
        return entry_suggestions
    
    def get_all_indicators(self):
        """Calculate all technical indicators"""
        if self.data is None or self.data.empty:
            return pd.DataFrame()
            
        indicators = {}
        
        # Moving Averages
        indicators['SMA_200'] = self.calculate_sma(TRADING_CONFIG['SMA_LONG'])
        indicators['EMA_50'] = self.calculate_ema(TRADING_CONFIG['EMA_MEDIUM'])
        indicators['SMA_20'] = self.calculate_sma(TRADING_CONFIG['SMA_SHORT'])
        indicators['SMA_100'] = self.calculate_sma(100)
        
        # RSI
        indicators['RSI'] = self.calculate_rsi()
        
        # MACD
        macd_line, signal_line, histogram = self.calculate_macd()
        indicators['MACD'] = macd_line
        indicators['MACD_Signal'] = signal_line
        indicators['MACD_Histogram'] = histogram
        
        # ADX
        adx, plus_di, minus_di = self.calculate_adx()
        indicators['ADX'] = adx
        indicators['Plus_DI'] = plus_di
        indicators['Minus_DI'] = minus_di
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands()
        indicators['BB_Upper'] = bb_upper
        indicators['BB_Middle'] = bb_middle
        indicators['BB_Lower'] = bb_lower
        
        # Stochastic
        k_percent, d_percent = self.calculate_stochastic()
        indicators['Stoch_K'] = k_percent
        indicators['Stoch_D'] = d_percent
        
        # Volume indicators
        indicators['Volume_SMA'] = self.calculate_volume_sma()
        indicators['OBV'] = self.calculate_on_balance_volume()
        indicators['MFI'] = self.calculate_money_flow_index()
        
        # Additional technical indicators
        indicators['Williams_R'] = self.calculate_williams_r()
        indicators['CCI'] = self.calculate_commodity_channel_index()
        indicators['VWAP'] = self.calculate_vwap()
        indicators['ROC'] = self.calculate_roc()
        indicators['TRIX'] = self.calculate_trix()
        indicators['UO'] = self.calculate_ultimate_oscillator()
        
        # Price channels
        pc_upper, pc_middle, pc_lower = self.calculate_price_channels()
        indicators['PC_Upper'] = pc_upper
        indicators['PC_Middle'] = pc_middle
        indicators['PC_Lower'] = pc_lower
        
        # Price data
        indicators['Close'] = self.data['Close']
        indicators['Volume'] = self.data['Volume']
        indicators['High'] = self.data['High']
        indicators['Low'] = self.data['Low']
        
        return pd.DataFrame(indicators)
    
    def calculate_stop_loss_exit_points(self, entry_price, target_return=None):
        """
        Calculate stop loss and exit points for swing trading
        
        Args:
            entry_price (float): Entry price for the position
            target_return (float): Target return (default from config)
        
        Returns:
            dict: Stop loss and exit point recommendations
        """
        target_return = target_return or TRADING_CONFIG['TARGET_RETURN']
        df = self.get_all_indicators()
        latest = df.iloc[-1]
        atr = self.calculate_atr()
        support_level, resistance_level = self.calculate_support_resistance()
        
        # Multiple stop loss methods
        stop_loss_methods = {}
        
        # 1. ATR-based stop loss
        atr_stop = entry_price - (TRADING_CONFIG['ATR_MULTIPLIER'] * atr)
        stop_loss_methods['ATR_Stop'] = atr_stop
        
        # 2. Percentage-based stop loss
        pct_stop_conservative = entry_price * (1 - TRADING_CONFIG['PCT_STOP_CONSERVATIVE'])
        pct_stop_aggressive = entry_price * (1 - TRADING_CONFIG['PCT_STOP_AGGRESSIVE'])
        stop_loss_methods['PCT_Conservative'] = pct_stop_conservative
        stop_loss_methods['PCT_Aggressive'] = pct_stop_aggressive
        
        # 3. EMA 50 based stop loss
        ema50_stop = latest['EMA_50'] * (1 - TRADING_CONFIG['EMA50_STOP_BUFFER'])
        stop_loss_methods['EMA50_Stop'] = ema50_stop
        
        # 4. Support level based stop loss
        support_stop = support_level * (1 - TRADING_CONFIG['SUPPORT_STOP_BUFFER'])
        stop_loss_methods['Support_Stop'] = support_stop
        
        # 5. Bollinger Band lower band stop
        bb_stop = latest['BB_Lower'] * (1 - TRADING_CONFIG['BB_STOP_BUFFER'])
        stop_loss_methods['BB_Stop'] = bb_stop
        
        # Select recommended stop loss (most conservative approach)
        recommended_stop = max([
            atr_stop,
            pct_stop_conservative,
            ema50_stop * 0.995  # Slightly below EMA50 for safety
        ])
        
        # Ensure stop loss is not too tight
        min_stop = entry_price * (1 - TRADING_CONFIG['MIN_STOP_LOSS_PCT'])
        if recommended_stop > min_stop:
            recommended_stop = min_stop
        
        # Calculate exit targets
        exit_targets = {}
        targets = TRADING_CONFIG['PARTIAL_TARGETS']
        
        # Partial profit targets
        exit_targets['Target_1'] = entry_price * (1 + targets['TARGET_1'])
        exit_targets['Target_2'] = entry_price * (1 + targets['TARGET_2']) 
        exit_targets['Target_3'] = entry_price * (1 + targets['TARGET_3'])
        exit_targets['Primary_Target'] = entry_price * (1 + target_return)
        
        # Resistance-based target
        if resistance_level > entry_price:
            resistance_target = resistance_level * 0.98  # Just below resistance
            if resistance_target > entry_price * 1.10:  # At least 10% gain
                exit_targets['Resistance_Target'] = resistance_target
        
        # Bollinger Band upper target
        bb_upper_target = latest['BB_Upper']
        if bb_upper_target > entry_price * 1.10:
            exit_targets['BB_Upper_Target'] = bb_upper_target
        
        # Risk-reward calculation
        risk_amount = entry_price - recommended_stop
        reward_amount = exit_targets['Primary_Target'] - entry_price
        risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        
        return {
            'stop_loss': {
                'recommended': recommended_stop,
                'risk_percent': ((entry_price - recommended_stop) / entry_price) * 100,
                'all_methods': stop_loss_methods
            },
            'exit_targets': exit_targets,
            'risk_reward_ratio': risk_reward_ratio,
            'position_sizing': {
                'max_risk_per_trade': TRADING_CONFIG['MAX_PORTFOLIO_RISK'] * 100,
                'suggested_position_size': f"Risk max {TRADING_CONFIG['MAX_PORTFOLIO_RISK']*100}% of portfolio on this trade"
            },
            'holding_period': {
                'target_period': TRADING_CONFIG['TARGET_HOLDING_PERIOD'],
                'monitoring_frequency': TRADING_CONFIG['REVIEW_FREQUENCY']
            }
        }
    
    def identify_entry_signals(self):
        """Enhanced entry signals with improved valuation analysis"""
        df = self.get_all_indicators()
        
        if df is None or df.empty:
            return None
        
        # Get the latest data point
        latest = df.iloc[-1]
        recent = df.tail(TRADING_CONFIG['RECENT_DAYS'])
        
        # Calculate trend, momentum, and volume scores
        trend_score, trend_signals = self.calculate_trend_strength()
        momentum_score, momentum_signals = self.calculate_momentum_strength()
        volume_score, volume_signals = self.calculate_volume_strength()
        
        # Cap scores
        trend_score = min(trend_score, TRADING_CONFIG['MAX_TREND_SCORE'])
        trend_score = max(trend_score, -TRADING_CONFIG['MAX_TREND_SCORE'])
        
        momentum_score = min(momentum_score, TRADING_CONFIG['MAX_MOMENTUM_SCORE'])
        momentum_score = max(momentum_score, -TRADING_CONFIG['MAX_MOMENTUM_SCORE'])
        
        volume_score = min(volume_score, TRADING_CONFIG['MAX_VOLUME_SCORE'])
        volume_score = max(volume_score, 0)
        
        # Base signal analysis
        base_signals = []
        base_score = 0
        
        # 1. MACD Analysis
        if (latest['MACD'] > latest['MACD_Signal'] and 
            recent['MACD'].iloc[-2] <= recent['MACD_Signal'].iloc[-2] and
            latest['MACD_Histogram'] > recent['MACD_Histogram'].iloc[-2]):
            base_signals.append("MACD Bullish Crossover")
            base_score += 3
        elif (latest['MACD'] > latest['MACD_Signal'] and 
            latest['MACD_Histogram'] > 0):
            base_signals.append("MACD Above Signal")
            base_score += 1
        
        # 2. RSI Analysis
        if 25 <= latest['RSI'] <= 45:
            base_signals.append("RSI Oversold Recovery Zone")
            base_score += 3
        elif 45 < latest['RSI'] <= 60:
            base_signals.append("RSI Neutral-Bullish")
            base_score += 1
        elif latest['RSI'] > 75:
            base_signals.append("RSI Overbought Warning")
            base_score -= 3
        
        # 3. Multiple Timeframe MA Analysis
        ma_alignment_score = 0
        if latest['Close'] > latest['EMA_50'] > latest['SMA_100'] > latest['SMA_200']:
            base_signals.append("Perfect MA Alignment")
            ma_alignment_score = 4
        elif latest['Close'] > latest['EMA_50'] > latest['SMA_200']:
            base_signals.append("Strong MA Alignment")
            ma_alignment_score = 3
        elif latest['Close'] > latest['EMA_50']:
            base_signals.append("Above EMA 50")
            ma_alignment_score = 2
        elif latest['Close'] > latest['SMA_200']:
            base_signals.append("Above SMA 200")
            ma_alignment_score = 1
        else:
            base_signals.append("Below Key MAs")
            ma_alignment_score = -2
        
        base_score += ma_alignment_score
        
        # 4. Volume Analysis
        if (latest['Volume'] > latest['Volume_SMA'] * 1.5 and 
            'OBV' in df.columns and 
            df['OBV'].tail(3).is_monotonic_increasing):
            base_signals.append("Strong Volume Confirmation")
            base_score += 2
        elif latest['Volume'] > latest['Volume_SMA']:
            base_signals.append("Above Average Volume")
            base_score += 1
        
        # 5. Williams %R Analysis
        if 'Williams_R' in latest and -50 <= latest['Williams_R'] <= -20:
            base_signals.append("Williams %R Bullish")
            base_score += 1
        elif 'Williams_R' in latest and latest['Williams_R'] > -10:
            base_signals.append("Williams %R Overbought")
            base_score -= 2
        
        # 6. Money Flow Index
        if 'MFI' in latest:
            if 20 <= latest['MFI'] <= 50:
                base_signals.append("MFI Oversold Recovery")
                base_score += 2
            elif latest['MFI'] > 80:
                base_signals.append("MFI Overbought")
                base_score -= 2
        
        # 7. Price Channel Analysis
        if 'PC_Upper' in latest and 'PC_Lower' in latest:
            pc_position = (latest['Close'] - latest['PC_Lower']) / (latest['PC_Upper'] - latest['PC_Lower'])
            if pc_position < 0.3:
                base_signals.append("Near Channel Support")
                base_score += 2
            elif pc_position > 0.8:
                base_signals.append("Near Channel Resistance")
                base_score -= 2
        
        # Cap base score
        base_score = min(base_score, TRADING_CONFIG['MAX_BASE_SCORE'])
        base_score = max(base_score, -TRADING_CONFIG['MAX_BASE_SCORE'])
        
        # Enhanced valuation analysis
        enhanced_valuation = self.calculate_enhanced_valuation_metrics()
        valuation_penalty = enhanced_valuation.get('combined_valuation_penalty', 0)
        
        # Cap valuation penalty
        valuation_penalty = max(valuation_penalty, TRADING_CONFIG['MIN_VALUATION_PENALTY'])
        
        # Combine all scores
        total_score = base_score + trend_score + momentum_score + volume_score + valuation_penalty
        
        # Cap the total score
        total_score = min(total_score, TRADING_CONFIG['MAX_SIGNAL_SCORE'])
        total_score = max(total_score, 0)
        
        # Combine all signals
        all_signals = base_signals + trend_signals + momentum_signals + volume_signals
        
        # Calculate entry price and trading plan
        entry_price = float(latest['Close'])
        trading_plan = None
        
        # Calculate entry analysis for all stocks
        entry_analysis = self.calculate_fair_value_entry_points(entry_price)
        
        # Determine if this is a good entry point
        is_good_entry = total_score >= TRADING_CONFIG['MIN_ENTRY_SCORE'] and valuation_penalty >= -3
        
        # Create detailed trading plan for BUY signals
        if is_good_entry:
            trading_plan = self.calculate_stop_loss_exit_points(entry_price)
        
        # Build technical data dictionary
        technical_data = {
            'RSI': float(latest['RSI']) if not pd.isna(latest['RSI']) else 0,
            'MACD': float(latest['MACD']) if not pd.isna(latest['MACD']) else 0,
            'MACD_Signal': float(latest['MACD_Signal']) if not pd.isna(latest['MACD_Signal']) else 0,
            'ADX': float(latest['ADX']) if not pd.isna(latest['ADX']) else 0,
            'SMA_200': float(latest['SMA_200']) if not pd.isna(latest['SMA_200']) else 0,
            'EMA_50': float(latest['EMA_50']) if not pd.isna(latest['EMA_50']) else 0,
            'BB_Position': float(((latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower'])) * 100) if not pd.isna(latest['BB_Lower']) else 0,
            'Volume_Ratio': float(latest['Volume'] / latest['Volume_SMA']) if not pd.isna(latest['Volume_SMA']) else 1,
            'Williams_R': float(latest['Williams_R']) if 'Williams_R' in latest and not pd.isna(latest['Williams_R']) else 0,
            'MFI': float(latest['MFI']) if 'MFI' in latest and not pd.isna(latest['MFI']) else 0,
            'CCI': float(latest['CCI']) if 'CCI' in latest and not pd.isna(latest['CCI']) else 0,
            'VWAP': float(latest['VWAP']) if 'VWAP' in latest and not pd.isna(latest['VWAP']) else 0,
            'ROC': float(latest['ROC']) if 'ROC' in latest and not pd.isna(latest['ROC']) else 0,
            'Position_52W': float(enhanced_valuation['position_52w']),
            'Distance_SMA200': float(enhanced_valuation['distance_from_sma200']),
            'Volatility_20D': float(enhanced_valuation['volatility_20d']),
            'Stoch_K': float(latest['Stoch_K']) if 'Stoch_K' in latest and not pd.isna(latest['Stoch_K']) else 0,
            'Stoch_D': float(latest['Stoch_D']) if 'Stoch_D' in latest and not pd.isna(latest['Stoch_D']) else 0,
            'Plus_DI': float(latest['Plus_DI']) if 'Plus_DI' in latest and not pd.isna(latest['Plus_DI']) else 0,
            'Minus_DI': float(latest['Minus_DI']) if 'Minus_DI' in latest and not pd.isna(latest['Minus_DI']) else 0
        }
        
        # *** KEY FIX: Calculate indicator scores ***
        indicator_scores = self.calculate_indicator_scores(technical_data)
        print(f"DEBUG: Calculated indicator scores for {self.symbol}: {len(indicator_scores)} indicators")
        
        # *** SINGLE RETURN STATEMENT WITH ALL DATA ***
        return {
            'symbol': self.symbol,
            'date': df.index[-1].strftime('%Y-%m-%d'),
            'current_price': entry_price,
            'signal_score': total_score,
            'entry_signal': is_good_entry,
            'signals': all_signals,
            'trading_plan': trading_plan,
            'entry_analysis': entry_analysis,
            'valuation_metrics': enhanced_valuation,
            'fundamental_analysis': {
                'pe_ratio': enhanced_valuation.get('pe_ratio'),
                'valuation_category': enhanced_valuation.get('valuation_category'),
                'valuation_score': enhanced_valuation.get('valuation_score'),
                'sector': enhanced_valuation.get('sector'),
                'industry': enhanced_valuation.get('industry'),
                'strengths': enhanced_valuation.get('valuation_strengths', []),
                'warnings': enhanced_valuation.get('valuation_warnings', [])
            },
            'recommendation_summary': self.generate_enhanced_recommendation_summary(
                total_score, is_good_entry, enhanced_valuation, latest, 
                base_score, trend_score, momentum_score, volume_score, valuation_penalty
            ),
            'score_breakdown': {
                'base_score': base_score,
                'trend_score': trend_score,
                'momentum_score': momentum_score,
                'volume_score': volume_score,
                'valuation_penalty': valuation_penalty,
                'total_score': total_score
            },
            'technical_data': technical_data,
            'indicator_scores': indicator_scores  # *** NOW INCLUDED! ***
        }

    def generate_enhanced_recommendation_summary(self, total_score, is_good_entry, valuation_data, latest, 
                                               base_score, trend_score, momentum_score, volume_score, valuation_penalty):
        """Generate enhanced recommendation summary with fundamental insights"""
        
        if is_good_entry:
            return {
                'type': 'BUY',
                'title': 'Strong Entry Signal',
                'summary': f'Score of {total_score}/20 with {valuation_data.get("valuation_category", "Unknown")} valuation indicates favorable conditions.',
                'key_points': [
                    f'Technical score above {TRADING_CONFIG["MIN_ENTRY_SCORE"]} threshold',
                    f'Fundamental valuation: {valuation_data.get("valuation_category", "Unknown")}',
                    'Multiple bullish indicators aligned'
                ],
                'valuation_insights': valuation_data.get('valuation_strengths', [])[:3]
            }
        
        # For non-buy signals, include fundamental context
        fundamental_category = valuation_data.get('valuation_category', 'Unknown')
        
        if total_score >= 5:
            key_issues = []
            
            if valuation_penalty <= -3:
                key_issues.append(f'Overvaluation concerns ({fundamental_category})')
            if base_score < 6:
                key_issues.append('Weak technical signals')
            if trend_score <= 1:
                key_issues.append('Poor trend alignment')
                
            summary = f'Mixed signals with {fundamental_category} valuation. Score: {total_score}/20.'
            
            return {
                'type': 'MONITOR',
                'title': 'Mixed Technical Picture',
                'summary': summary,
                'key_points': key_issues[:3],
                'what_to_watch': [
                    'Wait for technical score to reach 8+',
                    'Monitor for valuation improvement',
                    'Watch for trend confirmation'
                ],
                'valuation_insights': valuation_data.get('valuation_warnings', [])[:2]
            }
        
        else:
            major_issues = []
            
            if valuation_penalty <= -4:
                major_issues.append(f'Significantly overvalued ({fundamental_category})')
            if base_score <= 2:
                major_issues.append('Very weak technical signals')
            if trend_score <= 0:
                major_issues.append('Poor trend direction')
                
            return {
                'type': 'WAIT',
                'title': 'Poor Setup - Wait for Better Entry',
                'summary': f'Poor technical setup with {fundamental_category} valuation. Score: {total_score}/20.',
                'key_points': major_issues[:4],
                'what_to_watch': [
                    'Wait for major technical improvement',
                    'Monitor for valuation correction',
                    'Consider alternative investments'
                ],
                'valuation_insights': valuation_data.get('valuation_warnings', [])[:3]
            }

    def calculate_indicator_scores(self, latest_data):
            """Calculate individual indicator scores for color coding"""
            scores = {}
            
            # RSI Scoring
            rsi = latest_data.get('RSI', 50)
            if rsi <= 30:
                scores['RSI'] = {'value': rsi, 'score': 'bullish', 'signal': 'Oversold - Buy Signal'}
            elif rsi <= 45:
                scores['RSI'] = {'value': rsi, 'score': 'bullish', 'signal': 'Oversold Recovery'}
            elif rsi <= 55:
                scores['RSI'] = {'value': rsi, 'score': 'neutral', 'signal': 'Neutral Zone'}
            elif rsi <= 70:
                scores['RSI'] = {'value': rsi, 'score': 'neutral', 'signal': 'Bullish Territory'}
            elif rsi <= 80:
                scores['RSI'] = {'value': rsi, 'score': 'bearish', 'signal': 'Overbought Warning'}
            else:
                scores['RSI'] = {'value': rsi, 'score': 'bearish', 'signal': 'Extremely Overbought'}
            
            # MACD Scoring
            macd = latest_data.get('MACD', 0)
            macd_signal = latest_data.get('MACD_Signal', 0)
            macd_hist = latest_data.get('MACD_Histogram', 0)
            
            if macd > macd_signal and macd_hist > 0:
                scores['MACD'] = {'value': macd, 'score': 'bullish', 'signal': 'Bullish Momentum'}
            elif macd > macd_signal:
                scores['MACD'] = {'value': macd, 'score': 'neutral', 'signal': 'Above Signal Line'}
            elif macd < macd_signal and macd_hist < 0:
                scores['MACD'] = {'value': macd, 'score': 'bearish', 'signal': 'Bearish Momentum'}
            else:
                scores['MACD'] = {'value': macd, 'score': 'neutral', 'signal': 'Below Signal Line'}
            
            # ADX Scoring
            adx = latest_data.get('ADX', 0)
            plus_di = latest_data.get('Plus_DI', 0)
            minus_di = latest_data.get('Minus_DI', 0)
            
            if adx > 25:
                if plus_di > minus_di:
                    scores['ADX'] = {'value': adx, 'score': 'bullish', 'signal': 'Strong Uptrend'}
                else:
                    scores['ADX'] = {'value': adx, 'score': 'bearish', 'signal': 'Strong Downtrend'}
            elif adx > 20:
                if plus_di > minus_di:
                    scores['ADX'] = {'value': adx, 'score': 'neutral', 'signal': 'Moderate Uptrend'}
                else:
                    scores['ADX'] = {'value': adx, 'score': 'neutral', 'signal': 'Moderate Downtrend'}
            else:
                scores['ADX'] = {'value': adx, 'score': 'neutral', 'signal': 'Weak Trend'}
            
            # Williams %R Scoring
            williams_r = latest_data.get('Williams_R', -50)
            if williams_r >= -20:
                scores['Williams_R'] = {'value': williams_r, 'score': 'bearish', 'signal': 'Overbought'}
            elif williams_r >= -50:
                scores['Williams_R'] = {'value': williams_r, 'score': 'neutral', 'signal': 'Neutral'}
            elif williams_r >= -80:
                scores['Williams_R'] = {'value': williams_r, 'score': 'bullish', 'signal': 'Oversold'}
            else:
                scores['Williams_R'] = {'value': williams_r, 'score': 'bullish', 'signal': 'Deeply Oversold'}
            
            # Money Flow Index Scoring
            mfi = latest_data.get('MFI', 50)
            if mfi <= 20:
                scores['MFI'] = {'value': mfi, 'score': 'bullish', 'signal': 'Oversold'}
            elif mfi <= 40:
                scores['MFI'] = {'value': mfi, 'score': 'bullish', 'signal': 'Oversold Recovery'}
            elif mfi <= 60:
                scores['MFI'] = {'value': mfi, 'score': 'neutral', 'signal': 'Neutral'}
            elif mfi <= 80:
                scores['MFI'] = {'value': mfi, 'score': 'neutral', 'signal': 'Bullish Territory'}
            else:
                scores['MFI'] = {'value': mfi, 'score': 'bearish', 'signal': 'Overbought'}
            
            # Bollinger Bands Position Scoring
            bb_position = latest_data.get('BB_Position', 50)
            if bb_position <= 10:
                scores['BB_Position'] = {'value': bb_position, 'score': 'bullish', 'signal': 'Near Lower Band'}
            elif bb_position <= 30:
                scores['BB_Position'] = {'value': bb_position, 'score': 'bullish', 'signal': 'Lower Third'}
            elif bb_position <= 70:
                scores['BB_Position'] = {'value': bb_position, 'score': 'neutral', 'signal': 'Middle Range'}
            elif bb_position <= 90:
                scores['BB_Position'] = {'value': bb_position, 'score': 'bearish', 'signal': 'Upper Third'}
            else:
                scores['BB_Position'] = {'value': bb_position, 'score': 'bearish', 'signal': 'Near Upper Band'}
            
            # Volume Ratio Scoring
            volume_ratio = latest_data.get('Volume_Ratio', 1.0)
            if volume_ratio >= 2.0:
                scores['Volume_Ratio'] = {'value': volume_ratio, 'score': 'bullish', 'signal': 'Very High Volume'}
            elif volume_ratio >= 1.5:
                scores['Volume_Ratio'] = {'value': volume_ratio, 'score': 'bullish', 'signal': 'High Volume'}
            elif volume_ratio >= 1.0:
                scores['Volume_Ratio'] = {'value': volume_ratio, 'score': 'neutral', 'signal': 'Above Average'}
            elif volume_ratio >= 0.7:
                scores['Volume_Ratio'] = {'value': volume_ratio, 'score': 'neutral', 'signal': 'Below Average'}
            else:
                scores['Volume_Ratio'] = {'value': volume_ratio, 'score': 'bearish', 'signal': 'Low Volume'}
            
            # Moving Average Position Scoring
            sma_200_distance = latest_data.get('Distance_SMA200', 0)
            if sma_200_distance >= 10:
                scores['SMA_200_Position'] = {'value': sma_200_distance, 'score': 'bullish', 'signal': 'Well Above SMA200'}
            elif sma_200_distance >= 0:
                scores['SMA_200_Position'] = {'value': sma_200_distance, 'score': 'bullish', 'signal': 'Above SMA200'}
            elif sma_200_distance >= -5:
                scores['SMA_200_Position'] = {'value': sma_200_distance, 'score': 'neutral', 'signal': 'Near SMA200'}
            else:
                scores['SMA_200_Position'] = {'value': sma_200_distance, 'score': 'bearish', 'signal': 'Below SMA200'}
            
            # 52-Week Position Scoring
            position_52w = latest_data.get('Position_52W', 50)
            if position_52w >= 90:
                scores['Position_52W'] = {'value': position_52w, 'score': 'bearish', 'signal': 'Near 52W High'}
            elif position_52w >= 70:
                scores['Position_52W'] = {'value': position_52w, 'score': 'neutral', 'signal': 'Upper Range'}
            elif position_52w >= 30:
                scores['Position_52W'] = {'value': position_52w, 'score': 'neutral', 'signal': 'Middle Range'}
            elif position_52w >= 10:
                scores['Position_52W'] = {'value': position_52w, 'score': 'bullish', 'signal': 'Lower Range'}
            else:
                scores['Position_52W'] = {'value': position_52w, 'score': 'bullish', 'signal': 'Near 52W Low'}
            
            # Stochastic Scoring
            if 'Stoch_K' in latest_data and 'Stoch_D' in latest_data:
                stoch_k = latest_data['Stoch_K']
                stoch_d = latest_data['Stoch_D']
                
                if stoch_k <= 20 and stoch_d <= 20:
                    scores['Stochastic'] = {'value': f"{stoch_k:.1f}/{stoch_d:.1f}", 'score': 'bullish', 'signal': 'Oversold'}
                elif stoch_k >= 80 and stoch_d >= 80:
                    scores['Stochastic'] = {'value': f"{stoch_k:.1f}/{stoch_d:.1f}", 'score': 'bearish', 'signal': 'Overbought'}
                elif stoch_k > stoch_d:
                    scores['Stochastic'] = {'value': f"{stoch_k:.1f}/{stoch_d:.1f}", 'score': 'bullish', 'signal': 'Bullish Cross'}
                else:
                    scores['Stochastic'] = {'value': f"{stoch_k:.1f}/{stoch_d:.1f}", 'score': 'neutral', 'signal': 'Neutral'}
            
            # CCI Scoring
            if 'CCI' in latest_data:
                cci = latest_data['CCI']
                if cci >= 100:
                    scores['CCI'] = {'value': cci, 'score': 'bearish', 'signal': 'Overbought'}
                elif cci <= -100:
                    scores['CCI'] = {'value': cci, 'score': 'bullish', 'signal': 'Oversold'}
                elif cci > 0:
                    scores['CCI'] = {'value': cci, 'score': 'bullish', 'signal': 'Bullish'}
                else:
                    scores['CCI'] = {'value': cci, 'score': 'bearish', 'signal': 'Bearish'}
            
            # ROC Scoring
            if 'ROC' in latest_data:
                roc = latest_data['ROC']
                if roc >= 5:
                    scores['ROC'] = {'value': roc, 'score': 'bullish', 'signal': 'Strong Momentum'}
                elif roc >= 0:
                    scores['ROC'] = {'value': roc, 'score': 'bullish', 'signal': 'Positive Momentum'}
                elif roc >= -5:
                    scores['ROC'] = {'value': roc, 'score': 'bearish', 'signal': 'Negative Momentum'}
                else:
                    scores['ROC'] = {'value': roc, 'score': 'bearish', 'signal': 'Weak Momentum'}
            
            return scores