# technical_analyzer.py - Complete Technical Analysis Engine
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
        self.data = self.fetch_data(self.period)
        
    def fetch_data(self, period):
        """Fetch stock data from Yahoo Finance with improved error handling"""
        try:
            print(f"Fetching data for {self.symbol} with period {period}")
            ticker = yf.Ticker(self.symbol)
            
            # Try to get basic info first to validate symbol
            try:
                info = ticker.info
                if not info or 'symbol' not in info:
                    print(f"Symbol {self.symbol} may not be valid - no info available")
            except Exception as info_error:
                print(f"Warning: Could not fetch info for {self.symbol}: {info_error}")
            
            # Fetch historical data
            data = ticker.history(period=period)
            
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
            
            # Check for null values in recent data
            recent_data = data.tail(10)
            if recent_data[required_columns].isnull().any().any():
                print(f"Warning: Recent data contains null values for {self.symbol}")
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {self.symbol}: {e}")
            
            # Handle specific yfinance errors
            error_str = str(e).lower()
            if 'no data' in error_str or 'not found' in error_str:
                print(f"Symbol {self.symbol} not found in Yahoo Finance")
            elif 'connection' in error_str or 'timeout' in error_str:
                print(f"Network connection issue while fetching {self.symbol}")
            elif 'rate limit' in error_str:
                print(f"Rate limited while fetching {self.symbol}")
            
            return None
    
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
        """Calculate Average Directional Index (ADX)"""
        window = window or TRADING_CONFIG['ADX_PERIOD']
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        plus_dm[(plus_dm - minus_dm) < 0] = 0
        minus_dm[(minus_dm - plus_dm) < 0] = 0
        
        # Smoothed calculations
        tr_smooth = tr.rolling(window=window).mean()
        plus_dm_smooth = plus_dm.rolling(window=window).mean()
        minus_dm_smooth = minus_dm.rolling(window=window).mean()
        
        # Directional Indicators
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=window).mean()
        
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
        """Calculate valuation-based metrics using price action"""
        # Price relative to moving averages (valuation proxy)
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
        valuation = self.calculate_valuation_metrics()
        
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
        """Identify potential entry points based on comprehensive technical analysis"""
        df = self.get_all_indicators()
        
        # Get the latest data point
        latest = df.iloc[-1]
        recent = df.tail(TRADING_CONFIG['RECENT_DAYS'])
        
        # Calculate trend strength with score capping
        trend_score, trend_signals = self.calculate_trend_strength()
        trend_score = min(trend_score, TRADING_CONFIG['MAX_TREND_SCORE'])
        trend_score = max(trend_score, -TRADING_CONFIG['MAX_TREND_SCORE'])
        
        # Calculate momentum strength with score capping  
        momentum_score, momentum_signals = self.calculate_momentum_strength()
        momentum_score = min(momentum_score, TRADING_CONFIG['MAX_MOMENTUM_SCORE'])
        momentum_score = max(momentum_score, -TRADING_CONFIG['MAX_MOMENTUM_SCORE'])
        
        # Calculate volume strength with score capping
        volume_score, volume_signals = self.calculate_volume_strength()
        volume_score = min(volume_score, TRADING_CONFIG['MAX_VOLUME_SCORE'])
        volume_score = max(volume_score, 0)  # Volume score should not be negative
        valuation = self.calculate_valuation_metrics()
        
        # Base signal analysis with score capping
        base_signals = []
        base_score = 0
        
        # 1. MACD Analysis (More Strict)
        if (latest['MACD'] > latest['MACD_Signal'] and 
            recent['MACD'].iloc[-2] <= recent['MACD_Signal'].iloc[-2] and
            latest['MACD_Histogram'] > recent['MACD_Histogram'].iloc[-2]):
            base_signals.append("MACD Bullish Crossover")
            base_score += 3
        elif (latest['MACD'] > latest['MACD_Signal'] and 
              latest['MACD_Histogram'] > 0):
            base_signals.append("MACD Above Signal")
            base_score += 1
        
        # 2. RSI Analysis (More Conservative)
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
        
        # 4. Advanced Volume Analysis
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
        
        # Cap base score at maximum allowed
        base_score = min(base_score, TRADING_CONFIG['MAX_BASE_SCORE'])
        base_score = max(base_score, -TRADING_CONFIG['MAX_BASE_SCORE'])  # Also cap negative scores
        
        # 8. Valuation-based penalties with strict capping
        valuation_penalty = 0
        valuation_signals = []
        
        # Distance from SMA 200 penalty
        if valuation['distance_from_sma200'] > 25:
            valuation_penalty -= 3
            valuation_signals.append(f"Far above SMA200 ({valuation['distance_from_sma200']:.1f}%)")
        elif valuation['distance_from_sma200'] > 15:
            valuation_penalty -= 2
            valuation_signals.append(f"Extended above SMA200 ({valuation['distance_from_sma200']:.1f}%)")
        
        # 52-week position penalty
        if valuation['position_52w'] > 90:
            valuation_penalty -= 3
            valuation_signals.append(f"Near 52W high ({valuation['position_52w']:.1f}%)")
        elif valuation['position_52w'] > 80:
            valuation_penalty -= 2
            valuation_signals.append(f"High in 52W range ({valuation['position_52w']:.1f}%)")
        
        # Recent volatility check
        if valuation['volatility_20d'] > 50:  # High volatility
            valuation_penalty -= 1
            valuation_signals.append(f"High volatility ({valuation['volatility_20d']:.1f}%)")
        
        # Cap valuation penalty at minimum allowed
        valuation_penalty = max(valuation_penalty, TRADING_CONFIG['MIN_VALUATION_PENALTY'])
        
        # Combine all scores with proper capping
        total_score = base_score + trend_score + momentum_score + volume_score + valuation_penalty
        
        # Cap the score at maximum 20 points
        total_score = min(total_score, TRADING_CONFIG['MAX_SIGNAL_SCORE'])
        
        # Ensure minimum score is 0
        total_score = max(total_score, 0)
        
        # Combine all signals
        all_signals = base_signals + trend_signals + momentum_signals + volume_signals + valuation_signals
        
        # Calculate entry price and all analysis
        entry_price = float(latest['Close'])
        trading_plan = None
        
        # ALWAYS calculate entry analysis for all stocks (not just buy signals)
        entry_analysis = self.calculate_fair_value_entry_points(entry_price)
        
        # Determine if this is a good entry point (use capped score)
        is_good_entry = total_score >= TRADING_CONFIG['MIN_ENTRY_SCORE'] and valuation_penalty >= -3
        
        # Only create detailed trading plan for actual BUY signals
        if is_good_entry:
            trading_plan = self.calculate_stop_loss_exit_points(entry_price)
        
        return {
            'symbol': self.symbol,
            'date': df.index[-1].strftime('%Y-%m-%d'),
            'current_price': entry_price,
            'signal_score': total_score,
            'entry_signal': is_good_entry,
            'signals': all_signals,
            'trading_plan': trading_plan,
            'entry_analysis': entry_analysis,
            'valuation_metrics': valuation,
            'recommendation_summary': self.generate_recommendation_summary(total_score, is_good_entry, valuation, latest, base_score, trend_score, momentum_score, volume_score, valuation_penalty),
            'score_breakdown': {
                'base_score': base_score,
                'trend_score': trend_score,
                'momentum_score': momentum_score,
                'volume_score': volume_score,
                'valuation_penalty': valuation_penalty,
                'total_score': total_score
            },
            'technical_data': {
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
                'Position_52W': float(valuation['position_52w']),
                'Distance_SMA200': float(valuation['distance_from_sma200']),
                'Volatility_20D': float(valuation['volatility_20d'])
            }
        }
    
    def generate_recommendation_summary(self, total_score, is_good_entry, valuation, latest, base_score, trend_score, momentum_score, volume_score, valuation_penalty):
        """Generate a brief summary explaining the recommendation"""
        
        if is_good_entry:
            return {
                'type': 'BUY',
                'title': 'Strong Entry Signal',
                'summary': f'Score of {total_score}/20 indicates favorable technical conditions for entry.',
                'key_points': [
                    f'Technical score above {TRADING_CONFIG["MIN_ENTRY_SCORE"]} threshold',
                    'Multiple bullish indicators aligned',
                    'Risk/reward ratio supports entry'
                ]
            }
        
        # For MONITOR signals (5-7 points)
        elif total_score >= 5:
            key_issues = []
            strengths = []
            
            # Analyze what's holding it back
            if base_score < 6:
                key_issues.append('Weak base technical signals')
            else:
                strengths.append('Decent technical foundation')
                
            if trend_score <= 1:
                key_issues.append('Poor trend alignment')
            elif trend_score >= 3:
                strengths.append('Good trend structure')
                
            if momentum_score <= 2:
                key_issues.append('Lacking momentum')
            elif momentum_score >= 4:
                strengths.append('Good momentum')
                
            if valuation_penalty <= -3:
                key_issues.append('Overvaluation concerns')
            elif valuation_penalty >= -1:
                strengths.append('Fair valuation')
            
            # RSI specific issues
            if latest['RSI'] > 75:
                key_issues.append('RSI overbought (>75)')
            elif latest['RSI'] < 25:
                key_issues.append('RSI oversold (<25)')
                
            # Position in range
            if valuation['position_52w'] > 85:
                key_issues.append('Near 52-week highs')
            
            summary = f'Mixed signals with score of {total_score}/20. '
            if len(strengths) > 0:
                summary += f'Shows {", ".join(strengths[:2])} but '
            summary += f'held back by {", ".join(key_issues[:2])}.'
            
            return {
                'type': 'MONITOR',
                'title': 'Mixed Technical Picture',
                'summary': summary,
                'key_points': key_issues[:3],
                'what_to_watch': [
                    'Wait for score to reach 8+ for entry',
                    'Monitor for trend improvement',
                    'Watch for momentum pickup'
                ]
            }
        
        # For WAIT signals (0-4 points)  
        else:
            major_issues = []
            
            # Identify major problems
            if base_score <= 2:
                major_issues.append('Very weak technical signals')
            if trend_score <= 0:
                major_issues.append('Poor trend direction')
            if momentum_score <= 1:
                major_issues.append('No momentum support')
            if valuation_penalty <= -4:
                major_issues.append('Significantly overvalued')
                
            # Specific technical issues
            if latest['RSI'] > 80:
                major_issues.append('Extremely overbought (RSI >80)')
            elif latest['RSI'] < 20:
                major_issues.append('Extremely oversold (RSI <20)')
                
            if valuation['distance_from_sma200'] > 30:
                major_issues.append(f'{valuation["distance_from_sma200"]:.0f}% above SMA 200')
                
            if valuation['position_52w'] > 95:
                major_issues.append('At 52-week highs')
            elif valuation['position_52w'] < 5:
                major_issues.append('Near 52-week lows')
            
            # MACD issues
            if latest['MACD'] < latest['MACD_Signal'] and latest['MACD'] < 0:
                major_issues.append('MACD bearish divergence')
                
            summary = f'Poor setup with score of {total_score}/20. '
            summary += f'Multiple issues: {", ".join(major_issues[:3])}.'
            
            return {
                'type': 'WAIT',
                'title': 'Poor Technical Setup',
                'summary': summary,
                'key_points': major_issues[:4],
                'what_to_watch': [
                    'Wait for major technical improvement',
                    'Look for oversold bounce opportunity',
                    'Monitor for trend reversal signals',
                    'Consider alternative investments'
                ]
            }