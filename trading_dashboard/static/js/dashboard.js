// static/js/dashboard.js - Dashboard JavaScript

var currentActiveBtn = null;
var currentSymbol = null;

function initializeDashboard(symbols) {
    var grid = document.getElementById('symbolsGrid');
    symbols.forEach(function(symbol) {
        var btn = document.createElement('button');
        btn.className = 'symbol-btn';
        btn.textContent = symbol;
        btn.onclick = function() {
            analyzeSymbol(symbol, btn);
        };
        grid.appendChild(btn);
    });
}

function analyzeSymbol(symbol, btn) {
    // Update active button
    if (currentActiveBtn) {
        currentActiveBtn.classList.remove('active');
    }
    btn.classList.add('active');
    currentActiveBtn = btn;
    currentSymbol = symbol;
    
    // Show loading state
    showLoading();
    hideResults();
    hideError();
    
    fetch('/analyze/' + symbol)
        .then(function(response) {
            return response.json();
        })
        .then(function(data) {
            if (data.error) {
                showError(data.error);
            } else {
                displayResults(data);
            }
        })
        .catch(function(error) {
            showError('Network error: ' + error.message);
        })
        .finally(function() {
            hideLoading();
        });
}

function displayResults(data) {
    var resultDiv = document.getElementById('analysisResult');
    
    var signalInfo = getSignalInfo(data);
    var signalClass = signalInfo.signalClass;
    var signalText = signalInfo.signalText;
    var scoreClass = getScoreClass(data.signal_score);
    var tech = data.technical_data;
    
    var tradingPlanHTML = '';
    if (data.trading_plan && data.entry_signal) {
        tradingPlanHTML = generateTradingPlanHTML(data);
    }
    
    var entryAnalysisHTML = generateEntryAnalysisHTML(data);
    var metricsHTML = generateMetricsHTML(data, tech);
    var signalsHTML = '';
    
    if (data.signals && data.signals.length > 0) {
        signalsHTML = generateSignalsHTML(data.signals);
    }
    
    resultDiv.innerHTML = 
        '<div class="result-header">' +
            '<div>' +
                '<h2 class="symbol-title">' + data.symbol + '</h2>' +
                '<p>Price: $' + data.current_price.toFixed(2) + ' | Date: ' + data.date + '</p>' +
            '</div>' +
            '<div style="display: flex; align-items: center;">' +
                '<div class="score-circle ' + scoreClass + '">' +
                    data.signal_score + '/20' +
                '</div>' +
                '<div class="signal-badge ' + signalClass + '">' +
                    signalText +
                '</div>' +
            '</div>' +
        '</div>' +
        '<div class="metrics-grid">' + metricsHTML + '</div>' +
        entryAnalysisHTML +
        tradingPlanHTML +
        signalsHTML;
    
    resultDiv.style.display = 'block';
    resultDiv.classList.add('analysis-result');
}

function generateEntryAnalysisHTML(data) {
    var analysis = data.entry_analysis;
    var breakdown = data.score_breakdown;
    var valuation = data.valuation_metrics;
    
    var breakdownHTML = 
        '<div class="score-breakdown">' +
            '<h4>📊 Score Breakdown (' + data.signal_score + '/20 points):</h4>' +
            '<div class="breakdown-grid">' +
                '<div class="breakdown-item">' +
                    '<span class="breakdown-label">Base Signals:</span>' +
                    '<span class="breakdown-value ' + (breakdown.base_score >= 0 ? 'positive' : 'negative') + '">' + breakdown.base_score + '</span>' +
                '</div>' +
                '<div class="breakdown-item">' +
                    '<span class="breakdown-label">Trend Strength:</span>' +
                    '<span class="breakdown-value ' + (breakdown.trend_score >= 0 ? 'positive' : 'negative') + '">' + breakdown.trend_score + '</span>' +
                '</div>' +
                '<div class="breakdown-item">' +
                    '<span class="breakdown-label">Momentum:</span>' +
                    '<span class="breakdown-value ' + (breakdown.momentum_score >= 0 ? 'positive' : 'negative') + '">' + breakdown.momentum_score + '</span>' +
                '</div>' +
                '<div class="breakdown-item">' +
                    '<span class="breakdown-label">Volume:</span>' +
                    '<span class="breakdown-value ' + (breakdown.volume_score >= 0 ? 'positive' : 'negative') + '">' + breakdown.volume_score + '</span>' +
                '</div>' +
                '<div class="breakdown-item">' +
                    '<span class="breakdown-label">Valuation:</span>' +
                    '<span class="breakdown-value ' + (breakdown.valuation_penalty >= 0 ? 'positive' : 'negative') + '">' + breakdown.valuation_penalty + '</span>' +
                '</div>' +
                '<div class="breakdown-item total">' +
                    '<span class="breakdown-label">Total Score:</span>' +
                    '<span class="breakdown-value ' + (breakdown.total_score >= 8 ? 'positive' : 'negative') + '">' + breakdown.total_score + '</span>' +
                '</div>' +
            '</div>' +
        '</div>';
    
    var entryRecommendationHTML = '';
    
    if (data.entry_signal) {
        if (analysis && analysis.is_overpriced) {
            entryRecommendationHTML = generateOverpricedWarning(analysis);
        } else {
            entryRecommendationHTML = 
                '<div class="fair-value-notice">' +
                    '<h4>✅ Good Entry Signal</h4>' +
                    '<p>Technical analysis suggests current levels are reasonable for entry with score of ' + data.signal_score + '/20.</p>' +
                '</div>';
        }
    } else {
        entryRecommendationHTML = generateWaitEntryRecommendations(data, valuation);
    }
    
    return '<div class="entry-analysis-section">' + breakdownHTML + entryRecommendationHTML + '</div>';
}

function generateOverpricedWarning(analysis) {
    var reasonTags = '';
    if (analysis.reasons) {
        analysis.reasons.forEach(function(reason) {
            reasonTags += '<span class="reason-tag">' + reason + '</span>';
        });
    }
    
    var suggestedEntriesHTML = '';
    if (analysis.suggested_entries && analysis.suggested_entries.length > 0) {
        suggestedEntriesHTML = '<div class="suggested-entries">' +
            '<h5>💡 Consider Better Entry Points:</h5>';
        
        analysis.suggested_entries.forEach(function(entry) {
            suggestedEntriesHTML += 
                '<div class="entry-suggestion">' +
                    '<span class="entry-level">' + entry.level + '</span>' +
                    '<span class="entry-price">$' + entry.price.toFixed(2) + '</span>' +
                    '<span class="entry-discount">' + entry.discount.toFixed(1) + '% discount</span>' +
                '</div>';
        });
        
        suggestedEntriesHTML += '</div>';
    }
    
    return '<div class="overpriced-warning">' +
        '<h4>⚠️ Stock May Be Overpriced</h4>' +
        '<div class="overpriced-reasons">' + reasonTags + '</div>' +
        suggestedEntriesHTML +
        '</div>';
}

function generateWaitEntryRecommendations(data, valuation) {
    var currentPrice = data.current_price;
    var tech = data.technical_data;
    var entryLevels = [];
    
    if (tech.EMA_50 > 0 && tech.EMA_50 < currentPrice) {
        var discount = ((currentPrice - tech.EMA_50) / currentPrice) * 100;
        entryLevels.push({
            level: 'EMA 50 Support',
            price: tech.EMA_50,
            discount: discount,
            probability: 'Medium'
        });
    }
    
    if (tech.SMA_200 > 0 && tech.SMA_200 < currentPrice) {
        var discount = ((currentPrice - tech.SMA_200) / currentPrice) * 100;
        entryLevels.push({
            level: 'SMA 200 Support',
            price: tech.SMA_200,
            discount: discount,
            probability: 'High'
        });
    }
    
    if (tech.VWAP > 0 && tech.VWAP < currentPrice) {
        var discount = ((currentPrice - tech.VWAP) / currentPrice) * 100;
        entryLevels.push({
            level: 'VWAP Fair Value',
            price: tech.VWAP,
            discount: discount,
            probability: 'Medium'
        });
    }
    
    var percentageEntries = [
        { level: '5% Pullback', price: currentPrice * 0.95, discount: 5, probability: 'High' },
        { level: '10% Pullback', price: currentPrice * 0.90, discount: 10, probability: 'Medium' },
        { level: '15% Pullback', price: currentPrice * 0.85, discount: 15, probability: 'Low' }
    ];
    
    var allEntries = entryLevels.concat(percentageEntries);
    allEntries = allEntries.filter(function(entry) {
        return entry.discount >= 2;
    });
    
    allEntries.sort(function(a, b) {
        return a.discount - b.discount;
    });
    
    allEntries = allEntries.slice(0, 4);
    
    var signalType, signalIcon, signalMessage;
    if (data.signal_score >= 5) {
        signalType = 'hold-signal';
        signalIcon = '⚠️';
        signalMessage = 'Hold/Monitor - Watch for Better Entry';
    } else {
        signalType = 'wait-signal';
        signalIcon = '❌';
        signalMessage = 'Wait - Poor Setup Currently';
    }
    
    var waitReasons = [];
    if (data.signal_score < 8) waitReasons.push('Low signal score (' + data.signal_score + '/20)');
    if (valuation.position_52w > 80) waitReasons.push('High in 52W range (' + valuation.position_52w.toFixed(1) + '%)');
    if (valuation.distance_from_sma200 > 15) waitReasons.push('Extended above SMA200 (' + valuation.distance_from_sma200.toFixed(1) + '%)');
    if (tech.RSI > 70) waitReasons.push('RSI overbought (' + tech.RSI.toFixed(1) + ')');
    if (valuation.volatility_20d > 40) waitReasons.push('High volatility (' + valuation.volatility_20d.toFixed(1) + '%)');
    
    var waitReasonsHTML = '';
    if (waitReasons.length > 0) {
        waitReasonsHTML = '<div class="wait-reasons">' +
            '<h5>Why Wait:</h5>' +
            '<div class="reason-tags">';
        
        waitReasons.forEach(function(reason) {
            waitReasonsHTML += '<span class="reason-tag">' + reason + '</span>';
        });
        
        waitReasonsHTML += '</div></div>';
    }
    
    var entryLevelsHTML = '<div class="suggested-entry-levels">' +
        '<h5>💡 Consider Entry At These Levels:</h5>' +
        '<div class="entry-levels-grid">';
    
    allEntries.forEach(function(entry) {
        entryLevelsHTML += 
            '<div class="entry-level-item">' +
                '<div class="level-info">' +
                    '<span class="level-name">' + entry.level + '</span>' +
                    '<span class="probability-badge ' + entry.probability.toLowerCase() + '">' + entry.probability + ' Probability</span>' +
                '</div>' +
                '<div class="level-price">$' + entry.price.toFixed(2) + '</div>' +
                '<div class="level-discount">' + entry.discount.toFixed(1) + '% below current</div>' +
            '</div>';
    });
    
    entryLevelsHTML += '</div></div>';
    
    var monitoringHTML = 
        '<div class="monitoring-tips">' +
            '<h5>📋 What to Monitor:</h5>' +
            '<ul>' +
                '<li>Wait for price to reach suggested entry levels</li>' +
                '<li>Watch for RSI to drop below 50 for better momentum</li>' +
                '<li>Look for increased volume on any pullback</li>' +
                '<li>Monitor support at key moving averages</li>' +
                '<li>Re-evaluate if fundamental news changes the picture</li>' +
            '</ul>' +
        '</div>';
    
    return '<div class="wait-recommendation ' + signalType + '">' +
        '<h4>' + signalIcon + ' ' + signalMessage + '</h4>' +
        waitReasonsHTML +
        entryLevelsHTML +
        monitoringHTML +
        '</div>';
}

function generateTradingPlanHTML(data) {
    var plan = data.trading_plan;
    var analysis = data.entry_analysis;
    
    var buyEntryAnalysisHTML = '';
    if (analysis && analysis.is_overpriced) {
        buyEntryAnalysisHTML = generateOverpricedWarning(analysis);
    }
    
    return '<div class="trading-plan-section">' +
        '<h3 class="plan-title">📋 Trading Plan (3-4 Month Hold)</h3>' +
        buyEntryAnalysisHTML +
        '<div class="plan-grid">' +
            '<div class="plan-card entry-card">' +
                '<h4>🎯 Entry Point</h4>' +
                '<div class="plan-value">$' + data.current_price.toFixed(2) + '</div>' +
                '<div class="plan-note">Current market price</div>' +
            '</div>' +
            '<div class="plan-card stop-card">' +
                '<h4>🛑 Stop Loss</h4>' +
                '<div class="plan-value">$' + plan.stop_loss.recommended.toFixed(2) + '</div>' +
                '<div class="plan-note">Risk: ' + plan.stop_loss.risk_percent.toFixed(1) + '%</div>' +
            '</div>' +
            '<div class="plan-card target-card">' +
                '<h4>🚀 Primary Target (27.5%)</h4>' +
                '<div class="plan-value">$' + plan.exit_targets.Primary_Target.toFixed(2) + '</div>' +
                '<div class="plan-note">Expected in 3-4 months</div>' +
            '</div>' +
            '<div class="plan-card ratio-card">' +
                '<h4>⚖️ Risk/Reward Ratio</h4>' +
                '<div class="plan-value">1:' + plan.risk_reward_ratio.toFixed(1) + '</div>' +
                '<div class="plan-note">' + (plan.risk_reward_ratio >= 2 ? 'Good ratio' : 'Consider risk') + '</div>' +
            '</div>' +
        '</div>' +
        '<div class="profit-targets">' +
            '<h4>📈 Partial Profit Targets:</h4>' +
            '<div class="targets-grid">' +
                '<div class="target-item">' +
                    '<span class="target-label">Target 1 (15%)</span>' +
                    '<span class="target-price">$' + plan.exit_targets.Target_1.toFixed(2) + '</span>' +
                    '<span class="target-action">Sell 1/3</span>' +
                '</div>' +
                '<div class="target-item">' +
                    '<span class="target-label">Target 2 (22%)</span>' +
                    '<span class="target-price">$' + plan.exit_targets.Target_2.toFixed(2) + '</span>' +
                    '<span class="target-action">Sell 1/3</span>' +
                '</div>' +
                '<div class="target-item">' +
                    '<span class="target-label">Target 3 (27.5%)</span>' +
                    '<span class="target-price">$' + plan.exit_targets.Target_3.toFixed(2) + '</span>' +
                    '<span class="target-action">Sell Final 1/3</span>' +
                '</div>' +
            '</div>' +
        '</div>' +
        '<div class="risk-management">' +
            '<h4>⚠️ Risk Management:</h4>' +
            '<ul>' +
                '<li>Maximum risk: ' + plan.position_sizing.max_risk_per_trade + '% of total portfolio on this trade</li>' +
                '<li>Review position weekly for any changes in trend</li>' +
                '<li>Move stop loss to breakeven after 15% gain</li>' +
                '<li>Consider trailing stop loss after 20% gain</li>' +
                '<li>Exit immediately if stop loss is hit - no exceptions</li>' +
                '<li>Monitor key technical levels for trend changes</li>' +
            '</ul>' +
        '</div>' +
        '</div>';
}

function generateMetricsHTML(data, tech) {
    var valuation = data.valuation_metrics;
    
    return '<div class="metric-card">' +
            '<div class="metric-label">RSI (14)</div>' +
            '<div class="metric-value ' + getIndicatorClass(tech.RSI, 30, 70) + '">' + tech.RSI.toFixed(1) + '</div>' +
        '</div>' +
        '<div class="metric-card">' +
            '<div class="metric-label">MACD</div>' +
            '<div class="metric-value">' + tech.MACD.toFixed(3) + '</div>' +
        '</div>' +
        '<div class="metric-card">' +
            '<div class="metric-label">ADX Trend</div>' +
            '<div class="metric-value">' + tech.ADX.toFixed(1) + '</div>' +
        '</div>' +
        '<div class="metric-card">' +
            '<div class="metric-label">Williams %R</div>' +
            '<div class="metric-value ' + getIndicatorClass(tech.Williams_R, -80, -20, true) + '">' + tech.Williams_R.toFixed(1) + '</div>' +
        '</div>' +
        '<div class="metric-card">' +
            '<div class="metric-label">Money Flow Index</div>' +
            '<div class="metric-value ' + getIndicatorClass(tech.MFI, 20, 80) + '">' + tech.MFI.toFixed(1) + '</div>' +
        '</div>' +
        '<div class="metric-card">' +
            '<div class="metric-label">vs SMA 200</div>' +
            '<div class="metric-value ' + (valuation.distance_from_sma200 > 20 ? 'warning' : '') + '">' + valuation.distance_from_sma200.toFixed(1) + '%</div>' +
        '</div>' +
        '<div class="metric-card">' +
            '<div class="metric-label">52W Position</div>' +
            '<div class="metric-value ' + (valuation.position_52w > 85 ? 'warning' : '') + '">' + valuation.position_52w.toFixed(1) + '%</div>' +
        '</div>' +
        '<div class="metric-card">' +
            '<div class="metric-label">Volatility (20D)</div>' +
            '<div class="metric-value ' + (valuation.volatility_20d > 40 ? 'warning' : '') + '">' + valuation.volatility_20d.toFixed(1) + '%</div>' +
        '</div>' +
        '<div class="metric-card">' +
            '<div class="metric-label">VWAP</div>' +
            '<div class="metric-value">$' + tech.VWAP.toFixed(2) + '</div>' +
        '</div>' +
        '<div class="metric-card">' +
            '<div class="metric-label">Rate of Change</div>' +
            '<div class="metric-value ' + (tech.ROC > 0 ? 'positive' : 'negative') + '">' + tech.ROC.toFixed(1) + '%</div>' +
        '</div>' +
        '<div class="metric-card">' +
            '<div class="metric-label">Volume Ratio</div>' +
            '<div class="metric-value">' + ((data.current_price/tech.EMA_50-1)*100).toFixed(1) + '%</div>' +
        '</div>' +
        '<div class="metric-card">' +
            '<div class="metric-label">CCI</div>' +
            '<div class="metric-value ' + getIndicatorClass(tech.CCI, -100, 100) + '">' + tech.CCI.toFixed(1) + '</div>' +
        '</div>';
}

function getIndicatorClass(value, oversold, overbought, inverse) {
    if (inverse) {
        if (value <= oversold) return 'positive';
        if (value >= overbought) return 'warning';
        return '';
    } else {
        if (value <= oversold) return 'warning';
        if (value >= overbought) return 'negative';
        return '';
    }
}

function generateSignalsHTML(signals) {
    var signalsHTML = '<div class="signals-section">' +
        '<h3 class="signals-title">Bullish Signals Detected:</h3>' +
        '<div class="signals-list">';
    
    signals.forEach(function(signal) {
        signalsHTML += '<span class="signal-tag">' + signal + '</span>';
    });
    
    signalsHTML += '</div></div>';
    return signalsHTML;
}

function getSignalInfo(data) {
    if (data.entry_signal) {
        return {
            signalClass: 'signal-buy',
            signalText: '✅ BUY SIGNAL'
        };
    } else if (data.signal_score >= 5) {
        return {
            signalClass: 'signal-hold',
            signalText: '⚠️ HOLD/MONITOR'
        };
    } else {
        return {
            signalClass: 'signal-wait',
            signalText: '❌ WAIT'
        };
    }
}

function getScoreClass(score) {
    if (score >= 12) return 'score-high';
    if (score >= 8) return 'score-medium';
    if (score >= 5) return 'score-low';
    return 'score-very-low';
}

function showLoading() {
    document.getElementById('loadingMessage').style.display = 'block';
}

function hideLoading() {
    document.getElementById('loadingMessage').style.display = 'none';
}

function showResults() {
    document.getElementById('analysisResult').style.display = 'block';
}

function hideResults() {
    document.getElementById('analysisResult').style.display = 'none';
}

function showError(message) {
    var errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
}

function hideError() {
    document.getElementById('errorMessage').style.display = 'none';
}