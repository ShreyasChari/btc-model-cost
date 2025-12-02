/**
 * Bitcoin Cost Model - Frontend Application
 */

// Global state
let surfaceData = null;
let charts = {};
let volPredictorData = null;

// ============================================================================
// API Functions
// ============================================================================

async function fetchAPI(endpoint, options = {}) {
    const response = await fetch(`/api${endpoint}`, options);
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'API Error');
    }
    return response.json();
}

async function fetchDeribitData() {
    showStatus('Fetching live data from Deribit...', 'loading');
    try {
        const result = await fetchAPI('/surface/fetch-deribit');
        showStatus(`Loaded ${result.maturities} maturities, ${result.options_count} options`, 'success');
        await loadSurfaceData();
        
        // Load comparison (only available with live data)
        showComparisonSection();
        const threshold = parseFloat(document.getElementById('threshold-select').value);
        await loadComparison(threshold);
    } catch (error) {
        showStatus(`Error: ${error.message}`, 'error');
    }
}

async function submitManualSurface(data) {
    showStatus('Building surface...', 'loading');
    try {
        const result = await fetchAPI('/surface/manual', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        showStatus(`Surface built with ${result.maturities} maturities`, 'success');
        await loadSurfaceData();
        closeModal();
    } catch (error) {
        showStatus(`Error: ${error.message}`, 'error');
    }
}

async function loadSurfaceData() {
    try {
        surfaceData = await fetchAPI('/surface');
        updateMarketInfo();
        updateBreakEvensTable();
        renderCharts();
        await loadArbitrage();
        await loadTrades();
        await loadVolPredictor();
        showSections();
    } catch (error) {
        showStatus(`Error loading surface: ${error.message}`, 'error');
    }
}

async function loadArbitrage() {
    try {
        const opps = await fetchAPI('/arbitrage');
        renderOpportunities(opps);
    } catch (error) {
        console.error('Error loading arbitrage:', error);
    }
}

async function loadTrades() {
    try {
        const trades = await fetchAPI('/trades');
        renderTrades(trades);
    } catch (error) {
        console.error('Error loading trades:', error);
    }
}

async function loadComparison(threshold = 0.01) {
    try {
        const result = await fetchAPI(`/comparison?threshold=${threshold}`);
        renderComparisonSummary(result.summary);
        renderSignalsTable(result.signals);
        await loadHeatmap();
    } catch (error) {
        console.error('Error loading comparison:', error);
        // Comparison only works with live data
        document.getElementById('comparison-section').classList.add('hidden');
    }
}

async function loadHeatmap() {
    try {
        const data = await fetchAPI('/comparison/heatmap');
        renderHeatmap(data);
    } catch (error) {
        console.error('Error loading heatmap:', error);
    }
}

async function calculateTheta() {
    const strike = parseFloat(document.getElementById('theta-strike').value);
    const maturity = parseFloat(document.getElementById('theta-maturity').value) / 12;
    
    try {
        const result = await fetchAPI(`/theta-decomposition?strike=${strike}&maturity=${maturity}`);
        renderThetaResult(result);
    } catch (error) {
        document.getElementById('theta-result').innerHTML = 
            `<div class="error">Error: ${error.message}</div>`;
    }
}

// ============================================================================
// Vol Predictor Functions
// ============================================================================

async function loadVolPredictor() {
    const note = document.getElementById('vol-model-note');
    try {
        volPredictorData = await fetchAPI('/volpredictor');
        renderVolPredictor(volPredictorData);
    } catch (error) {
        console.error('Error loading Vol Predictor:', error);
        if (note) {
            note.textContent = `Unable to load DVOL analytics: ${error.message}`;
        }
    }
}

function renderVolPredictor(data) {
    if (!data || !data.latest || !data.stats) return;
    
    const latest = data.latest;
    const stats = data.stats;
    
    setText('predicted-vol', formatVol(latest.predicted));
    setText('actual-vol', formatVol(latest.actual));
    setText('vol-forecast', formatVol(latest.forecast_next));
    
    const errorValue = latest.actual - latest.predicted;
    const errorEl = document.getElementById('vol-error');
    if (errorEl) {
        errorEl.textContent = `${errorValue >= 0 ? '+' : ''}${(errorValue * 100).toFixed(2)} pts`;
        setDeltaClass(errorEl, errorValue);
    }
    
    setText('vol-rmse', formatVol(stats.rmse));
    setText('vol-mae', formatVol(stats.mae));
    setText('vol-bias', `${stats.bias >= 0 ? '+' : ''}${(stats.bias * 100).toFixed(2)} pts`);
    setText('vol-hit-rate', `${(stats.direction_hit_rate * 100).toFixed(0)}%`);
    setText('vol-corr', stats.correlation ? stats.correlation.toFixed(2) : '--');
    setText('vol-volatility', formatVol(stats.vol_of_vol));
    setText('vol-regime', stats.regime);
    setText('vol-sample', `${stats.sample_size} bars`);
    
    const note = document.getElementById('vol-model-note');
    if (note && data.model) {
        const updatedAt = data.refreshed_at || latest.timestamp;
        const updatedDisplay = updatedAt ? new Date(updatedAt).toLocaleString() : latest.timestamp;
        note.textContent = `Model: ${data.model.type} • α=${data.model.alpha} • Window=${data.model.window} • Updated ${updatedDisplay}`;
    }
    
    const volPanel = document.getElementById('volpredictor-panel');
    if (volPanel && !volPanel.hidden) {
        renderVolPredictorChart(data.timeseries || []);
    } else {
        charts.volpredictor = null;
    }
    renderVolInsights(data.insights || []);
}

function renderVolPredictorChart(timeseries) {
    const container = document.getElementById('volpredictor-plot');
    if (!container || !timeseries || timeseries.length === 0) return;
    if (!window.Plotly) {
        console.warn('Plotly library not loaded; cannot render DVOL chart');
        return;
    }
    
    container.style.width = '100%';
    container.style.height = '350px';
    
    const timestamps = timeseries.map(point => new Date(point.timestamp));
    const actualSeries = timeseries.map(point => point.actual * 100);
    const predictedSeries = timeseries.map(point => point.predicted * 100);
    
    const data = [
        {
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Actual DVOL',
            x: timestamps,
            y: actualSeries,
            line: { color: '#ff6384', width: 2 },
            marker: { color: '#ff6384', size: 6, symbol: 'circle' }
        },
        {
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Predicted DVOL',
            x: timestamps,
            y: predictedSeries,
            line: { color: '#36a2eb', width: 2, dash: 'dash' },
            marker: { color: '#36a2eb', size: 6, symbol: 'circle-open' }
        }
    ];
    
    const layout = {
        title: { text: 'Predicted vs Actual DVOL', font: { color: '#cbd5e0' } },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        margin: { l: 60, r: 20, t: 50, b: 80 },
        xaxis: {
            title: 'Timestamp',
            tickangle: 45,
            color: '#94a3b8',
            gridcolor: 'rgba(148,163,184,0.2)'
        },
        yaxis: {
            title: 'Vol Points (%)',
            color: '#94a3b8',
            gridcolor: 'rgba(148,163,184,0.2)'
        },
        legend: { orientation: 'h', x: 0.5, y: -0.2, xanchor: 'center', font: { color: '#cbd5e0' } }
    };
    
    const config = { responsive: true, displaylogo: false };
    
    if (charts.volpredictor) {
        Plotly.react(container, data, layout, config);
    } else {
        Plotly.newPlot(container, data, layout, config).then(() => {
            charts.volpredictor = container.id;
            Plotly.Plots.resize(container);
        });
    }
}

function renderVolInsights(insights) {
    const list = document.getElementById('vol-insights');
    if (!list) return;
    
    list.innerHTML = '';
    if (!insights || insights.length === 0) {
        const item = document.createElement('li');
        item.textContent = 'No analytics available yet.';
        list.appendChild(item);
        return;
    }
    
    insights.forEach(text => {
        const li = document.createElement('li');
        li.textContent = text;
        list.appendChild(li);
    });
}

function formatVol(value, decimals = 2) {
    if (value === null || value === undefined || Number.isNaN(value)) {
        return '--';
    }
    return `${value.toFixed(decimals)}%`;
}

function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

function setDeltaClass(element, delta) {
    element.classList.remove('positive', 'negative', 'neutral');
    if (delta > 0.0001) {
        element.classList.add('positive');
    } else if (delta < -0.0001) {
        element.classList.add('negative');
    } else {
        element.classList.add('neutral');
    }
}

// ============================================================================
// UI Functions
// ============================================================================

function showStatus(message, type) {
    const status = document.getElementById('status');
    status.textContent = message;
    status.className = `status ${type}`;
}

function showSections() {
    ['market-info', 'break-evens-section', 'charts-section', 
     'arbitrage-section', 'trades-section', 'theta-section'].forEach(id => {
        document.getElementById(id).classList.remove('hidden');
    });
}

function showComparisonSection() {
    document.getElementById('comparison-section').classList.remove('hidden');
}

function openModal() {
    document.getElementById('manual-modal').classList.remove('hidden');
}

function closeModal() {
    document.getElementById('manual-modal').classList.add('hidden');
}

function updateMarketInfo() {
    if (!surfaceData) return;
    
    document.getElementById('spot-price').textContent = `$${surfaceData.spot.toLocaleString()}`;
    
    // Get regime from break-evens analysis
    const be = surfaceData.break_evens;
    if (be && be.length >= 2) {
        const frontVol = be[0].atm_vol;
        const backVol = be[be.length - 1].atm_vol;
        
        if (frontVol > backVol + 0.02) {
            document.getElementById('term-structure').textContent = 'Backwardation';
            document.getElementById('term-structure').className = 'value warning';
        } else if (backVol > frontVol + 0.02) {
            document.getElementById('term-structure').textContent = 'Contango';
            document.getElementById('term-structure').className = 'value';
        } else {
            document.getElementById('term-structure').textContent = 'Flat';
            document.getElementById('term-structure').className = 'value';
        }
        
        const avgCorr = be.reduce((sum, b) => sum + b.spot_vol_corr, 0) / be.length;
        if (avgCorr < -0.1) {
            document.getElementById('skew-regime').textContent = 'Negative (Puts Exp)';
            document.getElementById('skew-regime').className = 'value warning';
            document.getElementById('regime').textContent = 'Distressed';
            document.getElementById('regime').className = 'value danger';
        } else if (avgCorr > 0.05) {
            document.getElementById('skew-regime').textContent = 'Positive';
            document.getElementById('skew-regime').className = 'value success';
            document.getElementById('regime').textContent = 'Complacent';
            document.getElementById('regime').className = 'value';
        } else {
            document.getElementById('skew-regime').textContent = 'Neutral';
            document.getElementById('skew-regime').className = 'value';
            document.getElementById('regime').textContent = 'Normal';
            document.getElementById('regime').className = 'value';
        }
    }
}

function updateBreakEvensTable() {
    const tbody = document.getElementById('break-evens-body');
    tbody.innerHTML = '';
    
    if (!surfaceData || !surfaceData.break_evens) return;
    
    surfaceData.break_evens.forEach(be => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${be.maturity_label}</td>
            <td>${(be.atm_vol * 100).toFixed(1)}%</td>
            <td>${(be.gamma_be * 100).toFixed(2)}%</td>
            <td>${(be.volga_be * 100).toFixed(2)}%</td>
            <td>${(be.vanna_be * 100).toFixed(3)}%</td>
            <td>${be.spot_vol_corr.toFixed(2)}</td>
        `;
        tbody.appendChild(row);
    });
}

// ============================================================================
// Chart Functions
// ============================================================================

function renderCharts() {
    if (!surfaceData || !surfaceData.surface_data) return;
    
    render3DSurface();
    renderSmileChart();
    renderTermChart();
    renderBreakevenChart();
}

function render3DSurface() {
    const sd = surfaceData.surface_data;
    
    const data = [{
        type: 'surface',
        x: sd.maturities.map(m => m * 12),  // Convert to months
        y: sd.moneyness,
        z: sd.vols.map(row => row.map(v => v * 100)),  // Convert to %
        colorscale: 'Viridis',
        colorbar: { title: 'IV (%)' }
    }];
    
    const layout = {
        title: 'Implied Volatility Surface',
        scene: {
            xaxis: { title: 'Maturity (months)' },
            yaxis: { title: 'Moneyness (K/S)' },
            zaxis: { title: 'IV (%)' }
        },
        margin: { l: 0, r: 0, t: 40, b: 0 }
    };
    
    Plotly.newPlot('surface-3d', data, layout);
}

function renderSmileChart() {
    const ctx = document.getElementById('smile-chart').getContext('2d');
    const sd = surfaceData.surface_data;
    
    if (charts.smile) charts.smile.destroy();
    
    const colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF'];
    
    const datasets = sd.maturities.map((mat, i) => ({
        label: `${(mat * 12).toFixed(0)}M`,
        data: sd.moneyness.map((m, j) => ({
            x: m,
            y: sd.vols[i][j] * 100
        })),
        borderColor: colors[i % colors.length],
        fill: false,
        tension: 0.3
    }));
    
    charts.smile = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            plugins: {
                title: { display: true, text: 'Volatility Smile by Maturity' }
            },
            scales: {
                x: { 
                    type: 'linear',
                    title: { display: true, text: 'Moneyness (K/S)' }
                },
                y: { 
                    title: { display: true, text: 'IV (%)' }
                }
            }
        }
    });
}

function renderTermChart() {
    const ctx = document.getElementById('term-chart').getContext('2d');
    const be = surfaceData.break_evens;
    
    if (charts.term) charts.term.destroy();
    
    charts.term = new Chart(ctx, {
        type: 'line',
        data: {
            labels: be.map(b => b.maturity_label),
            datasets: [{
                label: 'ATM Vol (%)',
                data: be.map(b => b.atm_vol * 100),
                borderColor: '#36A2EB',
                fill: false
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: { display: true, text: 'ATM Volatility Term Structure' }
            },
            scales: {
                y: { title: { display: true, text: 'IV (%)' } }
            }
        }
    });
}

function renderBreakevenChart() {
    const ctx = document.getElementById('breakeven-chart').getContext('2d');
    const be = surfaceData.break_evens;
    
    if (charts.breakeven) charts.breakeven.destroy();
    
    charts.breakeven = new Chart(ctx, {
        type: 'line',
        data: {
            labels: be.map(b => b.maturity_label),
            datasets: [
                {
                    label: 'Gamma BE (%)',
                    data: be.map(b => b.gamma_be * 100),
                    borderColor: '#FF6384',
                    fill: false
                },
                {
                    label: 'Volga BE (%)',
                    data: be.map(b => b.volga_be * 100),
                    borderColor: '#4BC0C0',
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: { display: true, text: 'Cost Model Break-Evens' }
            },
            scales: {
                y: { title: { display: true, text: 'Daily Break-Even (%)' } }
            }
        }
    });
}

// ============================================================================
// Arbitrage & Trade Rendering
// ============================================================================

function renderOpportunities(opps) {
    const container = document.getElementById('opportunities-list');
    container.innerHTML = '';
    
    // Summary
    const summary = opps.summary;
    const summaryHtml = `
        <div class="regime-summary ${summary.regime}">
            <h3>Market Regime: ${summary.regime.toUpperCase()}</h3>
            <p>Term Structure: ${summary.term_structure} | Skew: ${summary.skew_regime}</p>
        </div>
    `;
    container.innerHTML += summaryHtml;
    
    // Term structure opportunities
    if (opps.term_structure.length > 0) {
        container.innerHTML += '<h4>📊 Term Structure</h4>';
        opps.term_structure.forEach(opp => {
            container.innerHTML += renderOpportunityCard(opp);
        });
    }
    
    // Skew opportunities
    if (opps.skew.length > 0) {
        container.innerHTML += '<h4>📈 Skew</h4>';
        opps.skew.forEach(opp => {
            container.innerHTML += renderOpportunityCard(opp);
        });
    }
    
    // Smile opportunities
    if (opps.smile.length > 0) {
        container.innerHTML += '<h4>🦋 Smile</h4>';
        opps.smile.forEach(opp => {
            container.innerHTML += renderOpportunityCard(opp);
        });
    }
    
    if (opps.term_structure.length === 0 && opps.skew.length === 0 && opps.smile.length === 0) {
        container.innerHTML += '<p class="no-opps">No significant opportunities detected</p>';
    }
}

function renderOpportunityCard(opp) {
    const convictionClass = opp.conviction === 'HIGH' ? 'high' : opp.conviction === 'MEDIUM' ? 'medium' : 'low';
    return `
        <div class="opportunity-card ${convictionClass}">
            <div class="opp-header">
                <span class="conviction ${convictionClass}">${opp.conviction}</span>
                <span class="opp-type">${opp.type}</span>
            </div>
            <p class="opp-rationale">${opp.rationale}</p>
        </div>
    `;
}

function renderTrades(trades) {
    const container = document.getElementById('trades-list');
    container.innerHTML = '';
    
    if (!trades || trades.length === 0) {
        container.innerHTML = '<p class="no-trades">No trade recommendations available</p>';
        return;
    }
    
    trades.forEach((trade, i) => {
        const convictionClass = trade.conviction === 'HIGH' ? 'high' : 'medium';
        const html = `
            <div class="trade-card ${convictionClass}">
                <div class="trade-header">
                    <h4>Trade #${i + 1}: ${trade.strategy}</h4>
                    <span class="conviction ${convictionClass}">${trade.conviction}</span>
                </div>
                <p class="trade-rationale">${trade.rationale}</p>
                
                <div class="trade-legs">
                    <h5>Execution:</h5>
                    ${trade.legs.map(leg => `
                        <div class="leg ${leg.action.toLowerCase()}">
                            <span class="action">${leg.action}</span>
                            <span class="instrument">${leg.instrument}</span>
                            <span class="price">$${leg.price.toLocaleString(undefined, {maximumFractionDigits: 0})}</span>
                            ${leg.iv ? `<span class="iv">IV: ${(leg.iv * 100).toFixed(1)}%</span>` : ''}
                        </div>
                    `).join('')}
                </div>
                
                ${trade.net_debit !== undefined ? `
                    <div class="trade-summary">
                        Net ${trade.net_debit > 0 ? 'Debit' : 'Credit'}: 
                        <strong>$${Math.abs(trade.net_debit).toLocaleString(undefined, {maximumFractionDigits: 0})}</strong>
                    </div>
                ` : ''}
                ${trade.net_credit !== undefined ? `
                    <div class="trade-summary">
                        Net Credit: <strong>$${trade.net_credit.toLocaleString(undefined, {maximumFractionDigits: 0})}</strong>
                    </div>
                ` : ''}
                
                <div class="trade-greeks">
                    <h5>Greek Profile:</h5>
                    <ul>
                        ${Object.entries(trade.greeks).map(([k, v]) => `<li><strong>${k}:</strong> ${v}</li>`).join('')}
                    </ul>
                </div>
                
                <div class="trade-scenarios">
                    <span class="win">✅ Win: ${trade.scenarios.win}</span>
                    <span class="lose">❌ Lose: ${trade.scenarios.lose}</span>
                </div>
            </div>
        `;
        container.innerHTML += html;
    });
}

function renderThetaResult(result) {
    const container = document.getElementById('theta-result');
    
    const html = `
        <div class="theta-result-card">
            <div class="theta-header">
                <span>${result.option_type.toUpperCase()}</span>
                <span>K=${result.strike.toLocaleString()} | T=${(result.maturity * 12).toFixed(1)}M</span>
            </div>
            
            <div class="theta-info">
                <div class="info-row">
                    <span>IV:</span>
                    <span>${(result.implied_vol * 100).toFixed(1)}%</span>
                </div>
                <div class="info-row">
                    <span>Price:</span>
                    <span>$${result.price.toLocaleString(undefined, {maximumFractionDigits: 0})}</span>
                </div>
                <div class="info-row">
                    <span>Daily θ (BS):</span>
                    <span>$${result.bs_theta.toFixed(2)}</span>
                </div>
            </div>
            
            <div class="theta-decomp">
                <h5>Theta Decomposition:</h5>
                <div class="decomp-bar">
                    <div class="gamma-bar" style="width: ${result.gamma_pct}%">
                        Γ ${result.gamma_pct.toFixed(0)}%
                    </div>
                    <div class="vanna-bar" style="width: ${result.vanna_pct}%">
                        Va ${result.vanna_pct.toFixed(0)}%
                    </div>
                    <div class="volga-bar" style="width: ${result.volga_pct}%">
                        Vo ${result.volga_pct.toFixed(0)}%
                    </div>
                </div>
                <div class="decomp-details">
                    <div>Gamma Cost: $${result.gamma_cost.toFixed(2)}/day</div>
                    <div>Vanna Cost: $${result.vanna_cost.toFixed(2)}/day</div>
                    <div>Volga Cost: $${result.volga_cost.toFixed(2)}/day</div>
                </div>
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}

// ============================================================================
// Comparison Rendering Functions
// ============================================================================

function renderComparisonSummary(summary) {
    const container = document.getElementById('comparison-summary');
    
    const html = `
        <div class="info-grid">
            <div class="info-card">
                <span class="label">Total Options</span>
                <span class="value">${summary.total_options}</span>
            </div>
            <div class="info-card">
                <span class="label">Rich (Sell)</span>
                <span class="value danger">${summary.rich_count}</span>
            </div>
            <div class="info-card">
                <span class="label">Cheap (Buy)</span>
                <span class="value success">${summary.cheap_count}</span>
            </div>
            <div class="info-card">
                <span class="label">Avg Spread</span>
                <span class="value">${summary.avg_abs_spread_bps.toFixed(0)} bps</span>
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}

function renderSignalsTable(signals) {
    const tbody = document.getElementById('signals-body');
    tbody.innerHTML = '';
    
    if (!signals || signals.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" class="no-data">No significant mispricings found</td></tr>';
        return;
    }
    
    signals.forEach(s => {
        const signalClass = s.signal === 'RICH' ? 'rich' : s.signal === 'CHEAP' ? 'cheap' : '';
        const spreadColor = s.spread > 0 ? 'var(--accent-red)' : 'var(--accent-green)';
        
        const row = document.createElement('tr');
        row.className = signalClass;
        row.innerHTML = `
            <td class="instrument">${s.instrument || '-'}</td>
            <td>${s.option_type || '-'}</td>
            <td>$${s.strike.toLocaleString()}</td>
            <td>${s.maturity_label}</td>
            <td>${(s.market_iv * 100).toFixed(1)}%</td>
            <td>${(s.model_iv * 100).toFixed(1)}%</td>
            <td style="color: ${spreadColor}; font-weight: bold;">
                ${s.spread > 0 ? '+' : ''}${(s.spread * 100).toFixed(2)}%
                <small>(${s.spread_bps.toFixed(0)} bps)</small>
            </td>
            <td>
                <span class="signal-badge ${s.signal.toLowerCase()}">${s.signal}</span>
            </td>
        `;
        tbody.appendChild(row);
    });
}

function renderHeatmap(data) {
    if (!data || !data.spreads || data.spreads.length === 0) {
        return;
    }
    
    // Convert spreads to percentage and handle nulls
    const z = data.spreads.map(row => 
        row.map(v => v !== null ? v * 100 : null)
    );
    
    const trace = {
        type: 'heatmap',
        x: data.moneyness,
        y: data.maturities.map(m => `${(m * 12).toFixed(0)}M`),
        z: z,
        colorscale: [
            [0, '#17bf63'],      // Green (cheap)
            [0.5, '#ffffff'],    // White (fair)
            [1, '#e0245e']       // Red (rich)
        ],
        zmid: 0,
        colorbar: {
            title: 'Spread (%)',
            titleside: 'right'
        },
        hoverongaps: false,
        hovertemplate: 'Moneyness: %{x:.2f}<br>Maturity: %{y}<br>Spread: %{z:.2f}%<extra></extra>'
    };
    
    const layout = {
        title: 'Market vs Model IV Spread',
        xaxis: { title: 'Moneyness (K/S)' },
        yaxis: { title: 'Maturity' },
        margin: { l: 60, r: 20, t: 40, b: 50 }
    };
    
    Plotly.newPlot('heatmap-chart', [trace], layout);
}

// ============================================================================
// Event Listeners
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Kick off predictive analytics fetch immediately (server keeps it warm)
    loadVolPredictor();
    
    // Fetch Deribit button
    document.getElementById('btn-fetch-deribit').addEventListener('click', fetchDeribitData);
    
    // Manual input button
    document.getElementById('btn-manual-input').addEventListener('click', openModal);
    
    // Close modal
    document.getElementById('btn-close-modal').addEventListener('click', closeModal);
    
    // Add maturity button
    document.getElementById('btn-add-maturity').addEventListener('click', () => {
        const container = document.getElementById('maturities-container');
        const rows = container.querySelectorAll('.maturity-row');
        const newIndex = rows.length;
        
        const newRow = document.createElement('div');
        newRow.className = 'maturity-row';
        newRow.dataset.index = newIndex;
        newRow.innerHTML = `
            <input type="number" placeholder="Months" value="${(newIndex + 1) * 3}" step="1" class="mat-months">
            <input type="number" placeholder="ATM Vol %" value="45" step="1" class="mat-vol">
            <input type="number" placeholder="Skew" value="-0.04" step="0.01" class="mat-skew">
            <input type="number" placeholder="Curv" value="0.30" step="0.01" class="mat-curv">
            <button type="button" class="btn-remove" onclick="this.parentElement.remove()">×</button>
        `;
        container.appendChild(newRow);
    });
    
    // Manual form submit
    document.getElementById('manual-form').addEventListener('submit', (e) => {
        e.preventDefault();
        
        const spot = parseFloat(document.getElementById('spot-input').value);
        const rows = document.querySelectorAll('.maturity-row');
        
        const maturities = [];
        rows.forEach(row => {
            const months = parseFloat(row.querySelector('.mat-months').value);
            const vol = parseFloat(row.querySelector('.mat-vol').value) / 100;
            const skew = parseFloat(row.querySelector('.mat-skew').value);
            const curv = parseFloat(row.querySelector('.mat-curv').value);
            
            if (months > 0 && vol > 0) {
                maturities.push({
                    maturity: months / 12,
                    atm_vol: vol,
                    skew: skew,
                    curvature: curv
                });
            }
        });
        
        submitManualSurface({ spot, maturities });
    });
    
    // Theta calculator
    document.getElementById('btn-calc-theta').addEventListener('click', calculateTheta);
    
    // Comparison controls
    document.getElementById('threshold-select').addEventListener('change', async (e) => {
        const threshold = parseFloat(e.target.value);
        await loadComparison(threshold);
    });
    
    document.getElementById('btn-refresh-comparison').addEventListener('click', async () => {
        const threshold = parseFloat(document.getElementById('threshold-select').value);
        await loadComparison(threshold);
    });
    
    // Click outside modal to close
    document.getElementById('manual-modal').addEventListener('click', (e) => {
        if (e.target.id === 'manual-modal') {
            closeModal();
        }
    });

    setupMarketTabs();
});

function setupMarketTabs() {
    const buttons = document.querySelectorAll('#market-info .tab-btn');
    const panels = document.querySelectorAll('#market-info .tab-panel');
    
    function activateTab(targetId) {
        buttons.forEach(btn => {
            const isActive = btn.dataset.tab === targetId;
            btn.classList.toggle('active', isActive);
        });
        panels.forEach(panel => {
            const isMatch = panel.id === targetId;
            panel.classList.toggle('active', isMatch);
            panel.hidden = !isMatch;
            panel.style.display = isMatch ? '' : 'none';
        });
        if (targetId === 'volpredictor-panel') {
            requestAnimationFrame(() => {
                if (volPredictorData && volPredictorData.timeseries) {
                    renderVolPredictorChart(volPredictorData.timeseries);
                } else if (charts.volpredictor && window.Plotly) {
                    const el = document.getElementById(charts.volpredictor);
                    if (el) {
                        Plotly.Plots.resize(el);
                    }
                }
            });
        }
    }
    
    const initial = document.querySelector('#market-info .tab-btn.active');
    if (initial) {
        activateTab(initial.dataset.tab);
    }
    
    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            activateTab(btn.dataset.tab);
        });
    });
}
