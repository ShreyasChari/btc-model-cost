import React, { useState, useMemo, useCallback, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, Area, ComposedChart, Bar, BarChart, Cell, ScatterChart, Scatter, ZAxis } from 'recharts';

// ============================================================
// COST MODEL MATHEMATICS
// Based on Gilbert Eid's framework
// Master equation: θ = Γ × Ω_G + Va × Ω_Va + Vo × Ω_Vo
// ============================================================

const normalCDF = (x) => {
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x) / Math.sqrt(2);
  const t = 1.0 / (1.0 + p * x);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return 0.5 * (1.0 + sign * y);
};

const normalPDF = (x) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);

// Extended Greeks including Vanna and Volga
const calculateFullGreeks = (S, K, T, r, sigma, optionType = 'call') => {
  if (T <= 0.001 || sigma <= 0.001) {
    const intrinsic = optionType === 'call' ? Math.max(S - K, 0) : Math.max(K - S, 0);
    return { 
      delta: intrinsic > 0 ? (optionType === 'call' ? 1 : -1) : 0, 
      gamma: 0, theta: 0, vega: 0, vanna: 0, volga: 0, price: intrinsic,
      d1: 0, d2: 0
    };
  }
  
  const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
  const d2 = d1 - sigma * Math.sqrt(T);
  const nd1 = normalCDF(d1);
  const nd2 = normalCDF(d2);
  const npd1 = normalPDF(d1);
  const sqrtT = Math.sqrt(T);
  
  let delta, theta, price;
  if (optionType === 'call') {
    delta = nd1;
    price = S * nd1 - K * Math.exp(-r * T) * nd2;
    theta = (-(S * npd1 * sigma) / (2 * sqrtT) - r * K * Math.exp(-r * T) * nd2) / 365;
  } else {
    delta = nd1 - 1;
    price = K * Math.exp(-r * T) * normalCDF(-d2) - S * normalCDF(-d1);
    theta = (-(S * npd1 * sigma) / (2 * sqrtT) + r * K * Math.exp(-r * T) * normalCDF(-d2)) / 365;
  }
  
  const gamma = npd1 / (S * sigma * sqrtT);
  const vega = S * npd1 * sqrtT / 100; // per 1% vol move
  
  // Vanna: d(delta)/d(sigma) = d(vega)/d(S) = -d1 * npd1 / sigma
  // Or: vega/S * (1 - d1/(sigma*sqrt(T)))
  const vanna = -npd1 * d2 / sigma;
  
  // Volga: d(vega)/d(sigma) = vega * d1 * d2 / sigma
  const volga = vega * d1 * d2 / sigma;
  
  return { delta, gamma, theta, vega, vanna, volga, price, d1, d2 };
};

// Cost Model: Calculate Omega values (cost of carry for each Greek)
const calculateCostModelOmegas = (S, T, sigma, volOfVol = 0.6, spotVolCorr = -0.5) => {
  // Omega_Gamma: Cost of gamma carry (realized variance)
  // For ATM: Ω_G ≈ 0.5 * S² * σ² / T (annualized)
  // Daily: Ω_G ≈ 0.5 * S² * σ² / 365
  const omegaGamma = 0.5 * sigma * sigma / 365; // per unit gamma, per day

  // Omega_Vanna: Cost of skew carry
  // Related to correlation between spot and vol
  // Ω_Va ≈ ρ * σ_S * σ_σ where ρ is spot-vol correlation
  // For BTC, typically negative (vol rises when price falls)
  const omegaVanna = spotVolCorr * sigma * volOfVol / 365;

  // Omega_Volga: Cost of smile carry (vol of vol)
  // Ω_Vo ≈ 0.5 * σ_σ² (variance of implied vol)
  // For BTC, vol of vol is typically 50-100% annualized
  const omegaVolga = 0.5 * volOfVol * volOfVol / 365;

  return { omegaGamma, omegaVanna, omegaVolga };
};

// Decompose theta into Cost Model components
const decomposeThetaCostModel = (greeks, S, T, sigma, volOfVol = 0.6, spotVolCorr = -0.5) => {
  const omegas = calculateCostModelOmegas(S, T, sigma, volOfVol, spotVolCorr);
  
  // θ = Γ × Ω_G + Va × Ω_Va + Vo × Ω_Vo
  const gammaCost = greeks.gamma * S * S * omegas.omegaGamma;
  const vannaCost = greeks.vanna * S * omegas.omegaVanna;
  const volgaCost = greeks.volga * omegas.omegaVolga;
  
  const totalCostModel = gammaCost + vannaCost + volgaCost;
  
  return {
    gammaCost,      // Cost from gamma (volatility carry)
    vannaCost,      // Cost from vanna (skew carry)
    volgaCost,      // Cost from volga (smile carry)
    totalCostModel,
    bsTheta: greeks.theta,
    residual: greeks.theta - totalCostModel,
    omegas
  };
};

// Calculate theoretical price using Cost Model
const costModelPrice = (S, K, T, r, sigma, optionType, marketSkew = null) => {
  const greeks = calculateFullGreeks(S, K, T, r, sigma, optionType);
  const decomposition = decomposeThetaCostModel(greeks, S, T, sigma);
  
  // If we have market skew data, adjust the price
  // Model price = BS price + skew adjustment + smile adjustment
  let modelPrice = greeks.price;
  
  if (marketSkew) {
    // Adjust for skew: price += Vanna * skew_adjustment
    const moneyness = Math.log(K / S) / (sigma * Math.sqrt(T));
    const skewAdjustment = marketSkew.slope * moneyness * greeks.vanna;
    modelPrice += skewAdjustment;
  }
  
  return {
    bsPrice: greeks.price,
    modelPrice,
    greeks,
    decomposition
  };
};

// Parse Deribit instrument name
const parseInstrument = (name) => {
  const parts = name.toUpperCase().split('-');
  if (parts.length < 4) return null;
  
  const expiry = parts[1];
  const strike = parseFloat(parts[2]);
  const optionType = parts[3] === 'C' ? 'call' : 'put';
  
  const months = { JAN: 0, FEB: 1, MAR: 2, APR: 3, MAY: 4, JUN: 5, JUL: 6, AUG: 7, SEP: 8, OCT: 9, NOV: 10, DEC: 11 };
  const day = parseInt(expiry.substring(0, 2));
  const month = months[expiry.substring(2, 5)];
  const year = 2000 + parseInt(expiry.substring(5, 7));
  const expiryDate = new Date(year, month, day);
  const now = new Date();
  const dte = Math.max(0, Math.ceil((expiryDate - now) / (1000 * 60 * 60 * 24)));
  
  return { strike, optionType, dte, expiry };
};

// Generate random return with optional rough volatility
const generateDailyReturn = (annualizedVol, hurstParam = 0.4) => {
  const dailyVol = annualizedVol / Math.sqrt(365);
  const u1 = Math.random();
  const u2 = Math.random();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  
  // For rough volatility, we could add persistence, but keeping simple for now
  return z * dailyVol;
};

// Generate vol move (for vanna/volga P&L)
const generateVolMove = (volOfVol, spotReturn, correlation = -0.7) => {
  const u1 = Math.random();
  const u2 = Math.random();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  
  // Correlated vol move
  const independentZ = z;
  const correlatedZ = correlation * (spotReturn / Math.abs(spotReturn || 1)) + 
                      Math.sqrt(1 - correlation * correlation) * independentZ;
  
  return correlatedZ * volOfVol / Math.sqrt(365);
};

// ============================================================
// MAIN COMPONENT
// ============================================================

export default function GammaVarianceTrackerWithCostModel() {
  // Mode
  const [mode, setMode] = useState('simulation');
  const [showCostModel, setShowCostModel] = useState(true);
  
  // Market state
  const [btcPrice, setBtcPrice] = useState(100000);
  const [impliedVol, setImpliedVol] = useState(50);
  const [realizedVolAssumption, setRealizedVolAssumption] = useState(55);
  const [volOfVol, setVolOfVol] = useState(60); // Vol of vol for BTC
  const [spotVolCorr, setSpotVolCorr] = useState(-0.5); // Spot-vol correlation
  
  // DVOL prediction integration
  const [predictedDvol, setPredictedDvol] = useState(52);
  const [currentDvol, setCurrentDvol] = useState(48);
  
  // Simulation position
  const [strike, setStrike] = useState(100000);
  const [dte, setDte] = useState(30);
  const [contracts, setContracts] = useState(1);
  const [optionType, setOptionType] = useState('call');
  
  // Portfolio positions
  const [positions, setPositions] = useState([
    { instrument: 'BTC-27DEC24-100000-C', size: 1, direction: 'buy', iv: 50, marketPrice: 0.05 }
  ]);
  
  // Simulation state
  const [simulationData, setSimulationData] = useState([]);
  const [isRunning, setIsRunning] = useState(false);
  const [currentDay, setCurrentDay] = useState(0);
  const [speed, setSpeed] = useState(150);
  const [simDays, setSimDays] = useState(30);
  
  // Calculate portfolio Greeks with Cost Model decomposition
  const portfolioAnalysis = useMemo(() => {
    let totalDelta = 0, totalGamma = 0, totalTheta = 0, totalVega = 0;
    let totalVanna = 0, totalVolga = 0;
    let totalGammaCost = 0, totalVannaCost = 0, totalVolgaCost = 0;
    let ivSum = 0, ivCount = 0;
    let modelVsMarket = [];
    
    const positionsToAnalyze = mode === 'simulation' 
      ? [{ 
          instrument: `BTC-SIM-${strike}-${optionType === 'call' ? 'C' : 'P'}`,
          size: contracts, 
          direction: 'buy', 
          iv: impliedVol,
          strike, 
          optionType, 
          dte,
          parsed: { strike, optionType, dte }
        }]
      : positions.map(p => ({ ...p, parsed: parseInstrument(p.instrument) }));
    
    positionsToAnalyze.forEach(pos => {
      if (!pos.parsed) return;
      
      const T = pos.parsed.dte / 365;
      const posIV = (pos.iv || impliedVol) / 100;
      const greeks = calculateFullGreeks(btcPrice, pos.parsed.strike, T, 0.05, posIV, pos.parsed.optionType);
      const decomposition = decomposeThetaCostModel(greeks, btcPrice, T, posIV);
      const sign = pos.direction === 'buy' ? 1 : -1;
      
      totalDelta += greeks.delta * pos.size * sign;
      totalGamma += greeks.gamma * pos.size * sign;
      totalTheta += greeks.theta * pos.size * sign;
      totalVega += greeks.vega * pos.size * sign;
      totalVanna += greeks.vanna * pos.size * sign;
      totalVolga += greeks.volga * pos.size * sign;
      
      totalGammaCost += decomposition.gammaCost * pos.size * sign * btcPrice;
      totalVannaCost += decomposition.vannaCost * pos.size * sign * btcPrice;
      totalVolgaCost += decomposition.volgaCost * pos.size * sign * btcPrice;
      
      ivSum += pos.iv || impliedVol;
      ivCount++;
      
      // Model vs Market comparison
      if (pos.marketPrice) {
        const modelPriceResult = costModelPrice(btcPrice, pos.parsed.strike, T, 0.05, posIV, pos.parsed.optionType);
        const marketPriceUsd = pos.marketPrice * btcPrice;
        // modelPriceResult.modelPrice is already in USD (calculated with S=btcPrice), no need to multiply again
        const modelPriceUsd = modelPriceResult.modelPrice;
        const mispricing = (modelPriceUsd - marketPriceUsd) / marketPriceUsd * 100;
        
        modelVsMarket.push({
          instrument: pos.instrument,
          strike: pos.parsed.strike,
          dte: pos.parsed.dte,
          marketPrice: marketPriceUsd,
          modelPrice: modelPriceUsd,
          mispricing,
          signal: mispricing > 5 ? 'SELL' : mispricing < -5 ? 'BUY' : 'FAIR',
          greeks,
          decomposition
        });
      }
    });
    
    const avgIV = ivCount > 0 ? ivSum / ivCount : impliedVol;
    const dollarGamma = 0.5 * totalGamma * btcPrice * btcPrice * 0.0001;
    const dailyThetaUsd = totalTheta * btcPrice;
    const breakEvenMove = totalGamma > 0 ? Math.sqrt(2 * Math.abs(totalTheta) / (totalGamma * btcPrice)) * 100 : 0;
    
    // DVOL-based trade signal
    const volSpread = predictedDvol - currentDvol;
    const volSignal = volSpread > 3 ? 'LONG VOL' : volSpread < -3 ? 'SHORT VOL' : 'NEUTRAL';
    
    return {
      totalDelta, totalGamma, totalTheta, totalVega, totalVanna, totalVolga,
      totalGammaCost, totalVannaCost, totalVolgaCost,
      dollarGamma, dailyThetaUsd, breakEvenMove, avgIV,
      modelVsMarket,
      volSignal, volSpread,
      // Cost model percentages
      gammaCostPct: Math.abs(totalGammaCost) / (Math.abs(totalGammaCost) + Math.abs(totalVannaCost) + Math.abs(totalVolgaCost) + 0.0001) * 100,
      vannaCostPct: Math.abs(totalVannaCost) / (Math.abs(totalGammaCost) + Math.abs(totalVannaCost) + Math.abs(totalVolgaCost) + 0.0001) * 100,
      volgaCostPct: Math.abs(totalVolgaCost) / (Math.abs(totalGammaCost) + Math.abs(totalVannaCost) + Math.abs(totalVolgaCost) + 0.0001) * 100
    };
  }, [mode, btcPrice, strike, dte, impliedVol, contracts, optionType, positions, predictedDvol, currentDvol]);
  
  // Initialize simulation
  const initializeSimulation = useCallback(() => {
    setSimulationData([{
      day: 0,
      spot: btcPrice,
      vol: impliedVol,
      dailyReturnPct: 0,
      volChange: 0,
      // Cumulative P&L by source
      cumGammaPnL: 0,
      cumVannaPnL: 0,
      cumVolgaPnL: 0,
      cumThetaCost: 0,
      cumNetPnL: 0,
      // Variance tracking
      realizedVariance: 0,
      annualizedRealizedVol: 0,
      // Status
      breakEvenMove: portfolioAnalysis.breakEvenMove,
      status: 'EVEN'
    }]);
    setCurrentDay(0);
    setIsRunning(false);
  }, [btcPrice, impliedVol, portfolioAnalysis.breakEvenMove]);
  
  // Step simulation with Cost Model P&L decomposition
  const stepSimulation = useCallback(() => {
    const maxDays = mode === 'simulation' ? dte : simDays;
    if (currentDay >= maxDays) return;
    
    setSimulationData(prevData => {
      const last = prevData[prevData.length - 1];
      
      // Generate moves
      const dailyReturn = generateDailyReturn(realizedVolAssumption / 100);
      const volMove = generateVolMove(volOfVol / 100, dailyReturn, spotVolCorr);
      
      const newSpot = last.spot * (1 + dailyReturn);
      const newVol = Math.max(10, last.vol * (1 + volMove)); // Floor vol at 10%
      
      // Recalculate Greeks at new spot/vol (simplified scaling)
      const spotRatio = last.spot / btcPrice;
      const gamma = portfolioAnalysis.totalGamma * spotRatio;
      const vanna = portfolioAnalysis.totalVanna * spotRatio;
      const volga = portfolioAnalysis.totalVolga;
      const theta = portfolioAnalysis.totalTheta * spotRatio;
      
      // P&L decomposition (Cost Model style)
      // Gamma P&L: 0.5 * Γ * S² * (return)²
      const gammaPnL = 0.5 * gamma * last.spot * last.spot * (dailyReturn ** 2);
      
      // Vanna P&L: Vanna * S * Δσ
      const vannaPnL = vanna * last.spot * volMove * last.vol / 100;
      
      // Volga P&L: 0.5 * Volga * (Δσ)²
      const volgaPnL = 0.5 * volga * (volMove * last.vol / 100) ** 2;
      
      // Theta cost (what you pay per day)
      const thetaCost = theta * last.spot;
      
      // Cumulative
      const cumGammaPnL = last.cumGammaPnL + gammaPnL;
      const cumVannaPnL = last.cumVannaPnL + vannaPnL;
      const cumVolgaPnL = last.cumVolgaPnL + volgaPnL;
      const cumThetaCost = last.cumThetaCost + thetaCost;
      const cumNetPnL = cumGammaPnL + cumVannaPnL + cumVolgaPnL + cumThetaCost;
      
      // Variance tracking
      const dailyVariance = dailyReturn ** 2;
      const cumulativeVar = last.realizedVariance + dailyVariance;
      const annualizedRV = Math.sqrt(cumulativeVar * 365 / (currentDay + 1)) * 100;
      
      // Status
      const impliedDailyVar = (portfolioAnalysis.avgIV / 100) ** 2 / 365;
      const expectedVar = impliedDailyVar * (currentDay + 1);
      const status = cumulativeVar > expectedVar * 1.05 ? 'AHEAD' : 
                     cumulativeVar < expectedVar * 0.95 ? 'BEHIND' : 'ON TRACK';
      
      const newBreakEven = gamma > 0 ? Math.sqrt(2 * Math.abs(theta) / gamma) * 100 : 0;
      
      return [...prevData, {
        day: currentDay + 1,
        spot: newSpot,
        vol: newVol,
        dailyReturnPct: dailyReturn * 100,
        volChange: volMove * 100,
        gammaPnL, vannaPnL, volgaPnL, thetaCost,
        cumGammaPnL, cumVannaPnL, cumVolgaPnL, cumThetaCost, cumNetPnL,
        realizedVariance: cumulativeVar,
        annualizedRealizedVol: annualizedRV,
        expectedVar,
        breakEvenMove: newBreakEven,
        status
      }];
    });
    
    setCurrentDay(prev => prev + 1);
  }, [currentDay, mode, dte, simDays, realizedVolAssumption, volOfVol, spotVolCorr, portfolioAnalysis, btcPrice]);
  
  // Auto-run
  useEffect(() => {
    const maxDays = mode === 'simulation' ? dte : simDays;
    if (!isRunning || currentDay >= maxDays) {
      setIsRunning(false);
      return;
    }
    const timer = setTimeout(stepSimulation, speed);
    return () => clearTimeout(timer);
  }, [isRunning, currentDay, mode, dte, simDays, stepSimulation, speed]);
  
  // Position management
  const addPosition = () => {
    setPositions([...positions, { instrument: '', size: 1, direction: 'buy', iv: 50, marketPrice: 0 }]);
  };
  
  const updatePosition = (index, field, value) => {
    const newPositions = [...positions];
    newPositions[index][field] = value;
    setPositions(newPositions);
  };
  
  const removePosition = (index) => {
    setPositions(positions.filter((_, i) => i !== index));
  };
  
  const latestData = simulationData[simulationData.length - 1] || {};
  
  // Cost model breakdown data for pie chart
  const costBreakdownData = [
    { name: 'Gamma Cost', value: Math.abs(portfolioAnalysis.totalGammaCost), fill: '#22c55e' },
    { name: 'Vanna Cost', value: Math.abs(portfolioAnalysis.totalVannaCost), fill: '#3b82f6' },
    { name: 'Volga Cost', value: Math.abs(portfolioAnalysis.totalVolgaCost), fill: '#a855f7' }
  ];
  
  // Waterfall P&L data
  const waterfallData = [
    { name: 'Gamma P&L', value: latestData.cumGammaPnL || 0, fill: '#22c55e' },
    { name: 'Vanna P&L', value: latestData.cumVannaPnL || 0, fill: '#3b82f6' },
    { name: 'Volga P&L', value: latestData.cumVolgaPnL || 0, fill: '#a855f7' },
    { name: 'Theta Cost', value: latestData.cumThetaCost || 0, fill: '#ef4444' },
    { name: 'Net P&L', value: latestData.cumNetPnL || 0, fill: (latestData.cumNetPnL || 0) >= 0 ? '#14b8a6' : '#f97316' }
  ];

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-4">
      <div className="max-w-7xl mx-auto space-y-4">
        {/* Header */}
        <div className="text-center mb-4">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-green-400 via-blue-500 to-purple-500 bg-clip-text text-transparent">
            Cost Model Gamma & Variance Tracker
          </h1>
          <p className="text-gray-400 text-sm">θ = Γ × Ω_G + Va × Ω_Va + Vo × Ω_Vo</p>
        </div>
        
        {/* Mode & Feature Toggles */}
        <div className="flex justify-center gap-2 flex-wrap">
          <button onClick={() => { setMode('simulation'); setSimulationData([]); }}
            className={`px-3 py-1.5 rounded text-sm font-medium transition ${mode === 'simulation' ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'}`}>
            Single Option
          </button>
          <button onClick={() => { setMode('portfolio'); setSimulationData([]); }}
            className={`px-3 py-1.5 rounded text-sm font-medium transition ${mode === 'portfolio' ? 'bg-purple-600' : 'bg-gray-700 hover:bg-gray-600'}`}>
            Portfolio
          </button>
          <button onClick={() => setShowCostModel(!showCostModel)}
            className={`px-3 py-1.5 rounded text-sm font-medium transition ${showCostModel ? 'bg-green-600' : 'bg-gray-700 hover:bg-gray-600'}`}>
            Cost Model: {showCostModel ? 'ON' : 'OFF'}
          </button>
        </div>
        
        {/* DVOL Prediction Integration */}
        <div className="bg-gradient-to-r from-blue-900/30 to-purple-900/30 rounded-xl p-4 border border-blue-800">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div>
              <h3 className="text-sm font-semibold text-blue-400 mb-1">DVOL Prediction Signal</h3>
              <div className="flex items-center gap-4">
                <div>
                  <span className="text-xs text-gray-400">Current DVOL:</span>
                  <input type="number" value={currentDvol} onChange={(e) => setCurrentDvol(Number(e.target.value))}
                    className="ml-2 w-16 bg-gray-800 border border-gray-700 rounded px-2 py-0.5 text-sm" />
                </div>
                <div>
                  <span className="text-xs text-gray-400">Predicted:</span>
                  <input type="number" value={predictedDvol} onChange={(e) => setPredictedDvol(Number(e.target.value))}
                    className="ml-2 w-16 bg-gray-800 border border-gray-700 rounded px-2 py-0.5 text-sm" />
                </div>
              </div>
            </div>
            <div className={`px-4 py-2 rounded-lg font-bold text-lg ${
              portfolioAnalysis.volSignal === 'LONG VOL' ? 'bg-green-600/50 text-green-300' :
              portfolioAnalysis.volSignal === 'SHORT VOL' ? 'bg-red-600/50 text-red-300' :
              'bg-gray-700 text-gray-300'
            }`}>
              {portfolioAnalysis.volSignal}
              <span className="text-sm font-normal ml-2">
                ({portfolioAnalysis.volSpread > 0 ? '+' : ''}{portfolioAnalysis.volSpread.toFixed(1)}%)
              </span>
            </div>
          </div>
        </div>
        
        {/* Configuration Panel */}
        <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
          {mode === 'simulation' ? (
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-9 gap-2">
              <div>
                <label className="block text-xs text-gray-400 mb-1">BTC Price</label>
                <input type="number" value={btcPrice} onChange={(e) => setBtcPrice(Number(e.target.value))}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm" />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Strike</label>
                <input type="number" value={strike} onChange={(e) => setStrike(Number(e.target.value))}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm" />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">DTE</label>
                <input type="number" value={dte} onChange={(e) => setDte(Number(e.target.value))}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm" />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Type</label>
                <select value={optionType} onChange={(e) => setOptionType(e.target.value)}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm">
                  <option value="call">Call</option>
                  <option value="put">Put</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">IV %</label>
                <input type="number" value={impliedVol} onChange={(e) => setImpliedVol(Number(e.target.value))}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm" />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">RV %</label>
                <input type="number" value={realizedVolAssumption} onChange={(e) => setRealizedVolAssumption(Number(e.target.value))}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm" />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Vol of Vol %</label>
                <input type="number" value={volOfVol} onChange={(e) => setVolOfVol(Number(e.target.value))}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm" />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Spot-Vol ρ</label>
                <input type="number" step="0.1" value={spotVolCorr} onChange={(e) => setSpotVolCorr(Number(e.target.value))}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm" />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Contracts</label>
                <input type="number" value={contracts} onChange={(e) => setContracts(Number(e.target.value))}
                  className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm" />
              </div>
            </div>
          ) : (
            <>
              <div className="space-y-2 mb-3 max-h-40 overflow-y-auto">
                {positions.map((pos, idx) => (
                  <div key={idx} className="grid grid-cols-6 gap-2">
                    <input type="text" value={pos.instrument} placeholder="BTC-27DEC24-100000-C"
                      onChange={(e) => updatePosition(idx, 'instrument', e.target.value)}
                      className="col-span-2 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs" />
                    <input type="number" step="0.1" value={pos.size} placeholder="Size"
                      onChange={(e) => updatePosition(idx, 'size', parseFloat(e.target.value))}
                      className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs" />
                    <select value={pos.direction} onChange={(e) => updatePosition(idx, 'direction', e.target.value)}
                      className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs">
                      <option value="buy">Long</option>
                      <option value="sell">Short</option>
                    </select>
                    <input type="number" value={pos.iv} placeholder="IV%"
                      onChange={(e) => updatePosition(idx, 'iv', parseFloat(e.target.value))}
                      className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs" />
                    <button onClick={() => removePosition(idx)} className="text-red-400 hover:text-red-300 text-sm">✕</button>
                  </div>
                ))}
              </div>
              <div className="flex gap-2 flex-wrap">
                <button onClick={addPosition} className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs">+ Add</button>
                <input type="number" value={btcPrice} onChange={(e) => setBtcPrice(Number(e.target.value))}
                  className="w-24 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs" placeholder="BTC Price" />
                <input type="number" value={simDays} onChange={(e) => setSimDays(Number(e.target.value))}
                  className="w-16 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs" placeholder="Days" />
                <input type="number" value={realizedVolAssumption} onChange={(e) => setRealizedVolAssumption(Number(e.target.value))}
                  className="w-16 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs" placeholder="RV%" />
                <input type="number" value={volOfVol} onChange={(e) => setVolOfVol(Number(e.target.value))}
                  className="w-16 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs" placeholder="VoV%" />
              </div>
            </>
          )}
          
          {/* Controls */}
          <div className="flex gap-2 mt-3">
            <button onClick={initializeSimulation}
              className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 rounded text-sm font-medium">Initialize</button>
            <button onClick={() => setIsRunning(!isRunning)} disabled={simulationData.length === 0}
              className="px-3 py-1.5 bg-green-600 hover:bg-green-700 rounded text-sm font-medium disabled:opacity-50">
              {isRunning ? 'Pause' : 'Run'}
            </button>
            <button onClick={stepSimulation} disabled={simulationData.length === 0}
              className="px-3 py-1.5 bg-gray-600 hover:bg-gray-700 rounded text-sm font-medium disabled:opacity-50">Step</button>
            <input type="number" value={speed} onChange={(e) => setSpeed(Number(e.target.value))}
              className="w-20 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs" placeholder="Speed ms" />
          </div>
        </div>
        
        {/* Greeks Dashboard */}
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-2">
          <div className="bg-gray-900 rounded-lg p-2 border border-gray-800">
            <div className="text-xs text-gray-400">Delta</div>
            <div className="text-lg font-bold text-blue-400">{portfolioAnalysis.totalDelta.toFixed(4)}</div>
          </div>
          <div className="bg-gray-900 rounded-lg p-2 border border-gray-800" style={{boxShadow: '0 0 10px rgba(34, 197, 94, 0.2)'}}>
            <div className="text-xs text-gray-400">Gamma ($)</div>
            <div className="text-lg font-bold text-green-400">${portfolioAnalysis.dollarGamma.toFixed(0)}</div>
          </div>
          <div className="bg-gray-900 rounded-lg p-2 border border-gray-800">
            <div className="text-xs text-gray-400">Vanna</div>
            <div className="text-lg font-bold text-blue-300">{portfolioAnalysis.totalVanna.toFixed(4)}</div>
          </div>
          <div className="bg-gray-900 rounded-lg p-2 border border-gray-800">
            <div className="text-xs text-gray-400">Volga</div>
            <div className="text-lg font-bold text-purple-400">{portfolioAnalysis.totalVolga.toFixed(4)}</div>
          </div>
          <div className="bg-gray-900 rounded-lg p-2 border border-gray-800" style={{boxShadow: '0 0 10px rgba(239, 68, 68, 0.2)'}}>
            <div className="text-xs text-gray-400">Daily θ</div>
            <div className="text-lg font-bold text-red-400">-${Math.abs(portfolioAnalysis.dailyThetaUsd).toFixed(0)}</div>
          </div>
          <div className="bg-gray-900 rounded-lg p-2 border border-gray-800">
            <div className="text-xs text-gray-400">Break-Even</div>
            <div className="text-lg font-bold text-yellow-400">{portfolioAnalysis.breakEvenMove.toFixed(2)}%</div>
          </div>
          <div className="bg-gray-900 rounded-lg p-2 border border-gray-800">
            <div className="text-xs text-gray-400">Avg IV</div>
            <div className="text-lg font-bold text-purple-400">{portfolioAnalysis.avgIV.toFixed(1)}%</div>
          </div>
          <div className="bg-gray-900 rounded-lg p-2 border border-gray-800">
            <div className="text-xs text-gray-400">RV (sim)</div>
            <div className="text-lg font-bold text-cyan-400">{realizedVolAssumption}%</div>
          </div>
        </div>
        
        {/* Cost Model Breakdown */}
        {showCostModel && (
          <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
            <h3 className="text-lg font-semibold mb-3 text-green-400">Cost Model Decomposition</h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Theta breakdown bars */}
              <div className="space-y-3">
                <div className="text-sm text-gray-400 mb-2">Daily Theta Attribution</div>
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-green-400">Gamma Cost (Ω_G)</span>
                    <span>${Math.abs(portfolioAnalysis.totalGammaCost).toFixed(2)}</span>
                  </div>
                  <div className="h-3 bg-gray-800 rounded-full overflow-hidden">
                    <div className="h-full bg-green-500 transition-all" style={{ width: `${portfolioAnalysis.gammaCostPct}%` }} />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-blue-400">Vanna Cost (Ω_Va)</span>
                    <span>${Math.abs(portfolioAnalysis.totalVannaCost).toFixed(2)}</span>
                  </div>
                  <div className="h-3 bg-gray-800 rounded-full overflow-hidden">
                    <div className="h-full bg-blue-500 transition-all" style={{ width: `${portfolioAnalysis.vannaCostPct}%` }} />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-purple-400">Volga Cost (Ω_Vo)</span>
                    <span>${Math.abs(portfolioAnalysis.totalVolgaCost).toFixed(2)}</span>
                  </div>
                  <div className="h-3 bg-gray-800 rounded-full overflow-hidden">
                    <div className="h-full bg-purple-500 transition-all" style={{ width: `${portfolioAnalysis.volgaCostPct}%` }} />
                  </div>
                </div>
                <div className="pt-2 border-t border-gray-700">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Total θ (Cost Model)</span>
                    <span className="font-bold text-red-400">
                      -${Math.abs(portfolioAnalysis.totalGammaCost + portfolioAnalysis.totalVannaCost + portfolioAnalysis.totalVolgaCost).toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm mt-1">
                    <span className="text-gray-400">BS Theta</span>
                    <span className="font-bold text-red-400">-${Math.abs(portfolioAnalysis.dailyThetaUsd).toFixed(2)}</span>
                  </div>
                </div>
              </div>
              
              {/* Master equation display */}
              <div className="bg-gray-800 rounded-lg p-4">
                <div className="text-sm text-gray-400 mb-3">Master Equation</div>
                <div className="font-mono text-lg text-center mb-4">
                  <span className="text-red-400">θ</span> = 
                  <span className="text-green-400"> Γ×Ω<sub>G</sub></span> + 
                  <span className="text-blue-400"> Va×Ω<sub>Va</sub></span> + 
                  <span className="text-purple-400"> Vo×Ω<sub>Vo</sub></span>
                </div>
                <div className="grid grid-cols-3 gap-2 text-xs">
                  <div className="text-center">
                    <div className="text-green-400 font-semibold">Volatility Carry</div>
                    <div className="text-gray-400">Realized variance</div>
                  </div>
                  <div className="text-center">
                    <div className="text-blue-400 font-semibold">Skew Carry</div>
                    <div className="text-gray-400">Spot-vol correlation</div>
                  </div>
                  <div className="text-center">
                    <div className="text-purple-400 font-semibold">Smile Carry</div>
                    <div className="text-gray-400">Vol of vol</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
        
        {/* Simulation Charts */}
        {simulationData.length > 0 && (
          <>
            {/* P&L Race with Cost Model decomposition */}
            <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
              <div className="flex justify-between items-center mb-3">
                <h3 className="text-lg font-semibold text-green-400">Cost Model P&L Attribution</h3>
                <span className="text-sm text-gray-400">Day {currentDay}</span>
              </div>
              
              {/* Live Stats - Now with Gamma/Vanna/Volga */}
              <div className="grid grid-cols-5 gap-2 mb-4">
                <div className="bg-green-900/30 border border-green-800 rounded-lg p-2 text-center">
                  <div className="text-xs text-green-400">Gamma P&L</div>
                  <div className="text-xl font-bold text-green-400">+${(latestData.cumGammaPnL || 0).toFixed(0)}</div>
                </div>
                <div className="bg-blue-900/30 border border-blue-800 rounded-lg p-2 text-center">
                  <div className="text-xs text-blue-400">Vanna P&L</div>
                  <div className="text-xl font-bold text-blue-400">{(latestData.cumVannaPnL || 0) >= 0 ? '+' : ''}{(latestData.cumVannaPnL || 0).toFixed(0)}</div>
                </div>
                <div className="bg-purple-900/30 border border-purple-800 rounded-lg p-2 text-center">
                  <div className="text-xs text-purple-400">Volga P&L</div>
                  <div className="text-xl font-bold text-purple-400">{(latestData.cumVolgaPnL || 0) >= 0 ? '+' : ''}{(latestData.cumVolgaPnL || 0).toFixed(0)}</div>
                </div>
                <div className="bg-red-900/30 border border-red-800 rounded-lg p-2 text-center">
                  <div className="text-xs text-red-400">Theta Paid</div>
                  <div className="text-xl font-bold text-red-400">{(latestData.cumThetaCost || 0).toFixed(0)}</div>
                </div>
                <div className={`${(latestData.cumNetPnL || 0) >= 0 ? 'bg-teal-900/30 border-teal-800' : 'bg-orange-900/30 border-orange-800'} border rounded-lg p-2 text-center`}>
                  <div className="text-xs text-gray-400">Net P&L</div>
                  <div className={`text-xl font-bold ${(latestData.cumNetPnL || 0) >= 0 ? 'text-teal-400' : 'text-orange-400'}`}>
                    {(latestData.cumNetPnL || 0) >= 0 ? '+' : ''}{(latestData.cumNetPnL || 0).toFixed(0)}
                  </div>
                </div>
              </div>
              
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={simulationData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="day" stroke="#9ca3af" />
                    <YAxis stroke="#9ca3af" tickFormatter={(v) => `$${v >= 1000 ? (v/1000).toFixed(0) + 'k' : v.toFixed(0)}`} />
                    <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                      formatter={(value) => [`$${Number(value).toFixed(0)}`, '']} />
                    <Legend />
                    <ReferenceLine y={0} stroke="#6b7280" strokeDasharray="3 3" />
                    <Area type="monotone" dataKey="cumGammaPnL" name="Gamma P&L" stackId="1" fill="#22c55e" fillOpacity={0.4} stroke="#22c55e" />
                    <Area type="monotone" dataKey="cumVannaPnL" name="Vanna P&L" stackId="1" fill="#3b82f6" fillOpacity={0.4} stroke="#3b82f6" />
                    <Area type="monotone" dataKey="cumVolgaPnL" name="Volga P&L" stackId="1" fill="#a855f7" fillOpacity={0.4} stroke="#a855f7" />
                    <Line type="monotone" dataKey="cumThetaCost" name="Theta Cost" stroke="#ef4444" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="cumNetPnL" name="Net P&L" stroke="#14b8a6" strokeWidth={3} dot={false} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>
            
            {/* Variance & Vol tracking */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Variance Budget */}
              <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
                <h3 className="text-lg font-semibold mb-3 text-yellow-400">Variance Budget</h3>
                <div className={`rounded-lg p-3 mb-3 ${
                  latestData.status === 'AHEAD' ? 'bg-green-900/30 border border-green-700' :
                  latestData.status === 'BEHIND' ? 'bg-red-900/30 border border-red-700' :
                  'bg-yellow-900/30 border border-yellow-700'
                }`}>
                  <div className="flex justify-between">
                    <div>
                      <div className="text-xs text-gray-400">Status</div>
                      <div className={`text-xl font-bold ${
                        latestData.status === 'AHEAD' ? 'text-green-400' :
                        latestData.status === 'BEHIND' ? 'text-red-400' : 'text-yellow-400'
                      }`}>{latestData.status || 'EVEN'}</div>
                    </div>
                    <div className="text-right">
                      <div className="text-xs text-gray-400">Realized Vol</div>
                      <div className="text-xl font-bold">{(latestData.annualizedRealizedVol || 0).toFixed(1)}%</div>
                    </div>
                  </div>
                </div>
                <div className="h-40">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={simulationData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="day" stroke="#9ca3af" />
                      <YAxis stroke="#9ca3af" />
                      <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }} />
                      <Legend />
                      <Line type="monotone" dataKey="realizedVariance" name="Realized Var" stroke="#22c55e" strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="expectedVar" name="Expected Var" stroke="#eab308" strokeWidth={2} strokeDasharray="5 5" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
              
              {/* Vol Evolution (for Vanna/Volga) */}
              <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
                <h3 className="text-lg font-semibold mb-3 text-purple-400">Vol Path (Vanna/Volga driver)</h3>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={simulationData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="day" stroke="#9ca3af" />
                      <YAxis yAxisId="vol" stroke="#a855f7" tickFormatter={(v) => `${v.toFixed(0)}%`} />
                      <YAxis yAxisId="spot" orientation="right" stroke="#8b5cf6" tickFormatter={(v) => `$${(v/1000).toFixed(0)}k`} />
                      <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }} />
                      <Legend />
                      <Line yAxisId="vol" type="monotone" dataKey="vol" name="Implied Vol" stroke="#a855f7" strokeWidth={2} dot={false} />
                      <Line yAxisId="spot" type="monotone" dataKey="spot" name="Spot" stroke="#8b5cf6" strokeWidth={2} dot={false} />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
            
            {/* P&L Waterfall */}
            <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
              <h3 className="text-lg font-semibold mb-3 text-orange-400">P&L Waterfall</h3>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={waterfallData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis type="number" stroke="#9ca3af" tickFormatter={(v) => `$${v.toFixed(0)}`} />
                    <YAxis type="category" dataKey="name" stroke="#9ca3af" width={80} tick={{ fontSize: 12 }} />
                    <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }} />
                    <ReferenceLine x={0} stroke="#6b7280" />
                    <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                      {waterfallData.map((entry, index) => (<Cell key={`cell-${index}`} fill={entry.fill} />))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </>
        )}
        
        {/* Model vs Market (for portfolio mode with market prices) */}
        {mode === 'portfolio' && portfolioAnalysis.modelVsMarket.length > 0 && (
          <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
            <h3 className="text-lg font-semibold mb-3 text-cyan-400">Model vs Market Arbitrage</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-gray-400 border-b border-gray-700">
                    <th className="text-left py-2 px-2">Instrument</th>
                    <th className="text-right py-2 px-2">Market $</th>
                    <th className="text-right py-2 px-2">Model $</th>
                    <th className="text-right py-2 px-2">Mispricing</th>
                    <th className="text-center py-2 px-2">Signal</th>
                  </tr>
                </thead>
                <tbody>
                  {portfolioAnalysis.modelVsMarket.map((item, idx) => (
                    <tr key={idx} className="border-b border-gray-800">
                      <td className="py-2 px-2">{item.instrument}</td>
                      <td className="py-2 px-2 text-right">${item.marketPrice.toFixed(2)}</td>
                      <td className="py-2 px-2 text-right">${item.modelPrice.toFixed(2)}</td>
                      <td className={`py-2 px-2 text-right ${item.mispricing > 0 ? 'text-red-400' : 'text-green-400'}`}>
                        {item.mispricing > 0 ? '+' : ''}{item.mispricing.toFixed(1)}%
                      </td>
                      <td className="py-2 px-2 text-center">
                        <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                          item.signal === 'BUY' ? 'bg-green-600' :
                          item.signal === 'SELL' ? 'bg-red-600' : 'bg-gray-600'
                        }`}>{item.signal}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
        
        {/* Instructions */}
        {simulationData.length === 0 && (
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800 text-center">
            <div className="text-4xl mb-3">📊</div>
            <h3 className="text-lg font-semibold mb-2">Cost Model Ready</h3>
            <p className="text-gray-400 text-sm mb-3">
              Click "Initialize" to start the simulation with full Cost Model P&L decomposition.
            </p>
            <div className="text-xs text-gray-500 font-mono">
              θ = Γ × Ω<sub>G</sub> + Va × Ω<sub>Va</sub> + Vo × Ω<sub>Vo</sub>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
