# Quant Finance Lab 🧬📈

Welcome to my personal quantitative finance laboratory. This repository documents my journey of transitioning from Bioinformatics to Quantitative Finance, focusing on reconstructing mathematical models and trading strategies from scratch.

## 📂 Project Overview

### 1. Trend Following Strategy (Moving Average)
* **File:** `01_Trend_Following_Moving_Average.ipynb`
* **Objective:** To implement a classic "Golden Cross" strategy on AAPL stock using pure Python/Pandas logic.
* **Key Tech:** Data visualization, Signal generation using `.shift()` to avoid look-ahead bias, Backtesting framework.
* **Result:** Verified the strategy's Sharpe ratio vs. Buy & Hold, laying the foundation for vectorized backtesting.

### 2. Volatility Modeling (GARCH)
* **File:** `02_Volatility_Modeling_GARCH.ipynb`
* **Objective:** To model the "Volatility Clustering" phenomenon in financial time series, moving beyond constant volatility assumptions in Black-Scholes.
* **Key Tech:** Monte Carlo Simulation (Geometric Brownian Motion), GARCH(1,1) model implementation using `arch` library, Value-at-Risk (VaR) calculation.
* **Insight:** Identified statistical significance of volatility shocks and visualized the leverage effect in AAPL returns.

### 3. Statistical Arbitrage (Pairs Trading)
* **File:** `03_Pairs_Trading_Cointegration.ipynb`
* **Objective:** To build a market-neutral strategy based on Cointegration theory rather than simple correlation.
* **Key Tech:** Augmented Dickey-Fuller (ADF) Test, OLS Regression for Hedge Ratio, Z-Score signal generation.
* **Case Study:**
    * **PEP vs KO:** Failed stationarity test (P-value > 0.05), proving structural breaks in consumer staples.
    * **XOM vs CVX:** Confirmed cointegration (P-value < 0.05).
* **Performance:** Achieved a total Log Return of ~1.09 over 5 years, despite significant drawdowns in 2022 due to geopolitical structural breaks.

## 🚀 About Me
I am a Ph.D. candidate with a background in Computation. I specialize in decoding complex systems.
