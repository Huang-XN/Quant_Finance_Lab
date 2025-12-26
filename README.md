# Quant Finance Lab 🧬📈

Welcome to my personal quantitative finance laboratory. This repository documents my journey of transitioning from Bioinformatics to Quantitative Finance, focusing on reconstructing mathematical models and trading strategies from scratch.

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Installation & Dependencies](#-installation--dependencies)
- [Study Progression](#-study-progression)
- [About Me](#-about-me)

## 🛠️ Installation & Dependencies

```bash
# Required packages
pip install yfinance pandas numpy matplotlib seaborn
pip install arch scikit-learn hmmlearn joblib
pip install jupyter notebook
```

## 📂 Project Overview

### Study Progression

The studies are organized into two distinct tracks:
- **Econometric Track**: Statistical modeling and unsupervised learning (Studies 1-3, 7)
- **ML Track**: Supervised classification approaches (Studies 4-6)

---

### 1. Warm Up (Trend Following Strategy, Moving Average)
* **File:** `01_warm_up.ipynb`
* **Objective:** Implement a classic "Golden Cross" strategy on AAPL stock using pure Python/Pandas logic
* **Key Tech:** Data visualization, Signal generation using `.shift()` to avoid look-ahead bias, Backtesting framework
* **Result:** Verified the strategy's Sharpe ratio vs. Buy & Hold, laying the foundation for vectorized backtesting

---

### 2. Volatility Modeling (GARCH)
* **File:** `02_GARCH.ipynb`
* **Objective:** Model the "Volatility Clustering" phenomenon in financial time series, moving beyond constant volatility assumptions in Black-Scholes
* **Key Tech:** Monte Carlo Simulation (Geometric Brownian Motion), GARCH(1,1) model implementation using `arch` library, Value-at-Risk (VaR) calculation
* **Insight:** Identified statistical significance of volatility shocks and visualized the leverage effect in AAPL returns

---

### 3. Statistical Arbitrage (Pairs Trading)
* **File:** `03_Pair_Trading.ipynb`
* **Objective:** Build a market-neutral strategy based on Cointegration theory rather than simple correlation
* **Key Tech:** Augmented Dickey-Fuller (ADF) Test, OLS Regression for Hedge Ratio, Z-Score signal generation
* **Case Study:**
    * **PEP vs KO:** Failed stationarity test (P-value > 0.05), proving structural breaks in consumer staples
    * **XOM vs CVX:** Confirmed cointegration (P-value < 0.05)
* **Performance:** Achieved a total Log Return of ~1.09 over 5 years, despite significant drawdowns in 2022 due to geopolitical structural breaks 

---

### 4. Machine Learning Prediction
* **File:** `4_ML_prediction/04_Machine_Learning_Prediction.ipynb`
* **Objective:** Implement supervised machine learning pipeline for directional stock movement forecasting
* **Key Tech:** Feature engineering (lagged returns, RSI, SPY), RandomForest/LogisticRegression/SVM classifiers, Chronological train/test splitting
* **Results:** 
    * Test accuracy: ~51-55% (marginally above random)
    * Severe overfitting detected (43% train-test gap)
    * Strategy underperformed Buy & Hold benchmark
* **Insight:** Demonstrated the challenge of translating classification accuracy into economic profitability 

---

### 5. Hyperparameter Optimization
* **File:** `5_Hyperparameter_Optimization/5_Hyp_Opt.ipynb`
* **Objective:** Address overfitting issues from Study 4 through systematic hyperparameter tuning
* **Key Tech:** GridSearchCV with TimeSeriesSplit, RandomForest parameter optimization, Cross-validation with temporal ordering
* **Approach:** Used `TimeSeriesSplit` to prevent look-ahead bias while optimizing for:
    * `max_depth`: [None, 10, 20]
    * `min_samples_leaf`: [1, 2, 4] 
    * `min_samples_split`: [10, 20, 50]
    * `n_estimators`: [50, 100, 200]

---

### 6. Ensemble Learning
* **File:** `6_Ensemble_Learning/6_Ensemble.ipynb`
* **Objective:** Combine multiple classifiers to improve prediction robustness
* **Key Tech:** VotingClassifier, Model diversity analysis, Ensemble performance evaluation
* **Results:** Achieved 52.21% ensemble accuracy - marginal improvement over individual models
* **Insight:** Limited benefit from ensemble methods when individual classifiers struggle with weak signal-to-noise ratio

---

### 7. Market Regime Detection (Hidden Markov Model) 🕵️‍♂️
* **File:** `7_HMM/7_HMM.ipynb`
* **Objective:** Switch from "micro-prediction" (forecasting daily returns) to "macro-identification" (detecting market states), applying the logic of **Cell State Classification** to financial markets
* **Key Tech:** Gaussian HMM (Unsupervised Learning), Dynamic Regime Detection
* **Insight:**
    * Successfully identified the **2022 Bear Market** (State 2: High Volatility, Negative Returns) without any labeled data
    * Identified the **2024 Bull Run** (State 0/1: Low Volatility, Positive Returns)
    * **Visual Proof:**
    ![HMM Regimes](./7_HMM/Figure_1_HMM_for_AAPL.png)
    *(Note: The model autonomously learned to flag high-risk periods in red)* 

## 📊 Study Comparison

| Study | Approach | Target | Accuracy/Performance | Key Insight |
|-------|----------|--------|---------------------|-------------|
| 1 | Technical Analysis | Trend signals | Sharpe vs B&H | Foundation for backtesting |
| 2 | Econometric | Volatility | VaR modeling | Volatility clustering |
| 3 | Statistical | Mean reversion | 1.09 log return | Cointegration > correlation |
| 4 | ML Classification | Direction | 51-55% accuracy | Overfitting challenges |
| 5 | ML Optimization | Generalization | Improved validation | Temporal CV critical |
| 6 | ML Ensemble | Robustness | 52.21% accuracy | Limited diversity benefit |
| 7 | Unsupervised | Regimes | State identification | Macro > Micro prediction |

## 🚀 About Me

I am a **Ph.D. candidate in Biology** with a strong foundation in **Mathematics (B.S.)** and **Financial Mathematics (M.S.)**. 

For the past years, I have been working in biological research, specializing in extracting signals from high-dimensional, noisy biological sequencing data. I am now applying this "signal-from-noise" expertise to financial markets.
