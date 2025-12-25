import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# 1. 获取两家巨头的数据 (2020年至今，跨度大一点以包含不同市场周期)
tickers = ['XOM', 'CVX']
data = yf.download(tickers, start="2020-01-01", end="2025-01-01")['Close']

# 2. 这里的数学关键：用 Log Price 更好，因为 Log 差值 = 收益率比率
S1 = np.log(data['XOM']) # 变量 Y
S2 = np.log(data['CVX'])  # 变量 X

# 3. 计算对冲比率 (Hedge Ratio) - 使用 OLS 线性回归
# 我们假设: PEP = beta * KO + alpha + residual
# 这个 residual 就是我们要交易的 Spread
X = sm.add_constant(S2) # 加上截距项
model = sm.OLS(S1, X)
results = model.fit()
beta = results.params['CVX'] # 拿到了 beta

print(f"计算出的对冲比率 (Hedge Ratio / Beta): {beta:.4f}")
print("这意味着: 每做多 1 份 XOM，你需要做空 {:.4f} 份 CVX 来保持中性。".format(beta))

# 4. 构建 Spread (残差序列)
# Spread = XOM - beta * CVX - alpha
spread = S1 - beta * S2 - results.params['const']

# 5. 关键判决：ADF 检验 (检验平稳性)
adf_result = adfuller(spread)
print("\n========== ADF Test Results ==========")
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"P-value: {adf_result[1]:.6f}")
print("Critical Values:")
for key, value in adf_result[4].items():
    print(f"\t{key}: {value:.4f}")

# 6. 可视化：Z-Score (标准化后的 Spread)
# 我们不看绝对值，看它偏离均值几个标准差
z_score = (spread - spread.mean()) / spread.std()

plt.figure(figsize=(12, 6))
z_score.plot(label='Z-Score of Spread')
plt.axhline(z_score.mean(), color='black', alpha=0.5)
plt.axhline(1.5, color='red', linestyle='--', label='Sell Signal (+1.5 Sigma)')
plt.axhline(-1.5, color='green', linestyle='--', label='Buy Signal (-1.5 Sigma)')
plt.title(f"Pairs Trading Signal: XOM vs CVX (Beta={beta:.2f})")
plt.legend()
plt.ylabel("Standard Deviations from Mean")
plt.grid(True)
plt.savefig("Figure_2_Pairs_Trading_Signal(XOM_vs_CVX).png", dpi=300, bbox_inches='tight')
plt.show()