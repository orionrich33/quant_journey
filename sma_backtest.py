# ===============================
# SMA Crossover Strategy — EUR/USD
# ===============================
# Built by Lucjan | Quant Project 1
# Logic: Long EUR/USD when 20-day SMA > 50-day SMA

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# STEP 1: Load Historical Forex Data
# ===============================
ticker = "EURUSD=X"  # Euro to USD ticker on Yahoo Finance
data = yf.download(ticker, start="2015-01-01", end="2024-01-01")

# ===============================
# STEP 2: Calculate Moving Averages
# ===============================
data["SMA_20"] = data["Close"].rolling(window=20).mean()
data["SMA_50"] = data["Close"].rolling(window=50).mean()

# ===============================
# STEP 3: Generate Trade Signals
# ===============================
data["Signal"] = 0
data.loc[data.index[50:], "Signal"] = (data["SMA_20"][50:] > data["SMA_50"][50:]).astype(int)
data["Position"] = data["Signal"].diff()  # Trade entry/exit flag

# ===============================
# STEP 4: Plot Price and Trade Signals
# ===============================
plt.figure(figsize=(16, 8))
plt.plot(data["Close"], label="Close Price", alpha=0.6)
plt.plot(data["SMA_20"], label="20-Day SMA", linestyle="--")
plt.plot(data["SMA_50"], label="50-Day SMA", linestyle="--")

plt.plot(data[data["Position"] == 1].index,
         data["SMA_20"][data["Position"] == 1],
         "^", markersize=10, color="g", label="Buy Signal")

plt.plot(data[data["Position"] == -1].index,
         data["SMA_20"][data["Position"] == -1],
         "v", markersize=10, color="r", label="Sell Signal")

plt.title(f"{ticker} SMA Crossover Strategy")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ===============================
# STEP 5: Strategy Performance Calculation
# ===============================
data["Return"] = data["Close"].pct_change()
data["Strategy_Return"] = data["Return"] * data["Signal"].shift(1)
data["Equity_Curve"] = (1 + data["Strategy_Return"]).cumprod()

# ===============================
# STEP 6: Plot Equity Curve
# ===============================
plt.figure(figsize=(14, 6))
plt.plot(data["Equity_Curve"], label="Strategy Equity Curve")
plt.title(f"{ticker} Strategy Performance")
plt.xlabel("Date")
plt.ylabel("Equity (Growth of $1)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ===============================
# STEP 7: Performance Metrics
# ===============================
sharpe_ratio = data["Strategy_Return"].mean() / data["Strategy_Return"].std() * (252 ** 0.5)
final_return = (data["Equity_Curve"].iloc[-1] - 1) * 100

print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Total Return: {final_return:.2f}%")



