import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
data_file = "D:/code/contract_huice/2024主力连续_5min/JM9999.XDCE_2024_5min.csv"
df = pd.read_csv(data_file)

# 转换时间格式
df['datetime'] = pd.to_datetime(df['datetime'])

# 计算 SMA 160 和 EMA 160
df['SMA_160'] = df['close'].rolling(window=215).mean()
df['EMA_160'] = df['close'].ewm(span=215, adjust=False).mean()

# 绘制收盘价、SMA 和 EMA
plt.figure(figsize=(12,6))
plt.plot(df['datetime'], df['close'], label='Close Price', color='blue', alpha=0.6)
plt.plot(df['datetime'], df['SMA_160'], label='SMA 160', color='red')
plt.plot(df['datetime'], df['EMA_160'], label='EMA 160', color='green')

plt.xlabel('Datetime')
plt.ylabel('Price')
plt.title('SMA and EMA (160) Visualization')
plt.legend()
plt.grid()

plt.show()