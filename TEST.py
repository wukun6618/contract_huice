import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 计算真实波动幅度（TR）
def calculate_tr(df):
    """计算真实波动幅度"""
    df['prev_close'] = df['close'].shift(1)
    df['TR'] = np.maximum.reduce([
        df['high'] - df['low'],
        abs(df['prev_close'] - df['high']),
        abs(df['prev_close'] - df['low'])
    ])
    return df

# 2. 计算动态通道
def calculate_dynamic_channel(df, period=5, factor=1.5):
    """构建动态通道"""
    df = calculate_tr(df)
    df['avg_TR'] = df['TR'].rolling(period).mean()  # 计算波动均值
    df['middle'] = (df['high'] + df['low']) / 2  # 中轴计算
    df['upper_band'] = df['middle'] + df['avg_TR'] * factor  # 上轨
    df['lower_band'] = df['middle'] - df['avg_TR'] * factor  # 下轨
    return df

# 3. 计算突破信号
def calculate_breakout(df):
    """突破信号"""
    df['prev_upper_band'] = df['upper_band'].shift(1)
    df['breakout_signal'] = np.where(df['upper_band'] > df['prev_upper_band'], 1, 0)
    return df

# 4. 计算买卖信号
def calculate_trade_signals(df):
    """买卖信号逻辑"""
    df['prev_lower_band'] = df['lower_band'].shift(1)
    df['buy_signal'] = np.where(df['close'] > df['upper_band'], df['close'], np.nan)  # 仅绘制买点
    df['sell_signal'] = np.where(df['close'] < df['lower_band'], df['close'], np.nan)  # 仅绘制卖点
    return df

# 5. 绘制图表（包含买卖信号）
def plot_chart(df):
    """绘制行情数据、动态通道和买卖信号"""
    plt.figure(figsize=(12, 6))
    
    # 绘制收盘价
    plt.plot(df['datetime'], df['close'], label="收盘价", color="blue", alpha=0.6)
    
    # 绘制通道
    plt.plot(df['datetime'], df['upper_band'], label="上轨", color="orange", linestyle="dashed")
    plt.plot(df['datetime'], df['lower_band'], label="下轨", color="green", linestyle="dashed")
    
    # 标记买点
    plt.scatter(df['datetime'], df['buy_signal'], color="red", marker="^", label="买点")
    
    # 标记卖点
    plt.scatter(df['datetime'], df['sell_signal'], color="black", marker="v", label="卖点")

    plt.xlabel("日期")
    plt.ylabel("价格")
    plt.title("动态通道策略 - 买卖信号")
    plt.legend()
    plt.grid()
    plt.show()

# **执行完整策略**
data_file = "D:/code/contract_huice/2024主力连续_5min/FG9999.XZCE_2024_5min.csv"
df = pd.read_csv(data_file)
df['datetime'] = pd.to_datetime(df['datetime'])  # 时间格式转换

df = calculate_dynamic_channel(df)  # 计算动态通道
df = calculate_breakout(df)         # 计算突破信号
df = calculate_trade_signals(df)     # 计算买卖信号

plot_chart(df) 