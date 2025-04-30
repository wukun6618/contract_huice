import pandas as pd
import matplotlib.pyplot as plt

# 1. 加载数据
def load_data(market_file, trade_file):
    """加载行情数据和开仓记录"""
    market_data = pd.read_csv(market_file)
    trade_log = pd.read_csv(trade_file)

    # 处理时间格式
    market_data['datetime'] = pd.to_datetime(market_data['datetime'])
    trade_log['时间'] = pd.to_datetime(trade_log['时间'], format="%Y%m%d%H%M")  # 转换格式
    
    return market_data, trade_log

# 2. 计算多个均线（换成EMA）
def calculate_ema(data, periods):
    """计算多个周期指数移动平均（EMA）"""
    for period in periods:
        data[f'EMA{period}'] = data['close'].ewm(span=period, adjust=False).mean()
    return data

# 3. 标记买点（只保留收盘价高于 50 日EMA 且 50 日EMA 5 日内呈上升趋势 且收盘价未高出 50 日EMA 1% 的买点）
def mark_buy_points(market_data, trade_log):
    """根据开仓时间，在行情数据中标记符合条件的买点"""
    buy_signals = market_data.merge(trade_log, left_on='datetime', right_on='时间', how='inner')

    return buy_signals

# 4. 绘制图表（支持多条EMA均线）
def plot_chart(market_data, buy_signals, ema_periods):
    """绘制行情数据、EMA均线和买点"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    plt.figure(figsize=(12, 6))

    # 绘制收盘价
    plt.plot(market_data['datetime'], market_data['close'], label="收盘价", color="blue", alpha=0.6)

    # 绘制多个EMA均线
    colors = ["black", "green", "purple", "brown"]  # 可以调整颜色
    for i, period in enumerate(ema_periods):
        plt.plot(market_data['datetime'], market_data[f'EMA{period}'], 
                 label=f"{period}日EMA", linestyle="dashed", color=colors[i % len(colors)])

    # 标记符合条件的买点
    plt.scatter(buy_signals['datetime'], buy_signals['close'], color="red", marker="o", label="买点（收盘高于EMA215）")

    plt.xlabel("日期")
    plt.ylabel("价格")
    plt.title(f"行情数据 - EMA均线 ({', '.join(map(str, ema_periods))})")
    plt.legend()
    plt.grid()
    plt.show()

# **示例使用**
market_data, trade_log = load_data(
    "D:/code/contract_huice/2024主力连续_15min/FG9999.XZCE_2024_15min.csv", 
    "D:/code/contract_huice/trade_log_15m.csv"
)
EMA_LIST = [215, 50]
market_data = calculate_ema(market_data, EMA_LIST)  # 计算多条EMA均线
buy_signals = mark_buy_points(market_data, trade_log)  # 标记符合条件的买点
plot_chart(market_data, buy_signals, EMA_LIST)  # 绘制图表
