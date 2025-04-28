import pandas as pd
import numpy as np
from datetime import datetime as dt

import json
import requests
import time
import configparser
import xml.etree.ElementTree as ET
from xml.dom.minidom import parse
import os
import sqlite3
import logging


pd.set_option('expand_frame_repr', False)  #不换行
pd.set_option('display.max_rows', 5000)     #最多显示数据的行数
pd.set_option('display.unicode.ambiguous_as_wide', True) # 中文字段对齐
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.float_format', lambda x: '%.3f' % x) # dataframe格式化输出

list_data_values                    = []#[[0,0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
ATR_LEN                             = 5
YKB                                 = 5
Max_buynums                         = 20
DEFAULT_NUMBER_OF_POINTS            = 0.02
TRADE_DIRECT                        = 48
RSI_BASE_LENGTH                     = 7

M_Start_Time                        = "090000"
M_End_Time                          = "142500"
Lastkindextimecnt                   = '00000'

class b():
    pass
classlocal = b()
classlocal.monitor_start_time       = M_Start_Time
classlocal.monitor_end_time         = M_End_Time
classlocal.logginginfomoney_en      = 1
classlocal.logginginfolocalhold_en  = 1
classlocal.sell_debug_inf_en        = 0
classlocal.checklist_debug_en       = 0 #打印本地自选股行情
classlocal.Index_time_debug_en      = 0
classlocal.Trade_init_debug_en      = 0 #
classlocal.model_df_level2_debug_en = 0 #模型选出列表购买列表
classlocal.JLZY_debug_en            = 0 #棘轮止盈打印
classlocal.huicedebug_en            = 1 #回测的时候打开，运行的时候关闭
classlocal.mp_debug_origin_en       = 0 #模型选出打印
classlocal.ZXCS_debug_en            = 0 #执行周期和次数打印
classlocal.h_data_debug_en          = 0 #打印执行选股前的行情数据
classlocal.RSI_debug_en             = 0 #debug信息打印
classlocal.RSI_STOP_DEBUG           = 0 #行情止损打印
classlocal.check_list               = ['SA00.ZF']
classlocal.check_list_debug_en      = 0 #自定义行情品种
classlocal.h_data_type_debug_en     = 0 #h_data 行情数据类型
classlocal.barsincentry_debug_en    = 1 #打印合约信息
classlocal.get_signal_margin_en     = 0 #打印保证金率
classlocal.contract_debug_en        = 1 #云端合约打印
classlocal.RSI_takeprofit_debug_en  = 0 #RSI 止盈打印开关
classlocal.sqlite3_debug_en         = 0 #sqlite3 信息打印
# -------------------------------------------#
classlocal.shipan_en                = 0                 ##False :模拟账号；True :实盘账号；
# 行情设置开关
classlocal.lefthand_checken         = 0                 # 1 打开行情止损 0 关闭
classlocal.LongMarginRatio_add      = 0.09              # 在最低保证金基础增加的比例
classlocal.close_atr_trade_en       = 0                 #0：关掉ART 1:打开ATR行情止盈
classlocal.max_buy_nums             = Max_buynums
classlocal.Fundbal_AvailRate        = 0.05              #单只占总资金仓位
# -------------------------------------------#
# 数据类型
classlocal.p                        = 0                 # 绘图点用
classlocal.count                    = 0                 # 01 记录定时函数执行次数
classlocal.Period_Type              = '5m'
classlocal.trade_buy_record_dict    = {}                # 02 买入交易记录
classlocal.buy_code_count           = 0                 # 03 风控函数，防止买入过多。
classlocal.Reflash_buy_list         = 1



classlocal.write_cnt                = 0                 # 计时变量
classlocal.write_local_data         = 1                 # 0:不更新 1：写数据到本地，默认进去写一次，
classlocal.write_local_hold_data_freq = 50000           # 写数据到本地时间，只有实盘的时候这个时间才生效


classlocal.draw_df                  = pd.DataFrame()
# 0：无需刷新stock_level1_lsit 1:需要重新刷新stock_level1_lsit
classlocal.ATR_open_Length          = 4*ATR_LEN         # 图标bar线数量为20

classlocal.ATR_close_Length         = 3*ATR_LEN         # 图标Bar线数量为10
classlocal.M_HL                     = 3*ATR_LEN         # 中轴线参数设置为10

classlocal.MA_middle_length         = 90#99            # 中均线长度
classlocal.MA_long_length           = 144#144           # 长均线长度

classlocal.ATR                      = 0  # ATR平均真实波幅
classlocal.ATR_BuyK                 = 0  # 开多时ATR数值
classlocal.ATR_SellK                = 0  # 开空时ATR数值
classlocal.Price_BuyK               = 0  # 开多时的价格
classlocal.Price_SellK              = 0  # 开空时的价格
classlocal.close                    = 0  #
classlocal.open                     = 0  #
classlocal.low                      = 0  #
classlocal.highmax                  = 0  #
classlocal.lowmin                   = 0  #
classlocal.szxfd                    = 0.018 #
classlocal.modul_length             = 240 #多少日

#
classlocal.RSI_en                   = 1

classlocal.RSI_choosed              = 0
classlocal.RSI_value                = np.array([])
classlocal.RSI_value_dict           = {}
classlocal.RSIsp                    = 8888
classlocal.volume                   = 0
classlocal.selRSI_stopcheck         = 0
classlocal.sellRSI_time             = 16    #买入后多久执行


classlocal.k_yin_en                 = False #False:不管前一个是啥线入场 True:阳线入场
classlocal.RSI_threshold_high       = 75   #触发空的值
classlocal.RSI_threshold_Low        = 35   #RSI触发多的值
classlocal.RSI_length               = 14
classlocal.h_data                   = pd.DataFrame()
classlocal.alternate_day_trading_en = False #False :隔日交易；True :日内交易；
classlocal.night_trading_en         = True #False :禁止夜盘开盘；True :夜盘开盘打开；
classlocal.without_last_data        = True #True :RSI是根据前一根K线来算的；false :是根据最新的K线来算的；
classlocal.RSI_limit_takeprofit_en  = False #False :关闭RSI上限止盈；True :打开RSI上限止盈；
classlocal.RSI_UP_TAKEPROFIT_SET    = 73   #触发RSI上限止盈，这个需要配合RSI_limit_takeprofit_en为True时有效
################################################################################################
#1.总涨幅止盈
classlocal.Price_SellYA_Ratio       = 2    #涨到个点止盈,手动买入时生效
#2.时间止盈
classlocal.BarSinceEntrySet         = 200  #N天时间止损
#3.盈亏比预设
classlocal.Price_SetSellY_YKB       = YKB    #盈亏比设置为3
classlocal.Price_SetSellS           = DEFAULT_NUMBER_OF_POINTS  #默认止损,无论如何都会在
classlocal.Price_SetSellYratio      = DEFAULT_NUMBER_OF_POINTS*YKB  #
#3.5 盈亏比止盈触发后进行棘轮止盈
classlocal.Price_SellS1_ATRratio    = 15    #ATR棘轮止损默认
classlocal.Price_SellY1_ATRratio    = 0.02   #ATR 止盈 默认比例 越大约灵敏大
classlocal.TC_ATRratio              = 1.0


classlocal.Price_SellY              = 0     #棘轮止盈利开始价格
classlocal.Price_SellY1             = 0     #棘轮止盈止盈价格,根据ATR值相关

#classlocal.Price_SetSellY1          = 1     #
classlocal.Price_SellY_Flag         = 0     #第一阶段止盈达到
classlocal.TH_low                   = 5.9   #价格筛选下线单位元
classlocal.TH_High                  = 100    #价格筛选上线单位元
################################################################################################

classlocal.Kindex                   = 0     # 当前K线索引
classlocal.Lastkindextime           = ''     # 当前K线索引

classlocal.Lastkindextime_draw      = ''     # 用于画图

classlocal.Kindex_time              = 0     # 当前K线对应的时间
classlocal.zf_lastK                 = 0     # 当前K线对应的涨幅
classlocal.buy_list                 = []    #买入列表
classlocal.sell_list                = []    #卖出列表
classlocal.LeftMoey                 = 1     #剩余资金
classlocal.LeftMoeyLast             = 0     #上次剩余
classlocal.Total_market_cap         = 0     #持仓次市值
classlocal.Total_market_capLast     = 0     #上次持仓次市值
classlocal.sp_type                  = 'NONE'
classlocal.eastmoney_zx_name        = ''
classlocal.eastmoey_stockPath       = ''
classlocal.eastmoney_user_buy_list  = ''
classlocal.eastmoney_zx_name_list   = ''
classlocal.stockPath_hold           = ''
classlocal.user_buy_list            = ''


classlocal.trade_direction  = 'kong' #duo #kong
classlocal.code             = 'SA00.SF'
classlocal.kindextime       = '0'
classlocal.timetype         = '5m'
classlocal.tradetype        = 'open'  #open #close
classlocal.tradedata        = ''
classlocal.stop             = 0
classlocal.takeprofit       = 0

classlocal.last_price       = 0
classlocal.profit           = 0
classlocal.mediumprice      = 0
classlocal.tradestatus      = ''
classlocal.modle            = ''
classlocal.URLopen          = 'https://open.feishu.cn/open-apis/bot/v2/hook/763bec44-0f8e-447b-8341-2e567d7fd6a8'
classlocal.URLclose         = 'https://open.feishu.cn/open-apis/bot/v2/hook/763bec44-0f8e-447b-8341-2e567d7fd6a8'
classlocal.URLopen_huice    = 'https://open.feishu.cn/open-apis/bot/v2/hook/fb5aa4f9-16b9-49f2-8e3b-2583ec3f3e3e'
classlocal.URLclose_huice   = 'https://open.feishu.cn/open-apis/bot/v2/hook/fb5aa4f9-16b9-49f2-8e3b-2583ec3f3e3e'

###################################start###########################################################################
#
###################################start###########################################################################
def check_lowest_index(array):
    """
    检查数组的最低值索引是否是最新值（最后一个值的索引）。
    :param array: numpy.array，待检查的数组
    :return: bool，若最低值索引是最新值则返回 False，否则返回 True。
    """
    # 确保输入是 numpy.array
    if not isinstance(array, np.ndarray):
        array = np.array(array)

    # 获取数组的最低值索引
    lowest_index = np.argmin(array)  # 最低值索引
    latest_index = len(array) - 1   # 最新值的索引（最后一个元素的索引）
    if classlocal.RSI_debug_en:
        print(f"最低值索引: {lowest_index}, 最新值索引: {latest_index}")

    # 比较最低值索引和最新值索引
    if lowest_index == latest_index:
        if classlocal.RSI_debug_en:
            print("最低值索引是最新值索引，返回 True")
        return True
    else:
        if classlocal.RSI_debug_en:
            print("最低值索引不是最新值索引，返回 False")
        return False

###################################start###########################################################################
#
###################################start###########################################################################

def is_within_time_range(kindex_time, start_time, end_time):
    """
    检查时间是否在指定范围内。
    :param kindex_time: str，时间格式为 YYYYMMDDHHMMSS，例如 "20250408141000"
    :param start_time: str，起始时间，格式为 HHMMSS，例如 "090000"
    :param end_time: str，结束时间，格式为 HHMMSS，例如 "113000"
    :return: bool，True 表示在范围内，False 表示不在范围内
    """
    # 提取时间部分（忽略日期）
    current_time = kindex_time[8:14]  # 提取 HHMMSS 部分

    # 判断是否在范围内
    return start_time <= current_time <= end_time

###################################start###########################################################################
#
###################################start###########################################################################
def RSI_checkout(classlocal):
    righthand       = False
    h_data_len      = 6*classlocal.RSI_length  #6*14
    h_data_allt     = classlocal.h_data
    h_data_all      = h_data_allt.copy()
    #print("h_data_all_type:", type(h_data_all))
    # 使用 iloc 提取最后 50 行
    h_data          = h_data_all.iloc[-h_data_len:-1]  # 提取倒数第 50 行到倒数第二行
    # 去掉最后一行（基于索引）
    if classlocal.without_last_data:
        h_data_drop_last = h_data.drop(h_data.index[-1])  # 删除最后一行
    else:
        h_data_drop_last = h_data

    '''
    提取列时类型为 DataFrame：
    使用 h_data[['open']] 提取列会返回一个 DataFrame，而不是 Series。
    函数内部需要操作的是 pd.Series 或 numpy.array。
    open_prices     = h_data[['open']]
    close_prices    = h_data[['close']]
    min_prices      = h_data[['low']]
    '''
    open_prices = h_data['open']
    close_prices = h_data['close']
    min_prices = h_data['low']


    #----------------------------------------------------------------------------------------------------
    #不会按照最新的数据算来开
    length          = classlocal.RSI_length        #设置的值为14
    rsi             = calculate_rsi(h_data_drop_last, period=length)
    #print("rsi:", rsi)
    #----------------------------------------------------------------------------------------------------
#选股时间
    monitor_start_time  = classlocal.monitor_start_time
    monitor_end_time    = classlocal.monitor_end_time
    time_on1            = is_within_time_range(classlocal.Kindex_time,monitor_start_time,monitor_end_time)
    #夜盘
    time_on2            = is_within_time_range(classlocal.Kindex_time,'210000',"225000")
    #----------------------------------------------------------------------------------------------------
    if (time_on1 or (time_on2 and classlocal.night_trading_en)):
        if(time_on2 and classlocal.night_trading_en):
             if classlocal.RSI_debug_en:
                pass
                #print('夜盘打开')
        threshold       = classlocal.RSI_threshold_Low

        window          = (classlocal.RSI_length * 2) - 8  # 20 :14*2 -8
        threshold_days  = (classlocal.RSI_length / 2) - 1  # 6 :14/2 - 1
    #----------------------------------------------------------------------------------------------------
        righthand = check_rsi_conditions(rsi, open_prices, close_prices, 
                                         min_prices, threshold, threshold_days, window)
    #----------------------------------------------------------------------------------------------------
        if classlocal.RSI_debug_en and righthand:
            #print("rsi:", rsi)
            #print("h_data_RSI:", h_data_drop_last)
            formatted_rsi = np.around(rsi[-(window):], decimals=2)
            h_data_RSI    = h_data_drop_last.iloc[-(window):]
            # 添加 RSI 列到 DataFrame 中
            #h_data_RSI["RSI"] = formatted_rsi
            h_data_RSI = h_data_RSI.copy()
            h_data_RSI["RSI"] = formatted_rsi

            print("h_data_RSI含RSI值:\n", h_data_RSI)
            print("退出RSI筛选\n")

        classlocal.RSI_choosed        = 0
        if (righthand):
            classlocal.RSI_choosed    = 1
            #止损向下移动一点
            #classlocal.code           = contract
            classlocal.RSIsp          = classlocal.lowmin*0.9996
        else:
            classlocal.RSI_choosed    = 0
            classlocal.RSIsp  = 8888
    else:
        classlocal.RSI_choosed        =  0
        if time_on1 :
            print("开仓停止，平仓直到收盘")
        if time_on2 and classlocal.night_trading_en:
            print("夜盘开仓停止，平仓直到收盘")
    return classlocal.RSI_choosed
###################################start###########################################################################
#
###################################start###########################################################################
# 自定义 SMA 逐步平滑算法
def sma(series, period):
    """同花顺逐步平滑的 SMA 算法"""
    result = [series.iloc[0]]  # 初始化第一个值
    alpha = 1 / period  # 平滑权重
    for i in range(1, len(series)):
        prev = result[-1]
        current = series.iloc[i]
        smoothed = alpha * current + (1 - alpha) * prev
        result.append(smoothed)
    return pd.Series(result, index=series.index)

# 保持原样的 calculate_rsi 函数
def calculate_rsi(hdata, period):
    """
    按照标准方法计算 RSI 指标
    :param hdata: DataFrame 格式的收盘价数据
    :param period: int，周期长度，默认为14
    :return: pandas.Series 格式的 RSI 数值
    """
    # 计算 LC (REF(CLOSE, 1))
    #print('type',type(hdata))  # 检查类型
    #print('hdata.head()',hdata.head())  # 检查 DataFrame 的头部内容
    # 筛选出 'close' 列，保持 DataFrame 格式
    original_data = hdata[['close']]

    # 创建切片并修复副本问题
    hdata = original_data[['close']].copy()

    # 使用 .loc 安全赋值
    hdata.loc[:, 'LC'] = hdata['close'].shift(1)

    # TEMP1 = MAX(CLOSE - LC, 0)
    hdata['TEMP1'] = (hdata['close'] - hdata['LC']).apply(lambda x: max(x, 0))

    # TEMP2 = ABS(CLOSE - LC)
    hdata.loc[:, 'TEMP2'] = (hdata['close'] - hdata['LC']).abs()


    # 计算平滑值
    hdata['SMA_TEMP1'] = sma(hdata['TEMP1'].fillna(0), period)
    hdata['SMA_TEMP2'] = sma(hdata['TEMP2'].fillna(0), period)

    # RSI 计算公式
    hdata['RSI'] = (hdata['SMA_TEMP1'] / hdata['SMA_TEMP2']) * 100

    # 返回完整的 RSI Series
    latest_rsi = hdata['RSI']
    # 只获取 RSI 的值
    rsi_values = latest_rsi.values  # 使用 `.values` 代替 `.to_numpy()`

    return rsi_values


# 条件 2: 检查倒数三天内是否有两天是阳线，且最后一天必须是阳线
def check_last_three_days_conditions(open_prices, close_prices):
    """
    检查倒数三天内是否有两天是阳线，且最后一天必须是阳线。

    :param open_prices: pd.Series, 开盘价数据
    :param close_prices: pd.Series, 收盘价数据
    :return: bool, 是否满足条件
    """
    # 获取倒数三天的开盘价和收盘价
    last_three_days_open = open_prices.iloc[-3:]  # 倒数三天开盘价
    last_three_days_close = close_prices.iloc[-3:]  # 倒数三天收盘价

    # 统计阳线天数
    count_sunshine = (last_three_days_close > last_three_days_open).sum()  # 阳线数量
    is_last_day_sunshine = last_three_days_close.iloc[-1] > last_three_days_open.iloc[-1]  # 最后一天是否阳线

    # 判断条件
    if not (count_sunshine >= 2 and is_last_day_sunshine):
        if classlocal.RSI_debug_en:
            #print("倒数三天内阳线数量不足两天，或者最后一天不是阳线，不满足条件。")
            pass
        return False
    else:
        if classlocal.RSI_debug_en:
            print("倒数三天内有两天是阳线，且最后一天是阳线，条件满足。")
        return True

def check_rsi_conditions(rsi, open_prices, close_prices, min_prices, threshold, threshold_days, window):
    """
    检查以下条件：
    1. 最近 20 天内，RSI <= threshold 的天数需要大于 threshold_days。
    2. 倒数两天必须是阳线（收盘价 > 开盘价）。
    3. RSI 的最后一个值需要大于 threshold。
    4. 过去 20 天的最低价到最新收盘价的止损比例如果超过 2%，则返回 False。

    :param rsi: pandas.Series 或 numpy.array，RSI 数据。
    :param open_prices: pandas.Series 或 numpy.array，开盘价数据。
    :param close_prices: pandas.Series 或 numpy.array，收盘价数据。
    :param min_prices: pandas.Series 或 numpy.array，最低价数据。
    :param threshold: int，RSI 阈值。
    :param threshold_days: int，过去 20 天内 RSI <= threshold 的天数需要大于的值。
    :param window: int，判断范围（如 20 天）。
    :return: bool，是否满足条件。
    """
    # 确保 RSI、开盘价、收盘价和最低价长度一致
    if len(rsi) < window or len(open_prices) < window or len(close_prices) < window or len(min_prices) < window:
        raise ValueError("RSI、开盘价、收盘价和最低价数据的长度必须至少等于 window。")
    
    # 转换为 pandas.Series
    if isinstance(rsi, np.ndarray):
        rsi = pd.Series(rsi)
    if isinstance(open_prices, np.ndarray):
        open_prices = pd.Series(open_prices)
    if isinstance(close_prices, np.ndarray):
        close_prices = pd.Series(close_prices)
    if isinstance(min_prices, np.ndarray):
        min_prices = pd.Series(min_prices)

    # 条件 1: 过去 window 天内 RSI <= threshold 的天数
    count_below_threshold = (rsi[-window:] <= threshold).sum()
    if count_below_threshold <= threshold_days:
        if classlocal.RSI_debug_en:
            #print(f"过去 {window} 天内 RSI <= {threshold} 的天数为 {count_below_threshold}，不满足条件。")
            pass
        return False

    if not check_last_three_days_conditions(open_prices, close_prices):
        return False
    # 条件 4: 检查止损比例
    min_price_last_20_days = min_prices[-window:].min()  # 过去 20 天的最低价
    latest_close_price = close_prices.iloc[-1]          # 最新收盘价
    loss_percentage = (latest_close_price - min_price_last_20_days) / min_price_last_20_days * 100
    if classlocal.RSI_debug_en:
        print(f"过去 20 天的最低价为: {min_price_last_20_days}")
        print(f"最新收盘价为: {latest_close_price}")
        print(f"止损比例为: {loss_percentage:.2f}%")

    if loss_percentage > 2:
        if classlocal.RSI_debug_en:
            print("止损比例超过 2%，不满足条件。")
        return False

    # 如果所有条件均满足
    if classlocal.RSI_debug_en:
        print("所有条件均满足！")
    return True


import pandas as pd
###################################start###########################################################################
#
###################################start###########################################################################

def convert_datetime_format(file_path):
    """
    读取 CSV 文件并转换 datetime 格式
    :param file_path: CSV 文件路径
    :return: 处理后的 DataFrame
    """
    df = pd.read_csv(file_path)

    # 转换时间格式
    df['datetime']             = pd.to_datetime(df['datetime'], format="%Y-%m-%d %H:%M:%S")
    df['datetime']             = df['datetime'].dt.strftime("%Y%m%d%H%M")

    return df

###################################start###########################################################################
#
###################################start###########################################################################

def main():
    """
    主函数：执行 CSV 文件读取和时间格式转换
    """
    # 初始化交易记录 DataFrame
    columns = ['合约', '时间', '手数', '开仓价格', '止损价格', '止盈价格', '盈利',]
    trade_log_file = r"D:\code\test\5m\trade_log.csv"

    initial_capital = 500000
    position_size = initial_capital / 20  # 每次开仓手数

    file_path = r"D:\code\test\5m\2024主力连续_5min\FG9999.XZCE_2024_5min.csv"
    
    # 读取并转换数据
    df = convert_datetime_format(file_path)
    print(df.head())

    # 读取已有交易记录（如果 CSV 不存在，则创建新的 DataFrame）
    try:
        trade_log = pd.read_csv(trade_log_file, encoding="utf-8")
    except FileNotFoundError:
        trade_log = pd.DataFrame(columns=columns)

    # 在 for 循环中逐行取出时间并传递给 custom_function
    Right = 0
    for i, datetime_value in enumerate(df['datetime']):
        if i >= 500:  # 从第 500 行后开始处理
            classlocal.Kindex_time = datetime_value
            classlocal.h_data = df.iloc[500 + i - 100: 500 + i + 1]  # 确保总行数为 100

            Right = RSI_checkout(classlocal)  # 执行 RSI 判断
        
        if Right:  # 如果 RSI 触发信号
            contract = df.iloc[500 + i]['contract']  # 获取当前合约代码
            open_price = df.iloc[500 + i]['open']  # 获取开仓价格
            stop_loss = open_price * 0.98  # 假设止损为 2% 亏损
            take_profit = open_price * 1.05  # 假设止盈为 5% 盈利

            # 记录交易信息
            new_trade = pd.DataFrame([[contract, datetime_value, position_size, open_price, stop_loss, take_profit, None]],
                                     columns=columns)
            trade_log = pd.concat([trade_log, new_trade], ignore_index=True)

            # 追加交易记录到 CSV
            trade_log.to_csv(trade_log_file, index=False, encoding="utf-8-sig")

            print(f"开仓！合约: {contract}, 时间: {datetime_value}, 手数: {position_size}, 开仓价格: {open_price}")

###################################start###########################################################################
#
###################################start###########################################################################
def decimal_places_are_rounded(floatdata,div):
    floatdata   = round(floatdata, div)
    floatdata   = '{:.2f}'.format(floatdata)
    floatdata   = float(floatdata)
    return floatdata
###################################start###########################################################################
#
###################################start###########################################################################   
def get_contract_base_info_from_csv():
    # 读取 CSV 文件
    file_path = r"D:\code\test\5m\contract_base_info_filtered.csv"
    df = pd.read_csv(file_path, encoding="utf-8")
    # 将 '代码' 设置为索引，以便快速查询保证金率
    df.set_index("代码", inplace=True)
    return df

 ###################################start###########################################################################
#获取合约的保证金以便计算手数，get_instrumentdetail 返回的是一个字典
#保证金=报价*交易单位*保证金比例=2000*10*10%=2000元
###################################start###########################################################################   
def get_signal_margin(optioncode,PreClose,df):

    # 示例：根据代码查询最低交易保证金率
    code_to_search = optioncode
    if code_to_search in df.index:
        LongMarginRatio = df.loc[code_to_search, "最低交易保证金率"]
        VolumeMultiple  = df.loc[code_to_search, "合约乘数"]
    else:
        print(f"未找到代码 {code_to_search}")
        LongMarginRatio = 0

    if LongMarginRatio <= 0 :
        LongMarginRatio    = 0.030
    #保证金
    if VolumeMultiple <= 0 :
        VolumeMultiple    = 1

    LongMarginRatio1       = LongMarginRatio + classlocal.LongMarginRatio_add
    LongMargin             = PreClose  * VolumeMultiple *LongMarginRatio1
    LongMargin2            = decimal_places_are_rounded(LongMargin,4)

    if classlocal.get_signal_margin_en:
        print('代码:\n',optioncode)
        print('最低保证金率:\n',LongMarginRatio)
        print('保证金率:\n',LongMarginRatio1)
        print('合约乘数:\n',VolumeMultiple)
        print('收盘价:\n',PreClose)
        print('所需保证金:\n',LongMargin2)

    return LongMargin2
###################################start###########################################################################
#非常重要:仓位管理函数 默认单只股票占仓位的1/10
###################################start###########################################################################
def position_opening_calculat(classlocal,buy_list):

    list_data_values        = [0,0,0,0]
    list_clolumsp           = ['code','Kindex_time','SingleNum','close']
    dit1 = dict(zip(range(0,0), list_data_values))
    #转置矩阵
    M_df = pd.DataFrame(dit1,list_clolumsp).T
    close                   = classlocal.close
    LeftMoey                = classlocal.LeftMoey
    Totalmoney              = classlocal.Total_market_cap
    Fundbal_AvailRate       = classlocal.Fundbal_AvailRate
    signal_stock_money_maxt = Totalmoney *Fundbal_AvailRate
    #单只最高可分配金额,
    signal_stock_money_max  = decimal_places_are_rounded(signal_stock_money_maxt,2)

    if signal_stock_money_max > 0:
        for code in buy_list :
            margin_t        = get_signal_margin(code,close,margin_df)
            margin          = decimal_places_are_rounded(margin_t,3)
            if margin       <= 0 :
                continue
            #最大可买金额/当前金额
            single_buy_max  = int(signal_stock_money_max/margin) 
            LeftMoey        = LeftMoey - signal_stock_money_max

            if LeftMoey     <= 0:
                LeftMoey    = 0
            classlocal.LeftMoey             = LeftMoey
            #剩余金额够买剩下的,就分配手数
            M_df.loc[code,'code']           = code
            M_df.loc[code,'close']          = close
            M_df.loc[code,'Kindex_time']    = classlocal.Kindex_time
            M_df.loc[code,'SingleNum']      = single_buy_max
            if single_buy_max >= 1 :
                M_df.loc[code,'SingleNum']  = single_buy_max
            else :
                M_df.loc[code,'SingleNum']  = 0
    return M_df
#查询股份/可用资金等

###################################start###########################################################################
#
###################################start###########################################################################
if __name__ == "__main__":
    global margin_df
    margin_df = pd.DataFrame()
    margin_df = get_contract_base_info_from_csv()

    main()
