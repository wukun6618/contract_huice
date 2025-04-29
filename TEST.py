import pandas as pd
import numpy as np
from datetime import datetime as dt


class b():
    pass
classlocal = b()

classlocal.shipan_en                = 0                 ##False :模拟账号；True :实盘账号；
# 行情设置开关
classlocal.lefthand_checken         = 0                 # 1 打开行情止损 0 关闭
classlocal.LongMarginRatio_add      = 0.09              # 在最低保证金基础增加的比例
classlocal.close_atr_trade_en       = 0                 #0：关掉ART 1:打开ATR行情止盈

classlocal.Fundbal_AvailRate        = 0.05              #单只占总资金仓位
classlocal.get_signal_margin_en     = 1                 #打印保证金率


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
        print('df:\n',df)
        print('代码:\n',optioncode)
        print('最低保证金率:\n',LongMarginRatio)
        print('保证金率:\n',LongMarginRatio1)
        print('合约乘数:\n',VolumeMultiple)
        print('收盘价:\n',PreClose)
        print('所需保证金:\n',LongMargin2)
    
    #返回的是保证金
    return LongMargin2
df         = get_contract_base_info_from_csv()
optioncode = 'FG'
PreClose   = 1367
get_signal_margin(optioncode,PreClose,df)