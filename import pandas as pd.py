import pandas as pd
import os

source_folder = r"D:\code\contract_huice\2025主力连续_5min"
target_folder = r"D:\code\contract_huice\2025主力连续_15min"

os.makedirs(target_folder, exist_ok=True)

for file_name in os.listdir(source_folder):
    if file_name.endswith(".csv"):
        source_file_path = os.path.join(source_folder, file_name)
        
        # 修改目标文件名：将 "5min" 替换为 "15min"
        target_file_name = file_name.replace("5min", "15min")
        target_file_path = os.path.join(target_folder, target_file_name)

        # 读取文件并设置第一列为索引
        df = pd.read_csv(source_file_path, index_col=0)
        df.index.name = "datetime"  # 重命名索引为 datetime
        df.index = pd.to_datetime(df.index)  # 确保日期格式正确

        # 重采样为15分钟数据
        df_15m = df.resample("15min").agg({
        "open": "first",
        "close": "last",
        "high": "max",
        "low": "min",
        "contract": "last",
        "avg": "mean",
        })

        # 删除空值
        df_15m = df_15m.dropna()

        # 保存文件
        df_15m.to_csv(target_file_path)
        print(f"已保存到文件：{target_file_path}")

print("完成所有文件的转换！")