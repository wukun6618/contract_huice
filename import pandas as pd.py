import pandas as pd
import os

source_folder = r"C:\Users\wa\Downloads\期货主力连续_1min\2022主力连续_1min"
target_folder = r"C:\Users\wa\Downloads\code\2022主力连续_5min"

os.makedirs(target_folder, exist_ok=True)

for file_name in os.listdir(source_folder):
    if file_name.endswith(".csv"):
        source_file_path = os.path.join(source_folder, file_name)
                # 修改目标文件名：将 "1min" 替换为 "5min"
        target_file_name = file_name.replace("1min", "5min")
        target_file_path = os.path.join(target_folder, target_file_name)

        # 读取文件并设置第一列为索引
        df = pd.read_csv(source_file_path, index_col=0)
        df.index.name = "datetime"  # 重命名索引为 datetime
        df.index = pd.to_datetime(df.index)  # 确保日期格式正确

        # 重采样为5分钟数据
        df_5m = df.resample("5T").agg({
            "open": "first",
            "close": "last",
            "high": "max",
            "low": "min",
            "contract": "last",
            "avg": "mean",
        })
                # 删除空值
        df_5m = df_5m.dropna()

        # 保存文件
        df_5m.to_csv(target_file_path)
        print(f"已保存到文件：{target_file_path}")

print("完成所有文件的转换！")
