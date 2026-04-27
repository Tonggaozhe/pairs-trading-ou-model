# step2_fetch_data_pair1.py
# 下载 中信证券 vs 申万宏源 的历史数据，计算对数价差

import akshare as ak
import pandas as pd
import numpy as np
import os

# 股票配置
stock_a = {'code': '600030', 'name': '中信证券'}
stock_b = {'code': '000166', 'name': '申万宏源'}

# 输出目录
output_dir = 'data'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'pair1_citic_vs_shenwan.csv')

# 时间范围
end_date = pd.Timestamp.now().strftime('%Y%m%d')
start_date = (pd.Timestamp.now() - pd.DateOffset(years=3)).strftime('%Y%m%d')

print(f"开始下载 {stock_a['name']}({stock_a['code']}) 数据...")
df_a = ak.stock_zh_a_hist(symbol=stock_a['code'], start_date=start_date, end_date=end_date, adjust='qfq')

print(f"开始下载 {stock_b['name']}({stock_b['code']}) 数据...")
df_b = ak.stock_zh_a_hist(symbol=stock_b['code'], start_date=start_date, end_date=end_date, adjust='qfq')

# 合并数据
df = pd.merge(df_a[['日期', '收盘']], df_b[['日期', '收盘']], on='日期', how='inner')
df.columns = ['date', 'price_a', 'price_b']

# 计算对数价差
df['log_price_a'] = np.log(df['price_a'])
df['log_price_b'] = np.log(df['price_b'])
df['log_spread'] = df['log_price_a'] - df['log_price_b']

# 保存
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"数据已保存到 {output_file}，共 {len(df)} 行")