# step3_analyze_spread_pair1.py
# 分析对数价差，绘制 Z-Score 和交易信号

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 读取数据
input_file = 'data/pair1_citic_vs_shenwan.csv'
if not os.path.exists(input_file):
    print(f"错误：找不到文件 {input_file}")
    exit()

df = pd.read_csv(input_file)

# 计算 Z-Score
df['z_score'] = (df['log_spread'] - df['log_spread'].mean()) / df['log_spread'].std()

# 设置交易信号
entry_threshold = 1.0
exit_threshold = 0.3

df['signal'] = 0
df.loc[df['z_score'] > entry_threshold, 'signal'] = -1  # 做空价差（卖出A，买入B）
df.loc[df['z_score'] < -entry_threshold, 'signal'] = 1  # 做多价差（买入A，卖出B）
df.loc[np.abs(df['z_score']) < exit_threshold, 'signal'] = 0  # 平仓

# 绘图
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# 图1：对数价差
axes[0].plot(df['date'], df['log_spread'], label='对数价差', color='blue')
axes[0].axhline(y=df['log_spread'].mean(), color='red', linestyle='--', label=f'长期均值 μ={df["log_spread"].mean():.4f}')
axes[0].set_title('对数价差走势与长期均衡带')
axes[0].legend()
axes[0].grid(True)

# 图2：Z-Score
axes[1].plot(df['date'], df['z_score'], label='Z-Score', color='green')
axes[1].axhline(y=entry_threshold, color='red', linestyle='--', label=f'开仓阈值 ±{entry_threshold}')
axes[1].axhline(y=-entry_threshold, color='red', linestyle='--')
axes[1].axhline(y=exit_threshold, color='orange', linestyle=':', label=f'平仓阈值 ±{exit_threshold}')
axes[1].axhline(y=-exit_threshold, color='orange', linestyle=':')
axes[1].set_title('标准化价差（Z-Score）与交易区间')
axes[1].legend()
axes[1].grid(True)

# 图3：交易信号
axes[2].bar(df['date'], df['signal'], label='交易信号', color='purple')
axes[2].set_title('生成的交易信号')
axes[2].set_yticks([-1, 0, 1])
axes[2].set_yticklabels(['做空', '空仓', '做多'])
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()

print("分析完成，图表已显示")