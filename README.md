# 股票配对交易策略（统计套利）

本项目实现了一个基于**协整检验 + Z-Score 均值回归**的A股配对交易策略，采用Python实现了从股票筛选、数据获取、统计分析到信号生成的全过程。

## 项目结构
├── step1_find_high_corr_pair.py    # 筛选高相关性、协整、价差平稳的股票对
├── step2_fetch_data_pair1.py      # 下载指定股票对历史数据并计算对数价差
├── step3_analyze_spread_pair1.py  # 分析价差，绘制 Z-Score 与交易信号
├── data/                          # 存放下载的数据
└── selected_pairs.json            # Step1 的筛选结果

## 使用方法

1.安装依赖。
   bash
pip install akshare pandas numpy statsmodels matplotlib

2.运行 Step1 筛选股票对。

3.根据 Step1 输出，运行对应 Step2（例如中信证券 vs 申万宏源）。

4.运行 Step3 分析并绘图。

## 示例结果

- 筛选出券商行业 10 只股票，两两组合测试，找到多组高相关性、协整、价差平稳的配对。
- 对中信证券 vs 申万宏源进行完整流程，生成 Z-Score 图表和交易信号。

## 依赖

- Python 3.8+
- akshare
- pandas
- numpy
- statsmodels
- matplotlib


## 策略原理

### 1. 模型选择：Ornstein-Uhlenbeck (OU) 过程
- **为什么用 OU 模型**：OU 过程是描述“均值回归”现象的经典随机过程，其数学形式 `dX_t = θ(μ - X_t)dt + σ dW_t` 天然适合建模围绕长期均值波动的价差。
- **参数估计**：本项目在**极大似然估计法**的基础上，通过引入 OU 过程的欧拉离散化形式推导出参数 (θ, μ, σ) 的解析解，提高了估计效率和数值稳定性。改进了该过程在基础的极大似然估计中无解析解的问题。
- **优势**：相比简单的移动平均，OU 模型提供了均值回归速度 (θ)、长期均衡水平 (μ) 和波动率 (σ) 的量化估计，使策略信号更具统计基础。同时估计效率大大提高。

## 作者

[佟高哲]

## 许可证

MIT
