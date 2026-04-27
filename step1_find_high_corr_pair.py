# step1_find_high_corr_pair.py
# 筛选高相关性且通过协整检验的股票对
# 重点：券商行业，10只股票，两两组合测试

import akshare as ak
import pandas as pd
import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller
import warnings
warnings.filterwarnings('ignore')
import time
import os

# 定义行业股票池：券商行业，10只
industry_stocks = [
    ['600030', '中信证券'],
    ['600999', '招商证券'],
    ['601688', '华泰证券'],
    ['000776', '广发证券'],
    ['600837', '海通证券'],
    ['601788', '光大证券'],
    ['002736', '国信证券'],
    ['600109', '国金证券'],
    ['601162', '天风证券'],
    ['000166', '申万宏源']
]

# 设置时间范围
end_date = pd.Timestamp.now().strftime('%Y%m%d')
start_date = (pd.Timestamp.now() - pd.DateOffset(years=3)).strftime('%Y%m%d')

# 存储结果
results = []

print(f"开始筛选，共 {len(industry_stocks)} 只股票，组合数：{len(list(combinations(range(len(industry_stocks)), 2)))}")

for i, j in combinations(range(len(industry_stocks)), 2):
    code_a, name_a = industry_stocks[i]
    code_b, name_b = industry_stocks[j]
    
    print(f"正在测试：{name_a} vs {name_b}")
    
    try:
        # 获取历史数据
        df_a = ak.stock_zh_a_hist(symbol=code_a, start_date=start_date, end_date=end_date, adjust='qfq')
        df_b = ak.stock_zh_a_hist(symbol=code_b, start_date=start_date, end_date=end_date, adjust='qfq')
        
        if df_a.empty or df_b.empty:
            continue
            
        # 合并数据
        df = pd.merge(df_a[['日期', '收盘']], df_b[['日期', '收盘']], on='日期', how='inner')
        df.columns = ['date', 'price_a', 'price_b']
        df['log_price_a'] = np.log(df['price_a'])
        df['log_price_b'] = np.log(df['price_b'])
        df['log_spread'] = df['log_price_a'] - df['log_price_b']
        
        # 检查数据长度
        if len(df) < 60:
            continue
            
        # 计算相关系数
        corr = df['log_price_a'].corr(df['log_price_b'])
        
        # 协整检验
        coint_stat, coint_pvalue = coint(df['log_price_a'], df['log_price_b'])
        
        # ADF检验价差平稳性
        adf_stat, adf_pvalue, _, _, _ = adfuller(df['log_spread'])
        
        # 如果满足条件：相关系数>0.8，协整p<0.05，ADF p<0.05
        if corr > 0.8 and coint_pvalue < 0.05 and adf_pvalue < 0.05:
            spread_mean = df['log_spread'].mean()
            spread_std = df['log_spread'].std()
            spread_range = spread_std * 2  # 2倍标准差作为波动范围
            
            results.append({
                'stock_a': {'code': code_a, 'name': name_a},
                'stock_b': {'code': code_b, 'name': name_b},
                'metrics': {
                    'log_price_correlation': round(corr, 4),
                    'coint_pvalue': round(coint_pvalue, 4),
                    'adf_pvalue': round(adf_pvalue, 4),
                    'log_spread_mean': round(spread_mean, 4),
                    'log_spread_std': round(spread_std, 4),
                    '_spread_ralognge': round(spread_range, 4)
                }
            })
            
            print(f"找到配对：{name_a} vs {name_b}，相关系数：{corr:.4f}，协整p：{coint_pvalue:.4f}")
            
    except Exception as e:
        print(f"错误：{e}")
        continue

# 输出结果
if results:
    print("\n找到的优质股票对（已通过协整检验且价差平稳）：")
    for i, pair in enumerate(results, 1):
        print(f"{i}. {pair['stock_a']['name']}({pair['stock_a']['code']}) vs {pair['stock_b']['name']}({pair['stock_b']['code']})")
        print(f"   对数价格相关系数：{pair['metrics']['log_price_correlation']}")
        print(f"   协整检验 p值：{pair['metrics']['coint_pvalue']}")
        print(f"   价差平稳性 p值：{pair['metrics']['adf_pvalue']}")
        print(f"   价差均值：{pair['metrics']['log_spread_mean']:.4f}，标准差：{pair['metrics']['log_spread_std']:.4f}")
        print(f"   价差波动范围：{pair['metrics']['log_spread_range']:.4f}")
        print(f"   样本数：{len(df)} 天")
        print()
else:
    print("未找到符合条件的股票对")

# 保存结果到文件
with open('selected_pairs.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("结果已保存到 selected_pairs.json")