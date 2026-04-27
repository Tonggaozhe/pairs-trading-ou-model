# step0_find_high_corr_pair.py
# 目标：从同行业股票池中，自动筛选出相关性最高的股票对

import akshare as ak
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from itertools import combinations
import time

def fetch_industry_stocks(industry_code="电力行业", max_stocks=20):
    """
    获取指定行业板块的股票列表
    """
    print(f"正在获取 {industry_code} 的股票列表...")
    try:
        # 获取行业成分股
        df = ak.stock_board_industry_cons_ths(symbol=industry_code)
        if df.empty:
            print("获取行业列表失败，尝试备用方法...")
            # 备用：使用沪深300成分股
            df = ak.index_stock_cons(symbol="000300")
        
        # 提取股票代码和名称
        stock_list = df[['代码', '名称']].head(max_stocks).values.tolist()
        print(f"成功获取 {len(stock_list)} 只股票")
        return stock_list
        
    except Exception as e:
        print(f"获取行业股票失败: {e}")
        # 备用：使用预定义的电力股
        return [
            ['600900', '长江电力'],
            ['600025', '华能水电'],
            ['600011', '华能国际'],
            ['600023', '浙能电力'],
            ['600795', '国电电力'],
            ['600886', '国投电力'],
            ['601985', '中国核电'],
            ['000543', '皖能电力'],
            ['000690', '宝新能源'],
            ['000722', '湖南发展']
        ]

def calculate_pair_correlation(stock_a, stock_b, days=30):
    """
    计算两只股票最近N天的收益率相关性
    """
    try:
        # 获取最近N+5天的数据
        end_date = pd.Timestamp.now().strftime('%Y%m%d')
        start_date = (pd.Timestamp.now() - pd.DateOffset(days=days+5)).strftime('%Y%m%d')
        
        # 获取股票A数据
        df_a = ak.stock_zh_a_hist(
            symbol=stock_a[0], 
            period="daily", 
            start_date=start_date, 
            end_date=end_date, 
            adjust="qfq"
        )
        # 获取股票B数据
        df_b = ak.stock_zh_a_hist(
            symbol=stock_b[0], 
            period="daily", 
            start_date=start_date, 
            end_date=end_date, 
            adjust="qfq"
        )
        
        if df_a.empty or df_b.empty:
            return None, None, None
        
        # 对齐数据
        df_a = df_a.set_index('日期')[['收盘']].rename(columns={'收盘': 'close_a'})
        df_b = df_b.set_index('日期')[['收盘']].rename(columns={'收盘': 'close_b'})
        df_merged = pd.concat([df_a, df_b], axis=1).dropna()
        
        if len(df_merged) < days//2:  # 数据太少
            return None, None, None
        
        # 计算收益率相关性
        df_merged['ret_a'] = df_merged['close_a'].pct_change()
        df_merged['ret_b'] = df_merged['close_b'].pct_change()
        df_merged = df_merged.dropna()
        
        if len(df_merged) < 10:
            return None, None, None
        
        correlation = df_merged['ret_a'].corr(df_merged['ret_b'])
        price_corr = df_merged['close_a'].corr(df_merged['close_b'])
        
        return correlation, price_corr, len(df_merged)
        
    except Exception as e:
        print(f"计算 {stock_a[1]} 和 {stock_b[1]} 相关性时出错: {e}")
        return None, None, None

def find_best_pair(industry_code="电力行业", top_n=5):
    """
    寻找相关性最高的股票对
    """
    print("="*60)
    print("开始自动寻找高相关性股票对")
    print(f"目标行业: {industry_code}")
    print("="*60)
    
    # 1. 获取股票列表
    stocks = fetch_industry_stocks(industry_code)
    
    if len(stocks) < 2:
        print("股票数量不足，无法配对")
        return None
    
    # 2. 生成所有可能的配对组合
    pairs = list(combinations(stocks, 2))
    print(f"共生成 {len(pairs)} 个股票对进行测试...")
    
    # 3. 计算每个配对的相关性
    results = []
    for i, (stock_a, stock_b) in enumerate(pairs):
        print(f"进度: {i+1}/{len(pairs)} - 测试 {stock_a[1]}({stock_a[0]}) vs {stock_b[1]}({stock_b[0]})", end="\r")
        
        corr, price_corr, sample_size = calculate_pair_correlation(stock_a, stock_b)
        
        if corr is not None and sample_size is not None:
            results.append({
                'stock_a_code': stock_a[0],
                'stock_a_name': stock_a[1],
                'stock_b_code': stock_b[0],
                'stock_b_name': stock_b[1],
                'correlation': corr,
                'price_correlation': price_corr,
                'sample_size': sample_size
            })
        
        # 礼貌性暂停，避免请求过快
        time.sleep(0.5)
    
    print("\n" + "="*60)
    
    if not results:
        print("未能计算出任何有效的相关性结果")
        return None
    
    # 4. 按相关性排序
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('correlation', ascending=False)
    
    # 5. 输出结果
    print("相关性最高的股票对 (前5名):")
    print("-"*80)
    for i, row in df_results.head(top_n).iterrows():
        print(f"{i+1}. {row['stock_a_name']}({row['stock_a_code']}) vs {row['stock_b_name']}({row['stock_b_code']})")
        print(f"   收益率相关性: {row['correlation']:.4f}")
        print(f"   价格相关性: {row['price_correlation']:.4f}")
        print(f"   样本数: {row['sample_size']}")
        print("-"*80)
    
    # 6. 返回最佳配对
    best_pair = df_results.iloc[0]
    print(f" 选择最佳配对: {best_pair['stock_a_name']} vs {best_pair['stock_b_name']}")
    print(f"   收益率相关性: {best_pair['correlation']:.4f}")
    
    return {
        'stock_a': {
            'code': best_pair['stock_a_code'],
            'name': best_pair['stock_a_name']
        },
        'stock_b': {
            'code': best_pair['stock_b_code'],
            'name': best_pair['stock_b_name']
        },
        'correlation': best_pair['correlation'],
        'price_correlation': best_pair['price_correlation'],
        'all_pairs': df_results.head(10)  # 保存前10对供参考
    }

def save_pair_info(best_pair, filename="best_pair_info.json"):
    """保存最佳配对信息到文件"""
    import json
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(best_pair, f, ensure_ascii=False, indent=2)
    print(f"✅ 最佳配对信息已保存到 {filename}")

def update_step1_code(best_pair):
    """自动更新 step1_fetch_data.py 中的股票代码"""
    try:
        with open('step1_fetch_data.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换股票定义部分
        new_stock_definition = f"""# 定义股票 (由 step0_find_high_corr_pair.py 自动选择)
stocks = [
    {{"code": "{best_pair['stock_a']['code']}", "name": "{best_pair['stock_a']['name']}"}},
    {{"code": "{best_pair['stock_b']['code']}", "name": "{best_pair['stock_b']['name']}"}}
]"""
        
        # 使用正则表达式找到并替换 stocks 定义
        import re
        pattern = r'stocks\s*=\s*\[[\s\S]*?\]'
        new_content = re.sub(pattern, new_stock_definition, content, count=1)
        
        with open('step1_fetch_data.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✅ 已自动更新 step1_fetch_data.py 中的股票代码")
        
    except Exception as e:
        print(f"⚠️ 更新 step1_fetch_data.py 失败: {e}")
        print("请手动更新 step1_fetch_data.py 中的股票代码为:")
        print(f"  stocks = [")
        print(f"    {{\"code\": \"{best_pair['stock_a']['code']}\", \"name\": \"{best_pair['stock_a']['name']}\"}},")
        print(f"    {{\"code\": \"{best_pair['stock_b']['code']}\", \"name\": \"{best_pair['stock_b']['name']}\"}}")
        print(f"  ]")

# --- 主程序 ---
if __name__ == "__main__":
    # 可以选择不同的行业
    industries = ["电力行业", "银行", "煤炭行业", "钢铁行业", "汽车整车"]
    
    print("可选行业列表:")
    for i, industry in enumerate(industries, 1):
        print(f"  {i}. {industry}")
    
    # 默认使用电力行业
    selected_industry = industries[0]  # 电力行业
    
    # 寻找最佳配对
    best_pair = find_best_pair(selected_industry)
    
    if best_pair:
        # 保存配对信息
        save_pair_info(best_pair)
        
        # 自动更新 step1_fetch_data.py
        update_step1_code(best_pair)
        
        print("\n" + "="*60)
        print("🎉 下一步操作:")
        print("1. 运行 python step1_fetch_data.py 下载选中的股票数据")
        print("2. 运行 python step2_analyze_spread.py 分析价差")
        print("="*60)
    else:
        print("❌ 未能找到合适的高相关性股票对")