# step1_find_high_corr_pair.py
# 目标：从指定行业中找到既高相关（>0.8）又通过协整检验（p<0.05）的优质股票对
# 为后续的统计套利（基于OU过程）提供可靠的建模基础

import akshare as ak
import pandas as pd
import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller
import warnings
warnings.filterwarnings('ignore')
import time
import json
import os

# ==================== 核心功能函数 ====================

def fetch_industry_stocks(industry_name="电力行业", max_stocks=15):
    """
    获取指定行业的股票代码和名称列表
    参数:
        industry_name: 行业名称，如"电力行业"、"银行"、"煤炭行业"等
        max_stocks: 最多获取的股票数量
    返回:
        stock_list: 股票列表，格式为 [ [代码1, 名称1], [代码2, 名称2], ... ]
    """
    print(f"正在获取 {industry_name} 的股票列表...")
    try:
        # 尝试通过行业板块获取成分股
        df = ak.stock_board_industry_cons_ths(symbol=industry_name)
        if not df.empty:
            stocks = df[['代码', '名称']].head(max_stocks).values.tolist()
            print(f"✅ 成功从行业板块获取 {len(stocks)} 只股票")
            return stocks
    except Exception as e:
        print(f"⚠️ 通过行业板块获取失败 ({e})，尝试备用方法...")
    
    # 备用方案：使用预定义的热门股票池
    print("使用预定义股票池...")
    
    industry_pools = {
        "电力行业": [
            ['600900', '长江电力'],
            ['600025', '华能水电'],
            ['600011', '华能国际'],
            ['600023', '浙能电力'],
            ['600795', '国电电力'],
            ['600886', '国投电力'],
            ['601985', '中国核电'],
            ['000543', '皖能电力']
        ],
        "银行": [
            ['600036', '招商银行'],
            ['000001', '平安银行'],
            ['601398', '工商银行'],
            ['601939', '建设银行'],
            ['601288', '农业银行'],
            ['601328', '交通银行'],
            ['600016', '民生银行'],
            ['600000', '浦发银行']
        ],
        "煤炭行业": [
            ['601088', '中国神华'],
            ['601225', '陕西煤业'],
            ['600188', '兖矿能源'],
            ['601699', '潞安环能'],
            ['601898', '中煤能源'],
            ['600123', '兰花科创'],
            ['600348', '华阳股份'],
            ['000983', '山西焦煤']
        ],
        "钢铁行业": [
            ['600019', '宝钢股份'],
            ['000898', '鞍钢股份'],
            ['000932', '华菱钢铁'],
            ['000709', '河钢股份'],
            ['600022', '山东钢铁'],
            ['601003', '柳钢股份'],
            ['000825', '太钢不锈'],
            ['002110', '三钢闽光']
        ]
    }
    
    if industry_name in industry_pools:
        stocks = industry_pools[industry_name][:max_stocks]
        print(f"✅ 从预定义池获取 {len(stocks)} 只股票")
        return stocks
    else:
        # 默认返回电力行业
        print("⚠️ 未找到指定行业，默认使用电力行业")
        return industry_pools["电力行业"][:max_stocks]

def fetch_stock_price_data(stock_code, stock_name, lookback_days=100):
    """
    获取单只股票的历史价格数据
    参数:
        stock_code: 股票代码
        stock_name: 股票名称
        lookback_days: 获取最近多少天的数据
    返回:
        price_series: 价格序列（Pandas Series），索引为日期
    """
    try:
        end_date = pd.Timestamp.now().strftime('%Y%m%d')
        start_date = (pd.Timestamp.now() - pd.DateOffset(days=lookback_days+10)).strftime('%Y%m%d')
        
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )
        
        if df.empty or len(df) < 30:  # 最少需要30个交易日
            print(f"  ⚠️  {stock_name} 数据不足 ({len(df) if not df.empty else 0}行)")
            return None
        
        # 返回收盘价序列
        price_series = df.set_index('日期')['收盘']
        price_series.name = stock_name
        return price_series
        
    except Exception as e:
        print(f"  ❌ 获取 {stock_name} 数据失败: {e}")
        return None

def test_pair_cointegration(price_a, price_b, stock_a_name, stock_b_name):
    """
    测试两只股票价格序列的协整关系
    参数:
        price_a, price_b: 价格序列
        stock_a_name, stock_b_name: 股票名称
    返回:
        dict: 包含相关系数、协整检验结果、价差平稳性等信息
    """
    # 对齐数据
    aligned_data = pd.concat([price_a, price_b], axis=1).dropna()
    if len(aligned_data) < 30:
        return None
    
    price_a_aligned = aligned_data.iloc[:, 0]
    price_b_aligned = aligned_data.iloc[:, 1]
    
    # 1. 计算相关系数
    price_corr = price_a_aligned.corr(price_b_aligned)
    log_price_corr = np.log(price_a_aligned).corr(np.log(price_b_aligned))
    
    # 2. 协整检验 (Engle-Granger 方法)
    try:
        coint_stat, pvalue_coint, critical_values = coint(price_a_aligned, price_b_aligned)
    except Exception as e:
        print(f"  ⚠️  协整检验失败: {e}")
        return None
    
    # 3. 计算对数价差并检验平稳性
    log_spread = np.log(price_a_aligned) - np.log(price_b_aligned)
    adf_stat, adf_pvalue, _, _, _, _ = adfuller(log_spread, autolag='AIC')
    
    return {
        'stock_a_code': price_a.name,
        'stock_a_name': stock_a_name,
        'stock_b_code': price_b.name,
        'stock_b_name': stock_b_name,
        'price_correlation': price_corr,
        'log_price_correlation': log_price_corr,
        'coint_pvalue': pvalue_coint,
        'coint_statistic': coint_stat,
        'adf_pvalue': adf_pvalue,
        'adf_statistic': adf_stat,
        'sample_size': len(aligned_data),
        'is_cointegrated': pvalue_coint < 0.05,  # 协整检验通过
        'is_stationary': adf_pvalue < 0.05,      # 平稳性检验通过
        'log_spread_mean': log_spread.mean(),
        'log_spread_std': log_spread.std()
    }

def find_best_cointegrated_pair(industry_name="电力行业", min_correlation=0.8):
    """
    在指定行业中寻找最佳的协整股票对
    参数:
        industry_name: 行业名称
        min_correlation: 最低相关系数要求
    返回:
        dict: 最佳配对的信息
    """
    print("="*70)
    print("🔍 开始寻找高相关且协整的股票对")
    print(f"目标行业: {industry_name}")
    print(f"最低相关系数要求: {min_correlation}")
    print("="*70)
    
    # 1. 获取股票列表
    stocks = fetch_industry_stocks(industry_name)
    
    if len(stocks) < 2:
        print("❌ 股票数量不足，无法配对")
        return None
    
    print(f"📊 共 {len(stocks)} 只股票，生成 {len(list(combinations(stocks, 2)))} 个配对组合")
    
    # 2. 获取所有股票的价格数据
    print("\n📥 正在获取股票价格数据...")
    price_data = {}
    for code, name in stocks:
        price_series = fetch_stock_price_data(code, name)
        if price_series is not None:
            price_data[(code, name)] = price_series
        time.sleep(0.2)  # 礼貌性暂停
    
    if len(price_data) < 2:
        print("❌ 成功获取数据的股票不足2只")
        return None
    
    print(f"✅ 成功获取 {len(price_data)} 只股票的价格数据")
    
    # 3. 测试所有配对
    print("\n🔬 测试股票对协整关系...")
    results = []
    tested_pairs = 0
    
    stock_items = list(price_data.items())
    for i in range(len(stock_items)):
        for j in range(i+1, len(stock_items)):
            (code_a, name_a), price_a = stock_items[i]
            (code_b, name_b), price_b = stock_items[j]
            
            tested_pairs += 1
            print(f"  测试 {tested_pairs}: {name_a} vs {name_b}", end="\r")
            
            result = test_pair_cointegration(price_a, price_b, name_a, name_b)
            if result is not None:
                # 应用筛选条件
                if (result['log_price_correlation'] >= min_correlation and 
                    result['is_cointegrated'] and 
                    result['is_stationary']):
                    results.append(result)
            
            time.sleep(0.1)  # 减轻服务器压力
    
    print(f"\n✅ 测试完成，共测试 {tested_pairs} 个配对")
    
    if not results:
        print("❌ 未找到同时满足条件的配对")
        print("💡 建议: 1. 降低相关系数要求 2. 尝试其他行业")
        return None
    
    # 4. 结果排序与展示
    df_results = pd.DataFrame(results)
    
    # 按相关系数 + 协整p值综合排序
    df_results['score'] = (
        df_results['log_price_correlation'] * 0.6 +  # 相关性权重
        (1 - df_results['coint_pvalue']) * 0.3 +     # 协整强度权重
        (1 - df_results['adf_pvalue']) * 0.1         # 平稳性权重
    )
    df_results = df_results.sort_values('score', ascending=False)
    
    # 5. 输出最佳结果
    print("\n" + "="*70)
    print("🏆 找到的优质股票对 (已通过协整检验且价差平稳)")
    print("="*70)
    
    for idx, row in df_results.head(5).iterrows():
        print(f"\n{idx+1}. {row['stock_a_name']}({row['stock_a_code']}) vs {row['stock_b_name']}({row['stock_b_code']})")
        print(f"   对数价格相关系数: {row['log_price_correlation']:.4f}")
        print(f"   协整检验 p值: {row['coint_pvalue']:.4f} {'✅' if row['is_cointegrated'] else '❌'}")
        print(f"   价差平稳性 p值: {row['adf_pvalue']:.4f} {'✅' if row['is_stationary'] else '❌'}")
        print(f"   价差均值: {row['log_spread_mean']:.4f}, 标准差: {row['log_spread_std']:.4f}")
        print(f"   样本数: {row['sample_size']} 天")
        print("   " + "-"*50)
    
    # 6. 返回最佳配对
    best_pair = df_results.iloc[0]
    best_pair_dict = {
        'industry': industry_name,
        'stock_a': {
            'code': best_pair['stock_a_code'],
            'name': best_pair['stock_a_name']
        },
        'stock_b': {
            'code': best_pair['stock_b_code'],
            'name': best_pair['stock_b_name']
        },
        'metrics': {
            'log_price_correlation': float(best_pair['log_price_correlation']),
            'coint_pvalue': float(best_pair['coint_pvalue']),
            'adf_pvalue': float(best_pair['adf_pvalue']),
            'is_cointegrated': bool(best_pair['is_cointegrated']),
            'is_stationary': bool(best_pair['is_stationary']),
            'log_spread_mean': float(best_pair['log_spread_mean']),
            'log_spread_std': float(best_pair['log_spread_std']),
            'sample_size': int(best_pair['sample_size']),
            'selection_score': float(best_pair['score'])
        },
        'selection_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'all_candidates': df_results[['stock_a_name', 'stock_b_name', 
                                      'log_price_correlation', 'coint_pvalue',
                                      'adf_pvalue', 'score']].head(10).to_dict('records')
    }
    
    print(f"\n🎯 已选择最佳配对: {best_pair_dict['stock_a']['name']} vs {best_pair_dict['stock_b']['name']}")
    
    return best_pair_dict

def save_pair_info(best_pair, filename="best_pair_info.json"):
    """保存最佳配对信息到文件"""
    os.makedirs("data", exist_ok=True)
    filepath = os.path.join("data", filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(best_pair, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 最佳配对信息已保存到 {filepath}")
    return filepath

def update_step2_code(best_pair):
    """自动更新 step2_fetch_data.py 中的股票代码"""
    try:
        with open('step2_fetch_data.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换股票定义部分
        new_stock_definition = f"""# 定义股票 (由 step1_find_high_corr_pair.py 自动选择)
stocks = [
    {{"code": "{best_pair['stock_a']['code']}", "name": "{best_pair['stock_a']['name']}"}},
    {{"code": "{best_pair['stock_b']['code']}", "name": "{best_pair['stock_b']['name']}"}}
]"""
        
        # 使用正则表达式找到并替换 stocks 定义
        import re
        pattern = r'stocks\s*=\s*\[[\s\S]*?\]'
        new_content = re.sub(pattern, new_stock_definition, content, count=1)
        
        with open('step2_fetch_data.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✅ 已自动更新 step2_fetch_data.py 中的股票代码")
        
    except Exception as e:
        print(f"⚠️ 更新 step2_fetch_data.py 失败: {e}")
        print("请手动更新 step2_fetch_data.py 中的股票代码为:")
        print(f"  stocks = [")
        print(f"    {{\"code\": \"{best_pair['stock_a']['code']}\", \"name\": \"{best_pair['stock_a']['name']}\"}},")
        print(f"    {{\"code\": \"{best_pair['stock_b']['code']}\", \"name\": \"{best_pair['stock_b']['name']}\"}}")
        print(f"  ]")

# ==================== 主程序 ====================

if __name__ == "__main__":
    # 可选择的行业列表
    available_industries = ["电力行业", "银行", "煤炭行业", "钢铁行业"]
    
    print("\n" + "="*70)
    print("📈 统计套利 - 股票对自动筛选系统")
    print("="*70)
    print("\n可选行业:")
    for i, industry in enumerate(available_industries, 1):
        print(f"  {i}. {industry}")
    
    # 用户选择行业
    while True:
        try:
            choice = input("\n请选择行业 (输入数字 1-4, 默认1): ").strip()
            if choice == "":
                choice_num = 1
                break
            choice_num = int(choice)
            if 1 <= choice_num <= len(available_industries):
                break
            else:
                print(f"请输入 1-{len(available_industries)} 之间的数字")
        except ValueError:
            print("请输入有效的数字")
    
    selected_industry = available_industries[choice_num-1]
    
    # 设置最低相关系数要求
    min_corr_input = input("最低相关系数要求 (默认0.8): ").strip()
    min_correlation = float(min_corr_input) if min_corr_input else 0.8
    
    print(f"\n⚙️ 配置:")
    print(f"  行业: {selected_industry}")
    print(f"  最低相关系数: {min_correlation}")
    print(f"  开始筛选..." + "\n")
    
    # 寻找最佳配对
    best_pair = find_best_cointegrated_pair(selected_industry, min_correlation)
    
    if best_pair:
        # 保存配对信息
        info_file = save_pair_info(best_pair)
        
        # 自动更新 step2 代码
        update_step2_code(best_pair)
        
        print("\n" + "="*70)
        print("🎉 股票对筛选完成！下一步操作:")
        print("="*70)
        print("1. 运行 python step2_fetch_data.py 下载选中的股票数据")
        print("2. 运行 python step3_analyze_spread.py 进行详细价差分析")
        print("3. 运行 python step4_ou_estimation.py 进行OU过程建模")
        print("="*70)
        
        # 显示最终选中的配对
        print(f"\n📊 最终选中的配对:")
        print(f"  {best_pair['stock_a']['name']}({best_pair['stock_a']['code']})")
        print(f"  vs")
        print(f"  {best_pair['stock_b']['name']}({best_pair['stock_b']['code']})")
        print(f"\n📈 关键指标:")
        print(f"  对数价格相关系数: {best_pair['metrics']['log_price_correlation']:.4f}")
        print(f"  协整检验 p值: {best_pair['metrics']['coint_pvalue']:.4f}")
        print(f"  价差平稳性 p值: {best_pair['metrics']['adf_pvalue']:.4f}")
        print("="*70)
    else:
        print("\n❌ 未能找到符合条件的股票对")
        print("💡 建议:")
        print("  1. 尝试其他行业")
        print("  2. 降低相关系数要求")
        print("  3. 检查网络连接")3