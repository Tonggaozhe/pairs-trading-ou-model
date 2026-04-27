# step2_fetch_data.py
# 目标：直接指定股票代码，下载历史数据，并计算对数价差
# 注意：这里直接使用你在 Step 1 中找到的最佳配对：中国神华 vs 陕西煤业

import akshare as ak
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# ==================== 配置区 ====================
# 在这里直接指定你要下载的股票对
STOCK_A = {
    "code": "601088",  # 中国神华
    "name": "中国神华"
}

STOCK_B = {
    "code": "601225",  # 陕西煤业
    "name": "陕西煤业"
}

# 输出目录
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 输出文件名
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "pair_trading_data.csv")

# 数据获取时间范围
END_DATE = datetime.now().strftime("%Y%m%d")
START_DATE = (datetime.now() - timedelta(days=3 * 365)).strftime("%Y%m%d")  # 获取3年数据

def fetch_stock_data(stock_code, stock_name, start_date, end_date):
    """
    使用 AkShare 获取单只股票的历史行情
    """
    print(f"正在下载 {stock_name}({stock_code}) 的数据...")
    try:
        # 获取前复权数据
        df = ak.stock_zh_a_hist(
            symbol=stock_code, 
            start_date=start_date, 
            end_date=end_date, 
            adjust="qfq"  # 前复权
        )
        
        if df.empty:
            print(f"⚠️ 股票 {stock_name}({stock_code}) 在指定期间没有数据")
            return None
            
        # 重命名列
        df.rename(columns={
            "日期": "date", 
            "收盘": "close"
        }, inplace=True)
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # 只保留收盘价
        return df[['close']]
        
    except Exception as e:
        print(f"❌ 下载 {stock_name}({stock_code}) 失败: {e}")
        return None

def main():
    print("="*60)
    print("📥 Step 2: 下载指定股票对的历史数据")
    print("="*60)
    print(f"股票A: {STOCK_A['name']} ({STOCK_A['code']})")
    print(f"股票B: {STOCK_B['name']} ({STOCK_B['code']})")
    print(f"时间范围: {START_DATE} 至 {END_DATE}")
    print("="*60)
    
    # 1. 下载数据
    df_a = fetch_stock_data(STOCK_A["code"], STOCK_A["name"], START_DATE, END_DATE)
    df_b = fetch_stock_data(STOCK_B["code"], STOCK_B["name"], START_DATE, END_DATE)
    
    if df_a is None or df_b is None:
        print("❌ 数据下载失败，请检查股票代码或网络连接")
        return
    
    # 2. 合并数据
    print("\n正在合并数据...")
    merged_df = pd.concat([df_a['close'], df_b['close']], axis=1, join='inner')
    merged_df.columns = [f'price_{STOCK_A["name"][:2]}', f'price_{STOCK_B["name"][:2]}']
    
    # 处理缺失值
    merged_df.dropna(inplace=True)
    
    if len(merged_df) < 50:
        print(f"⚠️ 有效数据天数太少 ({len(merged_df)}天)，无法进行后续分析")
        return
    
    # 3. 计算核心指标
    # 对数价格
    merged_df['log_price_a'] = np.log(merged_df.iloc[:, 0])
    merged_df['log_price_b'] = np.log(merged_df.iloc[:, 1])
    
    # 对数价差
    merged_df['spread'] = merged_df['log_price_a'] - merged_df['log_price_b']
    
    # 标准化价差 (z-score)
    spread_mean = merged_df['spread'].mean()
    spread_std = merged_df['spread'].std()
    merged_df['spread_zscore'] = (merged_df['spread'] - spread_mean) / spread_std
    
    # 4. 保存数据
    merged_df.to_csv(OUTPUT_CSV)
    print(f"\n✅ 数据处理完成！")
    print(f"   数据已保存至: {OUTPUT_CSV}")
    
    # 5. 打印统计信息
    print(f"\n📊 数据统计信息:")
    print(f"   总数据天数: {len(merged_df)}")
    print(f"   价差均值: {spread_mean:.6f}")
    print(f"   价差标准差: {spread_std:.6f}")
    
    # 相关性计算（验证一下）
    correlation = merged_df.iloc[:, 0].corr(merged_df.iloc[:, 1])
    log_correlation = merged_df['log_price_a'].corr(merged_df['log_price_b'])
    print(f"   价格相关系数: {correlation:.4f}")
    print(f"   对数价格相关系数: {log_correlation:.4f}")
    
    # 6. 预览数据
    print(f"\n📄 数据预览 (前5行):")
    print(merged_df.head())
    
    print("\n" + "="*60)
    print("✅ Step 2 完成！可以开始运行 Step 3 了")
    print("="*60)

if __name__ == "__main__":
    main()