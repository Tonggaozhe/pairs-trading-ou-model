import akshare as ak
import pandas as pd
import time

def fetch_stock_data_akshare(stock_code, stock_name, max_retries=5):
    """
    使用 AKShare 获取股票数据，同时引入自动重试功能
    """
    print(f"--- 开始获取 {stock_name}({stock_code}) 数据 (使用 AKShare) ---")
    
    # 构造 AKShare 需要的日期参数
    # 获取最近一年的数据
    end_date = pd.Timestamp.now().strftime('%Y%m%d')
    start_date = (pd.Timestamp.now() - pd.DateOffset(years=1)).strftime('%Y%m%d')

    for attempt in range(max_retries):
        try:
            print(f"正在获取 {stock_name} 数据... (第 {attempt + 1} 次尝试)")
            
           
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq" 
            )

            # 检查数据是否为空
            if df.empty:
                print(f"⚠️ 第 {attempt + 1} 次尝试：返回的数据为空。")
            else:
                print(f"✅ 第 {attempt + 1} 次尝试：数据获取成功！")
                print(f"\n{stock_name} 前5行数据：")
                print(df.head())
                return df  # 成功拿到数据，直接返回

        except Exception as e:
            print(f"❌ 第 {attempt + 1} 次尝试：发生错误：{e}")

        # 如果没成功，等待一段时间再重试 (指数退避)
        wait_time = 2 ** attempt  # 2, 4, 8, 16... 秒
        print(f"等待 {wait_time} 秒后重试...")
        time.sleep(wait_time)

    print(f"❌ 经过 {max_retries} 次尝试，仍然无法获取 {stock_name} 的数据。")
    return None

# --- 主程序 ---

# 定义要获取的股票
stocks = [
    {"code": "600519", "name": "贵州茅台"},
    {"code": "000858", "name": "五粮液"}
]

# 循环获取
all_data = {}
for stock in stocks:
    df = fetch_stock_data_akshare(stock["code"], stock["name"])
    if df is not None:
        all_data[stock["name"]] = df
        # 保存数据到 Excel
        filename = f"{stock['name']}_data.xlsx"
        df.to_excel(filename, index=False)
        print(f"✅ {stock['name']} 数据已保存到 {filename}")
    else:
        print(f"❌ {stock['name']} 数据获取失败，跳过保存。")

print("\n--- 所有任务完成 ---")