import time
import logging
import os
import yaml
import pandas as pd
import yfinance as yf
import requests
from pathlib import Path


# -------------------------- 全局日志配置 --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("api_log.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# -------------------------- 读取配置文件 --------------------------
def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

# 初始化全局配置
config = load_config()
AV_KEY = config["alpha_vantage_key"]
FRED_KEY = config["fred_api_key"]
START = config["start_date"]
END = config["end_date"]
DELAY = config["request_delay"]

# 创建原始数据存储目录
raw_dir = Path("raw_data")
raw_dir.mkdir(exist_ok=True)
for sub_folder in ["stock", "macro", "sentiment"]:
    (raw_dir / sub_folder).mkdir(exist_ok=True)

# -------------------------- 通用请求重试封装 --------------------------
def safe_request(url, max_retry=3):
    retry_count = 0
    while retry_count < max_retry:
        try:
            resp = requests.get(url, timeout=20)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                logger.warning("API 限流触发，等待延长休眠时间")
                time.sleep(DELAY * 2)
            else:
                logger.error(f"请求失败，状态码：{resp.status_code}")
        except Exception as e:
            logger.error(f"网络异常：{str(e)}")
        retry_count += 1
        time.sleep(DELAY)
    logger.error("接口请求多次重试失败，终止")
    return None

# -------------------------- 1. Yahoo Finance：JPM股票行情 --------------------------
def fetch_jpm_stock():
    logger.info("开始拉取JPM股票日度行情（Yahoo Finance）")
    ticker = yf.Ticker("JPM")
    df = ticker.history(start=START, end=END, interval="1d")
    df.reset_index(inplace=True)
    df = df[["Date", "Open", "High", "Low", "Close", "Volume", "Dividends"]]
    save_path = raw_dir / "stock" / "raw_jpm_stock_2018_2024.csv"
    df.to_csv(save_path, index=False)
    logger.info(f"JPM股票原始数据已保存至 {save_path}，样本量：{df.shape[0]}")
    time.sleep(DELAY)
    return df

# -------------------------- 2. Yahoo Finance：VIX指数 --------------------------
def fetch_vix_yf():
    logger.info("用Yahoo Finance拉取VIX指数")
    vix_ticker = yf.Ticker("^VIX")
    df = vix_ticker.history(start=START, end=END, interval="1d")
    df.reset_index(inplace=True)
    df = df[["Date", "Close"]]
    df.columns = ["Date", "VIX_Close"]
    save_path = raw_dir / "macro" / "raw_vix_2018_2024.csv"
    df.to_csv(save_path, index=False)
    logger.info(f"VIX数据保存完成：{save_path}")
    return df

# -------------------------- 3. FRED：美债1Y/5Y/10Y收益率 --------------------------
def fetch_treasury_rates():
    logger.info("开始拉取美国国债收益率（FRED）")
    series_map = {
        "Rate_1Y": "DGS1",
        "Rate_5Y": "DGS5",
        "Rate_10Y": "DGS10"
    }
    rate_df_list = []
    for col_name, series_id in series_map.items():
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&api_key={FRED_KEY}"
            f"&observation_start={START}&observation_end={END}&file_type=json"
        )
        res = safe_request(url)
        if not res:
            logger.error(f"{series_id} 利率数据拉取失败")
            continue
        obs = res["observations"]
        temp_df = pd.DataFrame(obs)[["date", "value"]]
        temp_df.columns = ["Date", col_name]
        temp_df[col_name] = pd.to_numeric(temp_df[col_name], errors="coerce")
        rate_df_list.append(temp_df)
        time.sleep(DELAY)
    # 合并多期限利率数据
    final_df = rate_df_list[0]
    for df in rate_df_list[1:]:
        final_df = pd.merge(final_df, df, on="Date", how="outer")
    save_path = raw_dir / "macro" / "raw_treasury_rate_2018_2024.csv"
    final_df.to_csv(save_path, index=False)
    logger.info(f"国债利率原始数据已保存至 {save_path}")
    return final_df

# -------------------------- 统一入口：一键执行全部API拉取 --------------------------
def run_all_data_pull():
    logger.info("===== 启动全量原始数据采集任务 =====")
    # 拉取三类核心数据
    fetch_jpm_stock()
    fetch_vix_yf()
    fetch_treasury_rates()
    logger.info("===== 全部API数据拉取完成，原始数据存放于 ./raw_data/ =====")

# 程序入口测试
if __name__ == "__main__":
    run_all_data_pull()
