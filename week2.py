import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import pandas as pd
import numpy as np
from pathlib import Path
import nltk

# 新闻情感依赖
try:
    from newsapi import NewsApiClient
    from textblob import TextBlob
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    HAS_NEWS = True# 绑定远程仓库
    NEWS_API_KEY = "6135d732b3bc4e088c3f7a1d3adbf9b8"
except ImportError:
    HAS_NEWS = False
    NEWS_API_KEY = ""
    print("警告：未安装新闻相关依赖，新闻情绪将统一填充中性0.5")


# 抓取新闻并生成日度新闻情绪
def get_news_sentiment(start_date: str, end_date: str):
    if not HAS_NEWS or len(NEWS_API_KEY.strip()) == 0:
        return None
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        # 文档指定检索关键词
        query = "JPM OR FOMC OR US stock market OR earnings reports OR stock options OR volatility"
        articles = newsapi.get_everything(
            q=query,
            from_param=start_date,
            to=end_date,
            language="en",
            sort_by="publishedAt"
        )["articles"]

        sent_records = []
        for art in articles:
            pub_dt = pd.to_datetime(art["publishedAt"]).date()
            content = str(art["title"]) + " " + str(art["content"] or "")
            # TextBlob极性：[-1 极度负面, 1 极度正面]
            polarity = TextBlob(content).sentiment.polarity
            sent_records.append({"Date": pd.Timestamp(pub_dt), "polarity": polarity})

        if not sent_records:
            return None
        news_df = pd.DataFrame(sent_records)
        # 按日期取平均情感
        daily_sent = news_df.groupby("Date")["polarity"].mean().reset_index()
        # [-1,1] 映射至 [0(极度看空),1(极度看多)]
        daily_sent["news_sentiment"] = (daily_sent["polarity"] + 1) / 2
        return daily_sent[["Date", "news_sentiment"]]
    except Exception as e:
        print(f"新闻接口请求失败: {e}，当日新闻情绪使用0.5中性值")
        return None


def load_and_clean_data():
    # 1. 加载数据
    base_path = Path(__file__).parent
    treasury = pd.read_csv(base_path / "raw_data" / "macro" / "raw_treasury_rate_2018_2024.csv")
    vix = pd.read_csv(base_path / "raw_data" / "macro" / "raw_vix_2018_2024.csv")
    jpm = pd.read_csv(base_path / "raw_data" / "stock" / "raw_jpm_stock_2018_2024.csv")

    # 2. 统一日期格式，去除时区信息
    treasury['Date'] = pd.to_datetime(treasury['Date'])
    vix['Date'] = pd.to_datetime(vix['Date'].str[:10])
    jpm['Date'] = pd.to_datetime(jpm['Date'].str[:10])

    # 3. 缺失值线性插值
    treasury = treasury.set_index('Date').sort_index()
    treasury = treasury.interpolate(method='linear', axis=0).reset_index()

    # 4. IQR异常值处理
    def remove_outliers_iqr(df, columns):
        df_clean = df.copy()
        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean.loc[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound), col] = np.nan
        df_clean = df_clean.set_index('Date').sort_index()
        df_clean = df_clean.interpolate(method='linear', axis=0).reset_index()
        return df_clean

    treasury_cols = ['Rate_1Y', 'Rate_5Y', 'Rate_10Y']
    vix_cols = ['VIX_Close']
    jpm_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends']

    treasury = remove_outliers_iqr(treasury, treasury_cols)
    vix = remove_outliers_iqr(vix, vix_cols)
    jpm = remove_outliers_iqr(jpm, jpm_cols)

    # 5. 多表按日期内连接对齐
    merged = pd.merge(treasury, vix, on='Date', how='inner')
    merged = pd.merge(merged, jpm, on='Date', how='inner')
    merged = merged.sort_values('Date').reset_index(drop=True)

    return merged


def build_features(df):
    df = df.copy()

    # ========== 基础特征 ==========
    df['JPM_Daily_Return'] = df['Close'].pct_change()
    df['JPM_Volatility_20D'] = df['JPM_Daily_Return'].rolling(window=20).std() * np.sqrt(252)
    df['Dividend_Growth'] = df['Dividends'].pct_change()
    df['Dividend_Growth'] = df['Dividend_Growth'].fillna(0)
    df.loc[df['Dividends'] == 0, 'Dividend_Growth'] = 0
    df['JPM_MA5'] = df['Close'].rolling(window=5).mean()
    df['JPM_MA20'] = df['Close'].rolling(window=20).mean()
    df['Volume_Change'] = df['Volume'].pct_change().fillna(0)

    # ========== 利率特征 ==========
    df['Rate_10Y_Momentum_5D'] = df['Rate_10Y'].pct_change(periods=5)
    df['Spread_10Y_1Y'] = df['Rate_10Y'] - df['Rate_1Y']
    df['Spread_10Y_5Y'] = df['Rate_10Y'] - df['Rate_5Y']
    df['Spread_5Y_1Y'] = df['Rate_5Y'] - df['Rate_1Y']

    # ========== VIX基础特征 ==========
    df['VIX_Daily_Change'] = df['VIX_Close'].pct_change()
    df['VIX_MA20'] = df['VIX_Close'].rolling(window=20).mean()
    df['VIX_JPM_Correlation_20D'] = df['VIX_Close'].rolling(window=20).corr(df['Close'])

    # ========== 1.VIX衍生情绪分 ==========
    vix_min = df['VIX_Close'].min()
    vix_max = df['VIX_Close'].max()
    df['vix_sentiment'] = 1 - (df['VIX_Close'] - vix_min) / (vix_max - vix_min)

    # ========== 2.合并新闻情绪分 ==========
    start = df["Date"].min().strftime("%Y-%m-%d")
    end = df["Date"].max().strftime("%Y-%m-%d")
    news_df = get_news_sentiment(start, end)
    if news_df is not None:
        df = pd.merge(df, news_df, on="Date", how="left")
        df["news_sentiment"] = df["news_sentiment"].fillna(0.5)
    else:
        df["news_sentiment"] = 0.5

    # ========== 3.文档要求：综合市场情绪 Comprehensive Score 0~1 ==========
    # 权重：新闻0.6，VIX波动率0.4
    weight_news = 0.6
    weight_vix = 0.4
    df["comprehensive_market_sentiment"] = weight_news * df["news_sentiment"] + weight_vix * df["vix_sentiment"]

    # 剔除滚动窗口NaN
    df = df.dropna().reset_index(drop=True)
    return df


def main():
    print("开始数据清洗...")
    cleaned_data = load_and_clean_data()
    print(f"清洗后数据量: {cleaned_data.shape[0]} 行, {cleaned_data.shape[1]} 列")

    print("开始特征构建...")
    featured_data = build_features(cleaned_data)
    print(f"特征工程后数据量: {featured_data.shape[0]} 行, {featured_data.shape[1]} 列")
    feature_count = len([col for col in featured_data.columns if col != "Date"])
    print(f"特征数量: {feature_count} 个（不含日期）")

    # 保存文件
    base_path = Path(__file__).parent
    csv_path = base_path / 'processed_data.csv'
    featured_data.to_csv(csv_path, index=False)
    print(f"已保存CSV文件: {csv_path}")

    parquet_path = base_path / 'processed_data.parquet'
    featured_data.to_parquet(parquet_path, index=False)
    print(f"已保存Parquet文件: {parquet_path}")

    print("预处理完成！")


if __name__ == '__main__':
    main()