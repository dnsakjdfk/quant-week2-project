import ssl
# 解决nltk下载SSL报错（此处不再依赖新闻情感，保留兼容）
ssl._create_default_https_context = ssl._create_unverified_context
import pandas as pd
import numpy as np
from pathlib import Path


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

    # ========== VIX衍生情绪分 0~1 ==========
    vix_min = df['VIX_Close'].min()
    vix_max = df['VIX_Close'].max()
    df['vix_sentiment'] = 1 - (df['VIX_Close'] - vix_min) / (vix_max - vix_min)

    # ========== 综合市场情绪（仅VIX维度，无无效0.5新闻列） ==========
    df["comprehensive_market_sentiment"] = df["vix_sentiment"]

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