import pandas as pd
import numpy as np
from pathlib import Path


def load_and_clean_data():
    # 1. 加载数据
    base_path = Path(__file__).parent
    treasury = pd.read_csv(base_path / "raw_data" / "macro" / "raw_treasury_rate_2018_2024.csv")
    vix = pd.read_csv(base_path / "raw_data" / "macro" / "raw_vix_2018_2024.csv")
    jpm = pd.read_csv(base_path / "raw_data" / "stock" / "raw_jpm_stock_2018_2024.csv")

    # 2. 统一日期格式，去除时区信息，转为datetime
    treasury['Date'] = pd.to_datetime(treasury['Date'])
    # VIX和JPM的日期带时区，提取前10位日期字符串再转换
    vix['Date'] = pd.to_datetime(vix['Date'].str[:10])
    jpm['Date'] = pd.to_datetime(jpm['Date'].str[:10])

    # 3. 缺失值插值（线性插值）
    # 国债利率数据有缺失值，按日期线性插值
    treasury = treasury.set_index('Date').sort_index()
    treasury = treasury.interpolate(method='linear', axis=0).reset_index()

    # 4. 四分位距法(IQR)剔除异常值
    def remove_outliers_iqr(df, columns):
        df_clean = df.copy()
        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # 异常值替换为NaN，然后再插值
            df_clean.loc[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound), col] = np.nan
        # 再次插值填补异常值造成的空缺
        df_clean = df_clean.set_index('Date').sort_index()
        df_clean = df_clean.interpolate(method='linear', axis=0).reset_index()
        return df_clean

    # 对数值列进行异常值处理
    treasury_cols = ['Rate_1Y', 'Rate_5Y', 'Rate_10Y']
    vix_cols = ['VIX_Close']
    jpm_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends']

    treasury = remove_outliers_iqr(treasury, treasury_cols)
    vix = remove_outliers_iqr(vix, vix_cols)
    jpm = remove_outliers_iqr(jpm, jpm_cols)

    # 5. 时间轴对齐：按日期内连接，取共同交易日
    merged = pd.merge(treasury, vix, on='Date', how='inner')
    merged = pd.merge(merged, jpm, on='Date', how='inner')
    merged = merged.sort_values('Date').reset_index(drop=True)

    return merged


def build_features(df):
    df = df.copy()

    # ========== 基础特征 ==========
    # 1. JPM日收益率
    df['JPM_Daily_Return'] = df['Close'].pct_change()

    # 2. JPM 20日滚动波动率（年化）
    df['JPM_Volatility_20D'] = df['JPM_Daily_Return'].rolling(window=20).std() * np.sqrt(252)

    # 3. 股息增长率
    df['Dividend_Growth'] = df['Dividends'].pct_change()
    # 股息为0时增长率设为0
    df['Dividend_Growth'] = df['Dividend_Growth'].fillna(0)
    df.loc[df['Dividends'] == 0, 'Dividend_Growth'] = 0

    # 4. JPM 5日移动平均线
    df['JPM_MA5'] = df['Close'].rolling(window=5).mean()

    # 5. JPM 20日移动平均线
    df['JPM_MA20'] = df['Close'].rolling(window=20).mean()

    # 6. 成交量变化率
    df['Volume_Change'] = df['Volume'].pct_change().fillna(0)

    # ========== 利率相关特征 ==========
    # 7. 10年期国债利率动量（5日变化率）
    df['Rate_10Y_Momentum_5D'] = df['Rate_10Y'].pct_change(periods=5)

    # 8. 期限利差：10Y - 1Y
    df['Spread_10Y_1Y'] = df['Rate_10Y'] - df['Rate_1Y']

    # 9. 期限利差：10Y - 5Y
    df['Spread_10Y_5Y'] = df['Rate_10Y'] - df['Rate_5Y']

    # 10. 期限利差：5Y - 1Y
    df['Spread_5Y_1Y'] = df['Rate_5Y'] - df['Rate_1Y']

    # ========== VIX与情绪特征 ==========
    # 11. VIX日变化率
    df['VIX_Daily_Change'] = df['VIX_Close'].pct_change()

    # 12. VIX 20日滚动均值
    df['VIX_MA20'] = df['VIX_Close'].rolling(window=20).mean()

    # 13. 市场情绪评分（0-1区间，VIX越低情绪越高，归一化）
    vix_min = df['VIX_Close'].min()
    vix_max = df['VIX_Close'].max()
    df['Market_Sentiment'] = 1 - (df['VIX_Close'] - vix_min) / (vix_max - vix_min)

    # ========== 高阶相关性特征 ==========
    # 14. VIX与JPM收盘价20日滚动相关系数
    df['VIX_JPM_Correlation_20D'] = df['VIX_Close'].rolling(window=20).corr(df['Close'])

    # 去掉前20行的空值（滚动窗口造成的）
    df = df.dropna().reset_index(drop=True)

    return df


def main():
    print("开始数据清洗...")
    cleaned_data = load_and_clean_data()
    print(f"清洗后数据量: {cleaned_data.shape[0]} 行, {cleaned_data.shape[1]} 列")

    print("开始特征构建...")
    featured_data = build_features(cleaned_data)
    print(f"特征工程后数据量: {featured_data.shape[0]} 行, {featured_data.shape[1]} 列")
    print(f"特征数量: {featured_data.shape[1] - 1} 个（不含日期）")

    # 保存结果
    base_path = Path(__file__).parent
    # 保存CSV格式
    csv_path = base_path / 'processed_data.csv'
    featured_data.to_csv(csv_path, index=False)
    print(f"已保存CSV文件: {csv_path}")

    # 保存Parquet格式
    parquet_path = base_path / 'processed_data.parquet'
    featured_data.to_parquet(parquet_path, index=False)
    print(f"已保存Parquet文件: {parquet_path}")

    print("预处理完成！")


if __name__ == '__main__':
    main()