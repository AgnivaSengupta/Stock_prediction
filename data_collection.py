import yfinance as yf
import pandas as pd
import os

folder = "historical_data"
os.makedirs(folder, exist_ok=True)

tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META', 'JNJ', 'XOM', 'INTC', 'PLTR', 'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'SBIN.NS', 'LT.NS', 'HINDUNILVR.NS', 'ITC.NS', 'HCLTECH.NS', 'WIPRO.NS', 'BHARTIARTL.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'BAJFINANCE.NS', 'ADANIENT.NS', 'ADANIGREEN.NS', 'NTPC.NS', 'NESTLEIND.NS']

# Date range
start_date = '2020-01-01'
end_date = '2025-04-04'

# dictionary for storing the dataframes
dfs = {}

# Fetching data for each ticker
for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Ticker'] = ticker  # Adding ticker column
    data.reset_index(inplace=True) 
    #data = data.iloc[1:]
    #print(data.head())
    #print()
    data.to_csv(f'{folder}/{ticker}_data.csv', index=False)
    print(f"✅ Data saved to {folder} folder")

    dfs[ticker] = pd.read_csv(f'{folder}/{ticker}_data.csv')

# Removing the unnecessary first row from each dataframe
for ticker, df in dfs.items():
    if 0 in df.index:
        dfs[ticker].drop(0, inplace=True)

# Concatenating the df
combined_df = pd.concat(dfs.values(), ignore_index=True)

print(combined_df.head())

# shuffled df
shuffled_combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(shuffled_combined_df.head())

print(shuffled_combined_df.info())

shuffled_combined_df.to_csv('shuffled_combined_data.csv', index=False)

def normalize_ticker_df(df):
    norm_dfs = []
    mean_std_dict = {}

    for ticker, group in df.groupby("Ticker"):
        group[["Close", "Low", "High", "Open", "Volume"]] = group[["Close", "Low", "High", "Open", "Volume"]].apply(pd.to_numeric, errors='coerce')
        mean = group[["Close", "Low", "High", "Open", "Volume"]].mean()
        std = group[["Close", "Low", "High", "Open", "Volume"]].std()

        mean_std_dict[ticker] = {"mean": mean, "std": std}

        normalized = (group[["Close", "Low", "High", "Open", "Volume"]] - mean) / std
        normalized["Date"] = group["Date"].values
        normalized["Ticker"] = ticker
        norm_dfs.append(normalized)

    normalized_df = pd.concat(norm_dfs).sort_values(by=["Date"])
    return normalized_df, mean_std_dict

normalized_df, mean_std_dict = normalize_ticker_df(shuffled_combined_df)

normalized_df.reset_index(drop=True, inplace=True)

print(normalized_df.head(20))

normalized_df.to_csv('normalized_shuffled_combined_data.csv', index=False)


