# # import yfinance as yf

# # # Download MOVE data hourly interval
# # data = yf.download('MOVE', start='2024-09-01', interval='1h')

# # # download nasdaq index data
# # nasdaq = yf.download('^IXIC', start='2024-09-01', interval='1h')

# # # download S&P 500 index data
# # sp500 = yf.download('^GSPC', start='2024-09-01', interval='1h')

# # # VIX index data
# # vix = yf.download('^VIX', start='2024-09-01', interval='1h')

# # # gold data
# # gold = yf.download('GC=F', start='2024-09-01', interval='1h')

# # # US 3yr treasury yield data
# # us1y = yf.download('^IRX', start='2024-09-01', interval='1h')

# # # US 5yr treasury yield
# # us5y = yf.download('^FVX', start='2024-09-01', interval='1h')

# # # US 10yr treasury yield
# # us10y = yf.download('^TNX', start='2024-09-01', interval='1h')

# # # dollor index data
# # dxy = yf.download('DX-Y.NYB', start='2024-09-01', interval='1h')

# # # crude oil data
# # oil = yf.download('CL=F', start='2024-09-01', interval='1h')

# # # bitcoin data
# # btc = yf.download('BTC-USD', start='2024-09-01', interval='1h')

# # # combine all the data and download daily data for hourly interval from 2020 to till date

# import yfinance as yf
# from concurrent.futures import ThreadPoolExecutor
# from datetime import datetime, timedelta
# import pandas as pd

# # Define tickers
# tickers = {
#     'MOVE': '^MOVE',               # MOVE Index (if available)
#     # 'Nasdaq': '^IXIC',             # Nasdaq Index
#     # 'S&P 500': '^GSPC',            # S&P 500 Index
#     # 'VIX': '^VIX',                 # VIX Index
#     # 'Gold': 'GC=F',                # Gold futures
#     # 'US 3Y Yield': '^IRX',         # US 3-year Treasury yield
#     # 'US 5Y Yield': '^FVX',         # US 5-year Treasury yield
#     # 'US 10Y Yield': '^TNX',        # US 10-year Treasury yield
#     # 'Dollar Index': 'DX-Y.NYB',    # Dollar Index
#     # 'Crude Oil': 'CL=F',           # Crude oil futures
#     # 'Bitcoin': 'BTC-USD'           # Bitcoin in USD
# }

# # Define a function to generate date ranges in weekly intervals
# def generate_date_ranges(start_date, end_date):
#     ranges = []
#     while start_date < end_date:
#         next_date = min(start_date + timedelta(days=7), end_date)
#         ranges.append((start_date, next_date))
#         start_date = next_date
#     return ranges

# # Download function for individual intervals
# def download_data_chunk(ticker, start, end):
#     try:
#         print(f"Downloading data for {ticker} from {start} to {end}")
#         return yf.download(ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), interval="1h")
#     except Exception as e:
#         print(f"Error downloading data for {ticker}: {e}")
#         return pd.DataFrame()

# # Download and combine all data for a given ticker
# def download_full_data(ticker):
#     start_date = datetime(2020, 1, 1)
#     end_date = datetime.now()
#     date_ranges = generate_date_ranges(start_date, end_date)
    
#     # List to collect data chunks
#     all_data = []

#     # Download data in weekly chunks
#     for start, end in date_ranges:
#         data_chunk = download_data_chunk(ticker, start, end)
#         all_data.append(data_chunk)

#     # Concatenate all chunks into a single DataFrame
#     full_data = pd.concat(all_data)
#     full_data = full_data[~full_data.index.duplicated(keep='first')]  # Remove any duplicates

#     return full_data

# # Use concurrent downloading for each ticker
# results = {}
# with ThreadPoolExecutor() as executor:
#     futures = {executor.submit(download_full_data, ticker): name for name, ticker in tickers.items()}
#     for future in futures:
#         name = futures[future]
#         data = future.result()
#         results[name] = data
#         print(f"{name} data download and merge complete.")

# # Display or save data as needed
# for name, data in results.items():
#     print(f"\n{name} Data:")
#     print(data.head())


import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

# Define tickers
tickers = {
    'Nasdaq': '^IXIC',             # Nasdaq Index
    'S&P 500': '^GSPC',            # S&P 500 Index
    'VIX': '^VIX',                 # VIX Index
    'US 10Y Yield': '^TNX',        # US 10-year Treasury yield
}

# Function to generate date ranges in weekly intervals
def generate_date_ranges(start_date, end_date):
    ranges = []
    while start_date < end_date:
        next_date = min(start_date + timedelta(days=7), end_date)
        ranges.append((start_date, next_date))
        start_date = next_date
    return ranges

# Loop for each ticker to download data in chunks and handle exceptions
for name, ticker in tickers.items():
    print(f"\nDownloading data for {name} ({ticker})")
    start_date = datetime(2024,10,8)
    end_date = datetime.now()
    date_ranges = generate_date_ranges(start_date, end_date)
    
    # Collect data chunks in a list
    all_data = []

    for start, end in date_ranges:
        try:
            print(f"Downloading data for {ticker} from {start} to {end}")
            data_chunk = yf.download(ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), interval="60m")
            all_data.append(data_chunk)
        except Exception as e:
            print(f"Failed to download data for {ticker} from {start} to {end}: {e}")
            continue

    # Combine all downloaded chunks into a single DataFrame for the ticker
    if all_data:
        full_data = pd.concat(all_data)
        full_data = full_data[~full_data.index.duplicated(keep='first')]  # Remove any duplicates
        print(f"{name} data downloaded and combined successfully.")
    else:
        print(f"No data available for {name} ({ticker})")

    # Display or save the full_data for each ticker as needed
    print(full_data.head())
    full_data.to_csv(f"{name}_data_1h.csv")  # Save to CSV file
    


from deribit_ws import WS_Client

client_id = "R_5o2z7W"
secret = "gEBmPydgSX4xGfeT-ddUMAzaj7iz2Dhs_0BuQcj143c"
url = "wss://www.deribit.com/ws/api/v2"
client = WS_Client(client_id, secret, url)

from datetime import datetime
import pandas as pd
# download ohlcv data for BTC-PERPETUAL, DVOL and skew data
start_time = int(datetime(2024,10,8).timestamp()*1000)
end_time = int(datetime.now().timestamp()*1000)
ohlcv = client.get_tradingview_chart_data("BTC-PERPETUAL", start_time, end_time, "30")
ohlcv_df = pd.DataFrame(ohlcv)
ohlcv_df['ticks'] = pd.to_datetime(ohlcv_df['ticks'], unit='ms')
ohlcv_df = ohlcv_df.set_index('ticks')
ohlcv_df.to_csv('BTC_1d.csv')

# dvol index

dvol = client.vol_index_deribit(1642032000000-100*60*60*1000, 1642032000000, "1h")
dvol_df = pd.DataFrame(dvol['result']['data'])
dvol_df_final = pd.concat([dvol_df2, dvol_df], axis=0)
# drop duplicates
dvol_df_final = dvol_df_final.drop_duplicates(subset=0)
dvol_df_final.columns = ['timestamp','open','high','low','close']
dvol_df_final['timestamp'] = pd.to_datetime(dvol_df_final['timestamp'], unit='ms')
dvol_df_final = dvol_df_final.set_index('timestamp')
dvol_df_final.to_csv('dvol_1d.csv')


# Load csv files ending with _id.csv
import os

# Get all files in the current directory
files = os.listdir()
# Filter files that end with "_1d.csv"
files = [file for file in files if file.endswith("_1d.csv")]
# Load each file into a DataFrame
data = {}
for file in files:
    name = file.split("_")[0]
    data[name] = pd.read_csv(file, index_col=0, parse_dates=True)
    print(f"{name} data loaded successfully.")

# In BTC data, convert the 'ticks' column to a 'date' column and in dvol data, convert the 'timestamp' column to a 'date' column
data['BTC']['date'] = pd.to_datetime(data['BTC'].index).date
data['BTC'].set_index('date', inplace=True, drop=True)
data['dvol']['date'] = pd.to_datetime(data['dvol'].index).date
data['dvol'].set_index('date', inplace=True, drop=True)

# merge all data and from each key take Adj Close column from 'Nasdaq', 'S&P 500', 'VIX', 'US 10Y Yield' and close form 'BTC' and 'dvol' data and create a new dataframe
merged_data = pd.DataFrame()
for key in data.keys():
    if key in ['Nasdaq', 'S&P 500', 'VIX', 'US 10Y Yield']:
        merged_data[key] = data[key]['Adj Close']
merged_data.drop_duplicates(inplace=True)
# concat remaining BTC and dvol data 
# Ensure there are no duplicate index values in BTC and dvol data
btc_close = data['BTC']['close'].loc[~data['BTC']['close'].index.duplicated(keep='first')]
dvol_close = data['dvol']['close'].loc[~data['dvol']['close'].index.duplicated(keep='first')]

merged_data['BTC'] = btc_close
merged_data['DVOL'] = dvol_close

merged_data.to_csv('merged_data.csv')