
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
LOCAL_VENDOR_ROOT = BASE_DIR / "volpredictor" / "lib"
if LOCAL_VENDOR_ROOT.exists():
    for site in LOCAL_VENDOR_ROOT.glob("python*/site-packages"):
        if site.exists() and str(site) not in sys.path:
            sys.path.append(str(site))

from tvDatafeed import TvDatafeed, Interval  # noqa: E402

try:
    import telegram  # noqa: E402
except ImportError:  # Optional dependency - used only for notifications
    telegram = None

""" 355725 """ # user id of actant test account
tele_auth_token = "5493651278:AAFlpxKsQrAHnD0b2WhPhPE3uSVb8KZ6tlw"
bot = telegram.Bot(token=tele_auth_token) if telegram else None

LOG_PATH = BASE_DIR / 'VolPredictor.log'
logging.basicConfig(filename=str(LOG_PATH), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

warnings.filterwarnings("ignore")

user_name = "shreyasc@gmail.com"
pwd = "tradingview333!!!"
tv = TvDatafeed(user_name, pwd)

def VolModel(data, notify=True):
    latest_date = data.index[-1]
       
    data['Date'] = data.index
    data.reset_index(drop=True, inplace=True)
    # Convert 'Date' to datetime and set as index for easier handling of time-based lag
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Create the target variable by shifting 'DVOL' by 1 week (7 days * 24 hours for hourly data)
    # Get last 7 days date, and it should be  working day
    # last_date = data.index[-1]
    # seven_day_before = last_date - pd.Timedelta(days=7)
    # from data dataframw choose seven_day_before date or nearest date lower then seven_day_before
    # data = data.loc[data.index <= seven_day_before]
    # Shift after seven_day_before date
    # data['DVOL_1w_ahead'] = data['DVOL'].shift(-7*7)
    data['DVOL_1w_ahead'] = data['DVOL'].shift(-5*7)

    # Drop rows with NaN values created by the shift
    data = data.dropna()

    # Add time-based features
    data['hour'] = data.index.hour
    data['dayofweek'] = data.index.dayofweek

    # Adding rolling statistics as new features
    data['S&P_500_rolling_mean'] = data['S&P 500'].rolling(window=35).mean()  # 1-day rolling mean
    data['Nasdaq_rolling_mean'] = data['Nasdaq'].rolling(window=35).mean()
    data['VIX_rolling_mean'] = data['VIX'].rolling(window=35).mean()
    data['BTCUSD_rolling_mean'] = data['BTCUSD'].rolling(window=35).mean()

    # Drop rows with NaN values due to rolling mean calculations
    data = data.dropna()

    #Tickers - SP.SPX, NASDAQ.NDQ, TVC.US10Y, TVC.VIX, #Shreyas's Trading View, Binance.BTCUSD, Deribit.DVOL

    # Define feature and target columns
    feature_columns = [
        'S&P 500', 'Nasdaq', 'US 10Y', 'VIX', 'Global Net Liquidity', 'BTCUSD',
        'hour', 'dayofweek', 'S&P_500_rolling_mean', 'Nasdaq_rolling_mean',
        'VIX_rolling_mean', 'BTCUSD_rolling_mean'
    ]
    target_column = 'DVOL_1w_ahead'

    # Separate the data for training and prediction
    X = data[feature_columns]
    y = data[target_column]

    # Use all but the latest date for training, and the latest date for prediction
    X_train, y_train = X.iloc[:-1], y.iloc[:-1]
    X_predict = X.iloc[[-1]]  # Latest date data for prediction

    # Pipeline with scaling and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Scaling the data
        ('regressor', GradientBoostingRegressor(random_state=42))  # Initializing with Gradient Boosting
    ])

    # Hyperparameter grid for tuning GradientBoostingRegressor
    param_grid = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [3, 5, 7],
        'regressor__learning_rate': [0.01, 0.1, 0.2]
    }

    # Perform grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Predict the DVOL for the latest date
    y_pred_latest = best_model.predict(X_predict)
    print(len(data))
    print(len(X))
    print(len(y_pred_latest))
    data['Predicted DVOL'] = best_model.predict(X)
    # Output the predicted value for the latest date
    print(f"Predicted DVOL for the latest date {latest_date} utc: {y_pred_latest[0]}")
    msg = f" Predicted DVOL for the latest date {latest_date} utc: {round(y_pred_latest[0], 2)}"
    # table = tabulate(data.iloc[-5:], headers='keys', tablefmt='psql')
    # table = pd.DataFrame({'Date': latest_date, 'Predicted DVOL': [round(y_pred_latest[0], 2)]})
    # send_msg_to_ki(table)
    # send_msg_to_ki(msg)
    # bot.send_message(chat_id='-1001654565487', text=table)
    if notify:
        send_msg_on_telegram(msg)
    logging.info(f"{datetime.utcnow()} : Predicted DVOL for the latest date {latest_date}: {round(y_pred_latest[0], 2)}")
    return data 

def send_msg_on_telegram(msg,account="prod"):
    tele_auth_token = "5532825755:AAGbuZcnPk_nh0gAlT1ioXbBlN-eqp1Kh8c"
    if "TEST" in account:
        tel_group_id   = "test_vol_group"
    else:
        tel_group_id   = "vol_yield_prod"    
    params = {
        'parse_mode': 'MarkdownV2'
    }
    telegram_api_url = f"https://api.telegram.org/bot{tele_auth_token}/sendMessage?chat_id=@{tel_group_id}&text={msg}"
    tel_resp = requests.get(telegram_api_url)
    if tel_resp.status_code == 200:
        print ("Notification has been sent on Telegram") 
    else:
        print ("Could not send Message")  

def fetch_data(symbol, exchange, interval, bars, retries=10, delay=10):
    import time
    """Fetch data with retries to ensure a valid DataFrame."""
    for attempt in range(retries):
        df = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=bars)
        if df is not None and not df.empty:
            return df  # Successfully fetched non-empty data
        print(f"Attempt {attempt + 1} failed for {symbol}. Retrying in {delay} seconds...")
        time.sleep(delay)
    raise ValueError(f"Failed to fetch data for {symbol} after {retries} attempts.")

# Function to get total liquidity
def get_total_liquidity(bars=5000,interval = Interval.in_monthly):
    # Fetch data as DataFrames
    df_fed = fetch_data('FRED:WALCL', 'FRED', interval, bars)
    df_japan_assets = fetch_data('FRED:JPNASSETS', 'FRED', interval, bars)
    df_jpy_usd = fetch_data('FX_IDC:JPYUSD', 'IDC', interval, bars)
    df_ecb_assets = fetch_data('ECBASSETSW', 'FRED', interval, bars)
    df_eur_usd = fetch_data('FX:EURUSD', 'FX', interval, bars)
    df_rrp = fetch_data('RRPONTSYD', 'FRED', interval, bars)
    df_tga = fetch_data('WTREGEN', 'FRED', interval, bars)

    # Set all index date format 
    for df in [df_fed, df_japan_assets, df_jpy_usd, df_ecb_assets, df_eur_usd, df_rrp, df_tga]:
        try:
            df.index = df.index.strftime('%Y-%m-%d')
        except Exception as e:
            print(f"Error occurred while setting index format for {df.symbol}: {e}")
            continue
    # for each df, check if data is available till latest date and if not then ffiling the data using previous data
    df_dict = {'df_fed': df_fed, 'df_japan_assets': df_japan_assets, 'df_jpy_usd': df_jpy_usd, 'df_ecb_assets': df_ecb_assets, 'df_eur_usd': df_eur_usd, 'df_rrp': df_rrp, 'df_tga': df_tga}
    latest_date = max([data.index[-1] for data in df_dict.values()])
    for df, data in df_dict.items():
        if data.index[-1] != latest_date:
            data = data.reindex(data.index.append(pd.Index([latest_date])))
            # Fill the data using ffill method till latest date with 1 day frequency
            data.ffill(inplace=True)
            df_dict[df] = data
            print(f"Data for {df} was missing for the latest date. Filling with previous data.")
        else:
            df_dict[df] = data
    df_fed = df_dict['df_fed']
    df_japan_assets = df_dict['df_japan_assets']
    df_jpy_usd = df_dict['df_jpy_usd']
    df_ecb_assets = df_dict['df_ecb_assets']
    df_eur_usd = df_dict['df_eur_usd']
    df_rrp = df_dict['df_rrp']
    df_tga = df_dict['df_tga']  

    # Merge all data on matching timestamps
    merged_df = pd.concat([df_fed['close'], 
                        df_japan_assets['close'], df_jpy_usd['close'], 
                        df_ecb_assets['close'], df_eur_usd['close'], 
                        df_rrp['close'], df_tga['close']], 
                        axis=1)
    merged_df.dropna(inplace=True)

    # Rename columns for clarity
    merged_df.columns = ['fed', 'japan_assets', 'jpy_usd', 
                     'ecb_assets', 'eur_usd', 'rrp', 'tga']

    # Calculate each adjusted line with currency conversion
    merged_df['line_japan'] = merged_df['japan_assets'] * merged_df['jpy_usd']
    merged_df['line_ecb'] = merged_df['ecb_assets'] * merged_df['eur_usd']

    # Calculate total as per formula
    merged_df['total'] = (merged_df['fed'] + merged_df['line_japan'] + 
                        merged_df['line_ecb'] - merged_df['rrp'] - merged_df['tga'])

    # # Display the calculated total
    # print(merged_df['total'])
    return merged_df['total']
# example : nifty_index_data = tv.get_hist(symbol='NIFTY',exchange='NSE',interval=Interval.in_1_hour,n_bars=1000)

# Download data for each ticker
def get_live_data(tickers,interval = Interval.in_30_minute):
    data_live = {}
    for name, ticker in tickers.items():
        print(name)
        print(ticker)
        print(f"Downloading data for {name} ({ticker})...")
        ticker_ = ticker.split('.')[1]
        exchange = ticker.split('.')[0]
        # if ticker_ == 'GL':
        #     data[name] = get_total_liquidity()
        # else:
        data_live[name] = tv.get_hist(symbol=ticker_,exchange = exchange, interval=Interval.in_30_minute, n_bars=5000)
        while not isinstance(data_live[name], pd.DataFrame) or data_live[name].empty:
            print(f"Retrying download for {name} ({ticker})...")
            data_live[name] = tv.get_hist(symbol=ticker_,exchange = exchange, interval=Interval.in_30_minute, n_bars=5000)
            time.sleep(10)
        # data_live[name].index = data_live[name].index.tz_localize('UTC')
    # If data index is not multiple of 30 minutes, resample to 30 minutes
    # for name, df in data_live.items():
    #     if df.index.freq != '30T':
    #         data_live[name] = df.resample('30T').ffill()
    # For VIX add 20 minutes to the time index to match the 30-minute data
    # if interval == Interval.in_30_minute:
    #     data_live['VIX'].index = data_live['VIX'].index + pd.Timedelta(minutes=20)
    #     data_live['US 10Y'].index = data_live['US 10Y'].index + pd.Timedelta(minutes=30)
    #     data_live['DVOL'].index = data_live['DVOL'].index + pd.Timedelta(minutes=30)
    #     data_live['BTCUSD'].index = data_live['BTCUSD'].index + pd.Timedelta(minutes=30)
        
    return data_live


def filter_data(data = None):
    # Extract the close values for each key and rename the column to match the key
    # try:
    # close_dfs = {
    #     key: df[['close']].rename(columns={'close': key})
    #     for key, df in data.items()
    # }
    close_dfs = {}
    for key,df in data.items():
        close_dfs[key] = df[['close']].rename(columns={'close': key})
    # except Exception as e:
    #     print(f"Error in filter_data: {e}")
    #     return None
    # Use the S&P 500 DataFrame as the base for concatenation
    sp_index = close_dfs['S&P 500'].index
    # subtract 10 minutes from VIX index to match the 30-minute data
    # close_dfs['VIX'].index = close_dfs['VIX'].index - pd.Timedelta(minutes=10)
    combined_df = close_dfs['S&P 500']

    # Merge each other DataFrame on the S&P 500 index and ffill missing rows
    for key, df in close_dfs.items():
        if key != 'S&P 500':
            combined_df = pd.merge(
                combined_df,
                df,
                left_index=True,
                right_index=True,
                how='outer'
            )

    # Forward-fill only till the S&P 500 index
    combined_df = combined_df.loc[sp_index].ffill()

    # convert combined_df index to %Y-%m-%d %H:%M:%S and also from IST time to UTC time
    # Localize to IST
    localized_index = combined_df.index.tz_localize('Asia/Kolkata')

    # Convert to UTC
    utc_index = localized_index.tz_convert('UTC')
    # only use %Y-%m-%d %H:%M:%S and 30 minutes time frame
    combined_df.index = utc_index
    combined_df = combined_df[combined_df.index.minute == 30]
    combined_df.index = combined_df.index.strftime('%Y-%m-%d %H:%M:%S')
    # Display the resulting DataFrame

    return combined_df

def send_msg_to_ki(msg):
    tele_auth_token = "5493651278:AAFlpxKsQrAHnD0b2WhPhPE3uSVb8KZ6tlw"
    tel_group_id   = "trade_alerter"
    telegram_api_url = f"https://api.telegram.org/bot{tele_auth_token}/sendMessage?chat_id=@{tel_group_id}&text={msg}"
    tel_resp = requests.get(telegram_api_url)
    if tel_resp.status_code == 200:
        print ("Notification has been sent on Telegram") 
    else:
        print ("Could not send Message")    
        
def run_cycle(send_notifications=True, save_outputs=True):
    """Run a single Vol Predictor update cycle and return the prediction dataframe."""
    data_file = BASE_DIR / "Vol Prediction Data_new.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Seed data not found at {data_file}")
    
    # Load old data
    data_old = pd.read_csv(data_file, index_col=0)

    # Define tickers
    tickers = {
        'S&P 500': 'SP.SPX',
        'Nasdaq': 'NASDAQ.NDX',
        'US 10Y': 'TVC.US10Y',
        'VIX': 'TVC.VIX',
        'BTCUSD': 'Coinbase.BTCUSD',
        'DVOL': 'Deribit.DVOL'
        }

    data_live = get_live_data(tickers)
    last_date = data_old.index[-1]
    new_data = filter_data(data_live)

    # after old data, merge new data with old data
    data = pd.concat([data_old, new_data], axis=0)
    print(data.iloc[750:800])
    # drop duplicate index rows
    # if any row from old data in a given column is empty and that value is available in new data, then keep the new data value
    data_combined = data_old.combine_first(new_data)
    data = data_combined.copy()
    # drop duplicate rows based on index
    # data = data[~data.index.duplicated(keep='last')]
    # conver the index to datetime
    data.index = pd.to_datetime(data.index)
    data['YearMonth'] = data.index.to_period('M')
    print(data.iloc[750:800])

    # ignore duplicate rows based on duplicate S&P 500 data and keep the last row
    #####################################################
    global_liquidity_data = get_total_liquidity()
    # create df of global_liquidity_data
    global_liquidity_data = pd.DataFrame(global_liquidity_data)
    global_liquidity_data.index = pd.to_datetime(global_liquidity_data.index)
    global_liquidity_data['YearMonth'] = global_liquidity_data.index.to_period('M')

    # for missing global liquidity data, fill with global liquidity data, wrt year month column
    data_merged = pd.merge(data, global_liquidity_data, on='YearMonth', how='left')
    data_merged.index = data.index
    # for all missing global liquidity data, fill with total column data
    data_merged['Global Net Liquidity'] = data_merged['total'].combine_first(data_merged['Global Net Liquidity']).ffill()
    data_merged.drop(['YearMonth', 'total'], axis=1, inplace=True)

    # drop to csv file
    if save_outputs:
        data_merged.to_csv(data_file)
    print(data_merged.tail(10))
    prediction_data = VolModel(data_merged, notify=send_notifications)
    print(prediction_data)

    # save as csv
    if save_outputs:
        output_csv = BASE_DIR / "Vol Prediction Data_to_plot.csv"
        prediction_data.to_csv(output_csv)
        # Create plot between actual and predicted data and save the plot as image
        fig, ax = plt.subplots()
        ax.plot(prediction_data.index, prediction_data['DVOL'], label='Actual DVOL')
        ax.plot(prediction_data.index, prediction_data['Predicted DVOL'], label='Predicted DVOL')
        ax.set_xlabel('Date')
        ax.set_ylabel('DVOL')
        ax.set_title('Actual vs Predicted DVOL')
        ax.legend()
        plt.savefig(BASE_DIR / 'Actual_vs_Predicted_DVOL.png')
        plt.close(fig)
    return prediction_data

def main():
    return run_cycle()

def run_scheduled():
    main()
    import schedule
    schedule.every(1).hours.at(":00").do(main)
    while True:
        try:
            schedule.run_pending()
            time.sleep(5)
        except Exception as e:
            print(e)
            time.sleep(5)

if __name__ == "__main__":
    run_scheduled()
