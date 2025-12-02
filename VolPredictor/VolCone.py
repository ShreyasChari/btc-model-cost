import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from HelperFunctions import sql_db
import ccxt
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import requests
import time
import yfinance as yf
import pytz
exchange = ccxt.binance({
    'enableRateLimit': True,  # Enable rate limit for the exchange
})

def send_msg_on_telegram(msg):
    tele_auth_token = "5532825755:AAGbuZcnPk_nh0gAlT1ioXbBlN-eqp1Kh8c"
    tel_group_id   = "vol_yield_prod"
    telegram_api_url = f"https://api.telegram.org/bot{tele_auth_token}/sendMessage?chat_id=@{tel_group_id}&text={msg}"
    tel_resp = requests.get(telegram_api_url)
    if tel_resp.status_code == 200:
        print ("Notification has been sent on Telegram") 
    else:
        print ("Could not send Message")  

# get btc-perpetual data historic from 2020-01-01 to till date
def get_data(since, till,interval = '1h'):
    final_df = pd.DataFrame()
    while since < till:
        ohlcv = exchange.fetch_ohlcv(symbol = 'BTC/USDT', timeframe = interval, since = since, limit = 1000) 
        df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        try:
            since = df['datetime'].iloc[-1]+1
            final_df = pd.concat([final_df, df], axis = 0)
        except:
            since = till
            return final_df
        print(since)
    if final_df.empty:
        return final_df
    else:
        final_df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')

    return final_df


def getMSTRdata(ticker,start=datetime(2024,8,12,0,0,0),period='1h'):
    try:
        data = yf.download(ticker, start=start, interval=period)
        eastern = pytz.timezone('US/Eastern')
        # Current index time is ET time, convert that to UTC
        data.index = data.index.tz_localize(eastern).tz_convert(pytz.utc)
        data.index = data.index.strftime('%Y-%m-%d %H:%M:%S')
        data.rename(columns = {'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'}, inplace = True)
        data = data[['open','high','low','close']]
    except Exception as e:
        print(e)
        msg = "Failed download:- MSTR: No data found for this date range, symbol may be delisted"
        # send_msg_on_telegram(msg)
        data = pd.DataFrame()
    return data

def get_constant_maturity_data():
    initial_date = sql_db().get_query_response("Select min(timestamp) from amberdata.deltasurfaceconstantmaturities_hour")[0][0]
    data = sql_db().get_query_response(f"Select * from amberdata.deltasurfaceconstantmaturities_hour where timestamp >= {initial_date}")
    df = pd.DataFrame(data)
    column_names = sql_db().get_query_response("Select column_name from information_schema.columns where table_name = 'deltasurfaceconstantmaturities_hour'")
    column_names = [i[0] for i in column_names]
    # drop dupplicate column_names_list 
    column_names = list(dict.fromkeys(column_names))
    print(column_names)
    # rename columns to match the actual column names
    df.columns = column_names
    df = df[['timestamp','daysToExpiration','atm']]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    pivot_df = df.pivot_table(index='timestamp', columns='daysToExpiration', values='atm', aggfunc='first')
    pivot_df.columns = [f'{int(col)}-day_atmIV' for col in pivot_df.columns]
    # create datetime from start to end with hourly intervals   
    start_date = pivot_df.index[0]
    end_date = pivot_df.index[-1]
    date_range = pd.date_range(start=start_date, end=end_date, freq='h')
    pivot_df = pivot_df.reindex(date_range)
    pivot_df = pivot_df.interpolate(method='linear')
    pivot_df.to_pickle(r"D:\Data\KIVC\To Animesh\to Mahindra\From animesh\atm_iv.pkl")
    return pivot_df

def get_mstr_optionchain():
    from convexlib.api import ConvexApi
    # use "pro" or "live"
    convex_instance = ConvexApi("devtest@devtest.devtest","d3vt3st","live")
    # requesting option chain as rows
    rows = convex_instance.get_chain_as_rows("MSTR",params=["volm_bs","volatility","delta","price"],exps=[0,1,2,3,4,5,6,7,8],rng=1.0)
    # convert all rows to a dataframe
    df = pd.DataFrame(rows)
    df.columns = ['symbol','expiration','strike','opt_type',"volm_bs","volatility","delta","price"]
    # extract expiration from symbil name, 6 ELEMENTS AFTER MSTR in symbol name
    df['expiration'] = df['symbol'].str.extract(r'MSTR(.{6})')
    # convert expiration to datetime
    df['expiration'] = pd.to_datetime(df['expiration'], format='%y%m%d')
    # daysToExpiration = expiration - current date
    df['daysToExpiration'] = (df['expiration'] - datetime.now()).dt.days + 1
    df['daysToExpiration'] = [1 if i ==0 else i for i in df['daysToExpiration']]
    df.to_csv("MSTR_options.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    pivot_df = df.pivot_table(index='timestamp', columns='daysToExpiration', values='atm', aggfunc='first')
    pivot_df.columns = [f'{int(col)}-day_atmIV' for col in pivot_df.columns]
    # extract call, put option_type, strike, expiration
    atm_iv = df[df['delta']==0.5][['expiration','volatility']]
    # check rows where delta is near to 0.5 
    atm_iv = df[df['delta'].between(0.45,0.55)]
    atm_iv['diff'] = abs(atm_iv['delta']-0.5)
    atm_iv = atm_iv.sort_values(by='diff')
    atm_iv = atm_iv.groupby('expiration').first()
    atm_iv_value = atm_iv['volatility'].iloc[0]
    # get atm_iv for 1-day,7-day,14-day,21-day,30-day,60-day,90-day
    atm_iv = atm_iv.groupby('expiration').first()
    atm_iv = atm_iv['volatility']
    atm_iv = atm_iv.to_frame()
    atm_iv.columns = ['atm_iv']
    

def update_data():
    try:
        df = pd.read_pickle("D:\\Data\\KIVC\\To Animesh\\to Mahindra\\From animesh\\pickle_mini.pkl")
        # read deltasurfaceconstantmaturities data from 2020-01-01 to till date
        last_date = df['datetime'].iloc[-1]
        last_date_int = int(last_date.timestamp()*1000)
        df_last = get_data(last_date_int, int(datetime.now().timestamp())*1000)
        df_last['datetime'] = pd.to_datetime(df_last['datetime'], unit='ms')
        df = pd.concat([df, df_last], axis = 0)
        df.drop_duplicates(subset='datetime', keep='last', inplace=True)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.sort_values(by='datetime', inplace=True)
        df.to_pickle(r"D:\Data\KIVC\To Animesh\to Mahindra\From animesh\pickle_mini.pkl")
        return df
    except Exception as e:
        print(e)
        msg = "Error in updating VolCone spot Data"
        send_msg_on_telegram(msg)
        df = get_data(1577836800000, datetime.now().timestamp()*1000)
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        df.to_pickle(r"D:\Data\KIVC\To Animesh\to Mahindra\From animesh\pickle_mini.pkl")
        send_msg_on_telegram("VolCone spot Data has been updated")
    return df


def update_data_mstr():
    df = pd.read_pickle("D:/Data/KIVC/To Animesh/to Mahindra/From animesh/pickle_mstr.pkl")
    # df['datetime'] = pd.to_datetime(df['datetime'])
    df.index = pd.to_datetime(df.index)
    last_date = df.index[-1]
    df.rename_axis('datetime', inplace=True)
    # df.set_index('datetime', inplace=True)
    # drop last row from df
    # read deltasurfaceconstantmaturities data from 2020-01-01 to till date
    
    try:
        df_last = getMSTRdata("MSTR",last_date,period = "30m")
        # df_last['datetime'] = pd.to_datetime(df_last['datetime'], unit='ms')
        df = pd.concat([df, df_last], axis = 0)
        df.drop_duplicates(keep='last', inplace=True)
        df.dropna(inplace=True)
        # df.reset_index(inplace=True)
        # df.sort_values(by='datetime', inplace=True)
        mstr_data = df.copy()
        mstr_data.to_pickle(r"D:\Data\KIVC\To Animesh\to Mahindra\From animesh\pickle_mstr.pkl")
    except:
        print("Market closed") 
        df.drop_duplicates(subset='datetime', keep='last', inplace=True)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.sort_values(by='datetime', inplace=True)
        mstr_data = df.copy()
        mstr_data.to_pickle(r"D:\Data\KIVC\To Animesh\to Mahindra\From animesh\pickle_mstr.pkl")
    return df
    

# update_data()
# import schedule
# import time

# schedule.every(1).minute.do(update_data)
# while True:
#     schedule.run_pending()
#     time.sleep(1)

############################################################################################################
############################################################################################################

def vol_calc(no_days,coin = "BTC"):
    if coin == "BTC":
        df = update_data()
    elif coin == "MSTR":
        df = update_data_mstr()
    df.reset_index(inplace=True)
    # df = pd.read_pickle("D:\\Data\\KIVC\\To Animesh\\to Mahindra\\From animesh\\pickle_mini.pkl")
    df= df[['datetime','close']]
    df.rename(columns = {'close':'spot'}, inplace = True)
    df = df.drop_duplicates(subset='datetime')[['datetime','spot']]
    df = df.set_index('datetime')
    df['hourly_vol'] = df['spot'].pct_change().fillna(0)
    hourly_volatility = df['hourly_vol']

    # Calculate rolling volatilities for different periods
    if coin == "BTC":
        rolling_1day_vol = hourly_volatility.rolling(window=24).std() * np.sqrt(24*365)
        rolling_7day_vol = hourly_volatility.rolling(window=24*7).std() * np.sqrt(24*365)
        rolling_14day_vol = hourly_volatility.rolling(window=24*14).std() * np.sqrt(24*365)
        rolling_21day_vol = hourly_volatility.rolling(window=24*21).std() * np.sqrt(24*365)
        rolling_30day_vol = hourly_volatility.rolling(window=24*30).std() * np.sqrt(24*365)
        rolling_60day_vol = hourly_volatility.rolling(window=24*60).std() * np.sqrt(24*365)
        rolling_90day_vol = hourly_volatility.rolling(window=24*90).std() * np.sqrt(24*365)
    elif coin == "MSTR":
        rolling_1day_vol = hourly_volatility.rolling(window=24).std() * np.sqrt(24*252)
        rolling_7day_vol = hourly_volatility.rolling(window=24*7).std() * np.sqrt(24*252)
        rolling_14day_vol = hourly_volatility.rolling(window=24*14).std() * np.sqrt(24*252)
        rolling_21day_vol = hourly_volatility.rolling(window=24*21).std() * np.sqrt(24*252)
        rolling_28day_vol = hourly_volatility.rolling(window=24*28).std() * np.sqrt(24*252)
        rolling_70day_vol = hourly_volatility.rolling(window=24*70).std() * np.sqrt(24*252)
        rolling_98day_vol = hourly_volatility.rolling(window=24*98).std() * np.sqrt(24*252)
        

    # Create a DataFrame to hold the annualized volatilities
    if coin == "BTC":
        vol_df = pd.DataFrame({
            '1-day_current': rolling_1day_vol,
            '7-day_current': rolling_7day_vol,
            '14-day_current': rolling_14day_vol,
            '21-day_current': rolling_21day_vol,
            '30-day_current': rolling_30day_vol,
            '60-day_current': rolling_60day_vol,
            '90-day_current': rolling_90day_vol,
        }).dropna()
    elif coin == "MSTR":
        vol_df = pd.DataFrame({
            '1-day_current': rolling_1day_vol,
            '7-day_current': rolling_7day_vol,
            '14-day_current': rolling_14day_vol,
            '21-day_current': rolling_21day_vol,
            '28-day_current': rolling_28day_vol,
            '70-day_current': rolling_70day_vol,
            '98-day_current': rolling_98day_vol,
        }).dropna()

    if coin == "BTC":
        vol_stats = vol_df.rolling(window=no_days*24).agg(['mean', 'std', 'min', 'max'])
        vol_stats.columns = ['1-day_mean', '1-day_std', '1-day_min', '1-day_max',
                            '7-day_mean', '7-day_std', '7-day_min', '7-day_max',
                            '14-day_mean', '14-day_std', '14-day_min', '14-day_max',
                            '21-day_mean', '21-day_std', '21-day_min', '21-day_max',
                            '30-day_mean', '30-day_std', '30-day_min', '30-day_max',
                            '60-day_mean', '60-day_std', '60-day_min', '60-day_max',
                            '90-day_mean', '90-day_std', '90-day_min', '90-day_max']
        for period in ['1-day', '7-day', '14-day', '21-day', '30-day', '60-day', '90-day']:
            vol_stats[f'{period}_+1std'] = vol_stats[f'{period}_mean'] + vol_stats[f'{period}_std']
            vol_stats[f'{period}_-1std'] = vol_stats[f'{period}_mean'] - vol_stats[f'{period}_std']
            vol_stats[f'{period}_+2std'] = vol_stats[f'{period}_mean'] + 2 * vol_stats[f'{period}_std']
            vol_stats[f'{period}_-2std'] = vol_stats[f'{period}_mean'] - 2 * vol_stats[f'{period}_std']

        df = vol_stats.copy()
        df = pd.concat([df, vol_df], axis=1)
        df.dropna(inplace=True)
        df.to_pickle(f"D:\\Data\\KIVC\\To Animesh\\to Mahindra\\From animesh\\{coin}VolCone_{no_days}.pkl")
    elif coin == "MSTR":
        vol_stats = vol_df.rolling(window=no_days*2).agg(['mean', 'std', 'min', 'max'])
        vol_stats.columns = ['1-day_mean', '1-day_std', '1-day_min', '1-day_max',
                            '7-day_mean', '7-day_std', '7-day_min', '7-day_max',
                            '14-day_mean', '14-day_std', '14-day_min', '14-day_max',
                            '21-day_mean', '21-day_std', '21-day_min', '21-day_max',
                            '28-day_mean', '28-day_std', '28-day_min', '28-day_max',
                            '70-day_mean', '70-day_std', '70-day_min', '70-day_max',
                            '98-day_mean', '98-day_std', '98-day_min', '98-day_max']
        for period in ['1-day', '7-day', '14-day', '21-day', '28-day', '70-day', '98-day']:
            vol_stats[f'{period}_+1std'] = vol_stats[f'{period}_mean'] + vol_stats[f'{period}_std']
            vol_stats[f'{period}_-1std'] = vol_stats[f'{period}_mean'] - vol_stats[f'{period}_std']
            vol_stats[f'{period}_+2std'] = vol_stats[f'{period}_mean'] + 2 * vol_stats[f'{period}_std']
            vol_stats[f'{period}_-2std'] = vol_stats[f'{period}_mean'] - 2 * vol_stats[f'{period}_std']
        df = vol_stats.copy()
        df = pd.concat([df, vol_df], axis=1)
        df.dropna(inplace=True)
        df.to_pickle(f"D:\\Data\\KIVC\\To Animesh\\to Mahindra\\From animesh\\{coin}VolCone_{no_days}.pkl")    
    return df

read_pickle = pd.read_pickle("D:\\Data\\KIVC\\To Animesh\\to Mahindra\\From animesh\\BTCVolCone_182.pkl")

try:
    vol_calc(182)
    vol_calc(365)
    vol_calc(1095)
    get_constant_maturity_data()
    # get_mstr_optionchain()
    # vol_calc(182,coin = "MSTR
    # vol_calc(182,coin = "MSTR")
    # vol_calc(365,coin = "MSTR")
    # vol_calc(1095,coin = "MSTR")
    # mstr_iv = get_mstr_optionchain()
except Exception as e:
    print(e)
    msg = "Error in updating VolCone Data for MSTR"
    send_msg_on_telegram(msg)
    
import schedule
# schedule every 30 seconds
schedule.every(1).hour.do(vol_calc,182,coin = "BTC")
schedule.every(1).hour.do(vol_calc,365,coin = "BTC")
schedule.every(1).hour.do(vol_calc,1095,coin = "BTC")
# schedule.every(1).hour.do(vol_calc,182,coin = "MSTR")
# schedule.every(1).hour.do(vol_calc,365,coin = "MSTR")
# schedule.every(1).hour.do(vol_calc,1095,coin = "MSTR")
# schedule.every(1).hour.do(update_data_mstr)
schedule.every(1).hour.do(get_constant_maturity_data)
# schedule.every(1).hour.do(get_mstr_optionchain)
while True:
    try:
        schedule.run_pending()
        time.sleep(1)
    except Exception as e:
        print(e)
        msg = "Error in updating VolCone Data"
        send_msg_on_telegram(msg)
        time.sleep(60)
        continue
# # df = pd.read_csv("D:/Data/KIVC/To Animesh/VolCone.csv", parse_dates=['datetime']).set_index(['datetime'])
# def plotVolCone():
#     df = vol_calc()
#     days = [7, 14, 20, 30, 45, 60, 90]

#     # Initial plot data
#     initial_index = 0
#     min_columns = df.filter(regex='_min$', axis=1).iloc[initial_index]
#     max_columns = df.filter(regex='_max$', axis=1).iloc[initial_index]
#     mean_columns = df.filter(regex='_mean$', axis=1).iloc[initial_index]
#     sdp1_columns = df.filter(regex='_\+1std$', axis=1).iloc[initial_index]
#     sdp2_columns = df.filter(regex='_\+2std$', axis=1).iloc[initial_index]
#     sdn1_columns = df.filter(regex='_-1std$', axis=1).iloc[initial_index]
#     sdn2_columns = df.filter(regex='_-2std$', axis=1).iloc[initial_index]

#     fig, ax = plt.subplots(figsize=(14, 7))
#     plt.subplots_adjust(bottom=0.25)

#     # Initial plot
#     line_min, = ax.plot(days, min_columns, label='Min', color='red', marker='o')
#     line_max, = ax.plot(days, max_columns, label='Max', color='darkgreen', marker='o')
#     line_mean, = ax.plot(days, mean_columns, label='Mean', color='black', marker='o')
#     line_sdp1, = ax.plot(days, sdp1_columns, label='+1SD', color='royalblue', marker='o')
#     line_sdp2, = ax.plot(days, sdp2_columns, label='+2SD', color='blueviolet', marker='o')
#     line_sdn1, = ax.plot(days, sdn1_columns, label='-1SD', color='yellow', marker='o')
#     line_sdn2, = ax.plot(days, sdn2_columns, label='-2SD', color='orange', marker='o')

#     ax.set_xlabel('Expiry Days')
#     ax.set_ylabel('Volatility')
#     ax.set_xticks(days)
#     ax.legend()
#     ax.grid(True)

#     # Add a slider
#     ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
#     slider = Slider(ax_slider, 'Date Index', 0, len(df) - 1, valinit=initial_index, valstep=1)

#     # Update function
#     def update(val):
#         idx = int(slider.val)
#         min_columns = df.filter(regex='_min$', axis=1).iloc[idx]
#         max_columns = df.filter(regex='_max$', axis=1).iloc[idx]
#         mean_columns = df.filter(regex='_mean$', axis=1).iloc[idx]
#         sdp1_columns = df.filter(regex='_\+1std$', axis=1).iloc[idx]
#         sdp2_columns = df.filter(regex='_\+2std$', axis=1).iloc[idx]
#         sdn1_columns = df.filter(regex='_-1std$', axis=1).iloc[idx]
#         sdn2_columns = df.filter(regex='_-2std$', axis=1).iloc[idx]
        
#         line_min.set_ydata(min_columns)
#         line_max.set_ydata(max_columns)
#         line_mean.set_ydata(mean_columns)
#         line_sdp1.set_ydata(sdp1_columns)
#         line_sdp2.set_ydata(sdp2_columns)
#         line_sdn1.set_ydata(sdn1_columns)
#         line_sdn2.set_ydata(sdn2_columns)
        
#         ax.set_title(f"UTC : {df.index[idx].strftime('%Y-%m-%d %H:%M:%S')}")
#         fig.canvas.draw_idle()

#     # Connect the update function to the slider
#     slider.on_changed(update)

#     plt.show()
############################################################################################################
############################################################################################################

############################################################################################################
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider

# # Load your data
# df = pd.read_csv("D:/Data/KIVC/To Animesh/VolCone.csv", parse_dates=['datetime']).set_index(['datetime'])
# days = [7, 14, 20, 30, 45, 60, 90]

# # Extract columns
# min_columns = df.filter(regex='_min$', axis=1).iloc[0]
# max_columns = df.filter(regex='_max$', axis=1).iloc[0]
# mean_columns = df.filter(regex='_mean$', axis=1).iloc[0]
# sdp1_columns = df.filter(regex='_\+1std$', axis=1).iloc[0]
# sdp2_columns = df.filter(regex='_\+2std$', axis=1).iloc[0]
# sdn1_columns = df.filter(regex='_-1std$', axis=1).iloc[0]
# sdn2_columns = df.filter(regex='_-2std$', axis=1).iloc[0]

# fig, ax = plt.subplots(figsize=(14, 7))
# plt.subplots_adjust(bottom=0.25)

# # Initial plot
# line_min, = ax.plot(days, min_columns, label='Min', color='red', marker='o')
# line_max, = ax.plot(days, max_columns, label='Max', color='darkgreen', marker='o')
# line_mean, = ax.plot(days, mean_columns, label='Mean', color='black', marker='o')
# line_sdp1, = ax.plot(days, sdp1_columns, label='+1SD', color='royalblue', marker='o')
# line_sdp2, = ax.plot(days, sdp2_columns, label='+2SD', color='blueviolet', marker='o')
# line_sdn1, = ax.plot(days, sdn1_columns, label='-1SD', color='yellow', marker='o')
# line_sdn2, = ax.plot(days, sdn2_columns, label='-2SD', color='orange', marker='o')

# ax.set_xlabel('Expiry Days')
# ax.set_ylabel('Volatility')
# ax.set_xticks(days)
# ax.legend()
# ax.grid(True)
# plt.show()
############################################################################################################
# # Define the slider
# ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
# slider = Slider(ax_slider, 'Day', min(days), max(days), valinit=min(days), valstep=1)

# # # Update function for the slider
# def update(val):
#     current_day = int(slider.val)
#     filtered_days = [day for day in days if day <= current_day]
#     filtered_min = min_columns[:len(filtered_days)]
#     filtered_max = max_columns[:len(filtered_days)]
#     filtered_mean = mean_columns[:len(filtered_days)]
#     filtered_sdp1 = sdp1_columns[:len(filtered_days)]
#     filtered_sdp2 = sdp2_columns[:len(filtered_days)]
#     filtered_sdn1 = sdn1_columns[:len(filtered_days)]
#     filtered_sdn2 = sdn2_columns[:len(filtered_days)]

#     ax.clear()
#     ax.plot(filtered_days, filtered_min, label='Min', color='red', marker='o')
#     ax.plot(filtered_days, filtered_max, label='Max', color='darkgreen', marker='o')
#     ax.plot(filtered_days, filtered_mean, label='Mean', color='black', marker='o')
#     ax.plot(filtered_days, filtered_sdp1, label='+1SD', color='royalblue', marker='o')
#     ax.plot(filtered_days, filtered_sdp2, label='+2SD', color='blueviolet', marker='o')
#     ax.plot(filtered_days, filtered_sdn1, label='-1SD', color='yellow', marker='o')
#     ax.plot(filtered_days, filtered_sdn2, label='-2SD', color='orange', marker='o')
#     ax.axvline(x=current_day, color='gray', linestyle='--')
#     ax.set_xlabel('Expiry Days')
#     ax.set_ylabel('Volatility')
#     ax.set_xticks(days)
#     ax.legend()
#     ax.grid(True)
#     fig.canvas.draw_idle()

# slider.on_changed(update)

# plt.show()


# for vol_period in ['14-day', '20-day', '30-day']:
#     plt.plot(vol_stats.index, vol_stats[f'{vol_period}_mean'], label=f'{vol_period} Mean')
#     plt.fill_between(vol_stats.index, vol_stats[f'{vol_period}-1std'], vol_stats[f'{vol_period}+1std'], alpha=0.2, label=f'{vol_period} ± 1 Std Dev')
#     plt.fill_between(vol_stats.index, vol_stats[f'{vol_period}-2std'], vol_stats[f'{vol_period}+2std'], alpha=0.1, label=f'{vol_period} ± 2 Std Dev')
#     plt.plot(vol_stats.index, vol_stats[f'{vol_period}_14pct'], linestyle='--', alpha=0.6, label=f'{vol_period} 2.5th Percentile')
#     plt.plot(vol_stats.index, vol_stats[f'{vol_period}_90pct'], linestyle='--', alpha=0.6, label=f'{vol_period} 97.5th Percentile')
#
# plt.title('Volatility Cone for BTC')
# plt.xlabel('Date')
# plt.ylabel('Annualized Volatility')
# plt.legend()
# plt.grid(True)
# plt.show()
#
# # Calculate percentiles for the volatility cone
# percentiles = vol_df.quantile([0.1, 0.5, 0.9])
#
# # Plotting the volatility cone
# plt.figure(figsize=(14, 7))
# plt.plot(vol_df.index, vol_df['1-day'], label='1-day Volatility')
# plt.plot(vol_df.index, vol_df['1-week'], label='1-week Volatility')
# plt.plot(vol_df.index, vol_df['1-month'], label='1-month Volatility')
#
# # Plot percentile bands
# plt.fill_between(vol_df.index, percentiles.loc[0.1, '1-day'], percentiles.loc[0.9, '1-day'], color='blue', alpha=0.1)
# plt.fill_between(vol_df.index, percentiles.loc[0.1, '1-week'], percentiles.loc[0.9, '1-week'], color='orange', alpha=0.1)
# plt.fill_between(vol_df.index, percentiles.loc[0.1, '1-month'], percentiles.loc[0.9, '1-month'], color='green', alpha=0.1)
#
# plt.title('Volatility Cone for BTC')
# plt.xlabel('Date')
# plt.ylabel('Annualized Volatility')
# plt.legend()
# plt.show()




# Load your data