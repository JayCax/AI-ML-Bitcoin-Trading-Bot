from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
import pandas as pd
import mplfinance as mpf


api_key = "BTHSwDSUMGu8fK2PLTN7UtoBGAc2BMXgTLfAO3bwecfMSE5rhbnsHyYna2VzfT9z"
secret = "6oza52BTBzRO9CizH5opYewZudWElkzcFUl8BZ6jT3k5YBR6KK3pGCxDIry6czF7"

# install at this step: pip install python-binance pandas mplfinance

client = Client(api_key, secret)

"""Quick visualization"""

# depth = client.get_order_book(symbol='BTCUSDT')

# depth_df = pd.DataFrame(depth['asks'])
# depth_df.columns = ['Price', 'Volume']
# depth_df.head()


# starting at the point where training data ends
historical_btc_usdt = client.get_historical_klines('BTCUSDT', Client.KLINE_INTERVAL_1HOUR, '1 Mar 2021')

hist_df = pd.DataFrame(historical_btc_usdt)

hist_df.columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume',
                    'Number of Trades', 'TB Base Volume', 'TB Quote Volume', 'Ignore']

# hist_df.head()
#
# hist_df.tail()

hist_df.shape

# comment next two lines out if you wish to retain date as timestamp rather than date
hist_df['Open Time'] = pd.to_datetime(hist_df['Open Time']/1000, unit='s')
hist_df['Close Time'] = pd.to_datetime(hist_df['Close Time']/1000, unit='s')

numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume', 'TB Base Volume', 'TB Quote Volume']

hist_df[numeric_columns] = hist_df[numeric_columns].apply(pd.to_numeric, axis=1)

hist_df.describe()

hist_df.info()

# visualization

# determine where to close visualization - take how many rows  of data for visualization
# IN HOURS
hist_df.set_index('Close Time').tail(14400)

# time frame since Mar 01, 2021 - when kaggle training data terminates
# IN HOURS
mpf.plot(hist_df.set_index('Close Time').tail(14420),
        type='candle', style='charles',
        volume=True,
        title='BTCUSDT data since March 01, 2021',
        mav=(10,20,30))