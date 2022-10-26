from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
import pandas as pd
import mplfinance as mpf
import bitmex
import json

# testing with bitmex paper trading
api_key = "yc9cECO9UjxiUYgEGNP8q7Yb"
secret = "GTLDBMMXaPtM1-7g7ZQFl96Yqwqd5J1N_8CrPIbyRsgEJ2lg"
base_url = "https://testnet.bitmex.com/"

def render_API_data():
    """Following are Binance US API keys - not compatible for paper trading in US jurisdiction"""
    # api_key = "BTHSwDSUMGu8fK2PLTN7UtoBGAc2BMXgTLfAO3bwecfMSE5rhbnsHyYna2VzfT9z"
    # secret = "6oza52BTBzRO9CizH5opYewZudWElkzcFUl8BZ6jT3k5YBR6KK3pGCxDIry6czF7"


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


def execute_sample_trade_bitmex():
    client = bitmex.bitmex(test=True, api_key=api_key, api_secret=secret)

    result = client.Quote.Quote_get(symbol="XBTUSD", reverse=True, count=1).result()
    print(result[0][0]['bidPrice'])

    print(client.Instrument.Instrument_get(filter=json.dumps({'symbol': 'XBTJPY'})).result())

    symbol = 'XBTUSDT'
    ordType = 'Market'
    orderQty_Buy = "100"  # Positive value to long
    orderQty_Sell = "-100"  # Negative value to short
    client.Order.Order_new(symbol=symbol, ordType=ordType, orderQty=orderQty_Buy).result()  # Long
    client.Order.Order_new(symbol=symbol, ordType=ordType, orderQty=orderQty_Sell).result()  # Short

    #client.Order.Order_new(symbol, orderQty=10, price=12345.0).result()


def execute_sample_trade_binance():
    #client.binance
    pass

if __name__ == "__main__":
    #render_API_data()
    execute_sample_trade_bitmex()