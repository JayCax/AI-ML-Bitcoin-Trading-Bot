from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
import pandas as pd
import mplfinance as mpf
import bitmex
import json

"""Following are Binance US API keys - not compatible for paper trading in US jurisdiction"""
# api_key = "BTHSwDSUMGu8fK2PLTN7UtoBGAc2BMXgTLfAO3bwecfMSE5rhbnsHyYna2VzfT9z"
# secret = "6oza52BTBzRO9CizH5opYewZudWElkzcFUl8BZ6jT3k5YBR6KK3pGCxDIry6czF7"


# testing with bitmex paper trading
bitmex_api_key = "yc9cECO9UjxiUYgEGNP8q7Yb"
bitmex_secret_key = "GTLDBMMXaPtM1-7g7ZQFl96Yqwqd5J1N_8CrPIbyRsgEJ2lg"
base_url = "https://testnet.bitmex.com/"


def render_API_data():
    # install at this step: pip install python-binance pandas mplfinance

    client = Client(bitmex_api_key, bitmex_secret_key)

    """Quick visualization"""

    # starting at the point where training data ends
    historical_btc_usdt = client.get_historical_klines('BTCUSDT', Client.KLINE_INTERVAL_1HOUR, '1 Mar 2021')

    hist_df = pd.DataFrame(historical_btc_usdt)

    hist_df.columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume',
                       'Number of Trades', 'TB Base Volume', 'TB Quote Volume', 'Ignore']

    # comment next two lines out if you wish to retain date as timestamp rather than date
    hist_df['Open Time'] = pd.to_datetime(hist_df['Open Time'] / 1000, unit='s')
    hist_df['Close Time'] = pd.to_datetime(hist_df['Close Time'] / 1000, unit='s')

    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume', 'TB Base Volume',
                       'TB Quote Volume']

    hist_df[numeric_columns] = hist_df[numeric_columns].apply(pd.to_numeric, axis=1)

    print(hist_df.head())

    print(hist_df.tail())

    print("The shape of the data frame is: ", hist_df.shape)

    # print(hist_df.describe())

    print(hist_df.info())

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
             mav=(10, 20, 30))

    return None


class BitmexClient:
    def __init__(self, public_key: str, secret_key: str, testnet: bool):
        if testnet:
            self._base_url = "https://testnet.bitmex.com"
            self._wss_url = "wss://testnet.bitmex.com/realtime"
        else:
            self._base_url = "https://www.bitmex.com"
            self._wss_url = "wss://www.bitmex.com/realtime"

        self._public_key = public_key
        self._secret_key = secret_key

        self.client = bitmex.bitmex(test=True, api_key=bitmex_api_key, api_secret=bitmex_secret_key)

        print("Bitmex Client successfully initialized")

    def execute_sample_trade_bitmex(self):

        # publish user permissions / access
        print(self.client.Position.Position_get(filter=json.dumps({'symbol': 'XBTUSD'})).result())

        # get specific BTC data
        result = self.client.Quote.Quote_get(symbol="XBTUSD", reverse=True, count=1).result()
        print(result[0][0]['bidPrice'])

        # # get the btc data, exchange stats for the pair
        # print(client.Instrument.Instrument_get(filter=json.dumps({'symbol': 'XBTUSD'})).result())

        # symbol = 'XBTUSD'
        # ordType = 'Market'
        # orderQty_Buy = "100"  # Positive value to long
        # orderQty_Sell = "-100"  # Negative value to short
        # client.Order.Order_new(symbol=symbol, ordType=ordType, orderQty=orderQty_Buy).result()  # Long
        # client.Order.Order_new(symbol=symbol, ordType=ordType, orderQty=orderQty_Sell).result()  # Short

        params = {
            "symbol": "XBTUSD",
            "side": "BUY",
            "type": "LIMIT",
            "timeInForce": "GTC",
            "quantity": 0.002,
            "price": 18000,
        }

        # # limit buy
        # client.Order.Order_new(symbol='XBTUSD', orderQty=100, type="LIMIT", price=22050).result()

        # spot buy
        self.client.Order.Order_new(symbol='XBTUSD', orderQty=100).result()


def execute_sample_trade_binance():
    # client.binance
    pass


if __name__ == "__main__":
    # render_API_data()
    # execute_sample_trade_bitmex()
    test = BitmexClient(True, bitmex_api_key, bitmex_secret_key).execute_sample_trade_bitmex()
