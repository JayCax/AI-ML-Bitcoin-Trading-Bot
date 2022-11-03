# binance, ccxt and bitmex need separate pip install

from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
import bitmex
import ccxt
import mplfinance as mpf
import pandas as pd
import json

"""
Following are Binance US API keys - not compatible for paper trading in US jurisdiction
    * CAN BE USED FOR OBTAINING TEST DATA / HISTORICAL VISUALIZATIONS / ETC 
"""
binance_api_key = "BTHSwDSUMGu8fK2PLTN7UtoBGAc2BMXgTLfAO3bwecfMSE5rhbnsHyYna2VzfT9z"
binance_secret_key = "6oza52BTBzRO9CizH5opYewZudWElkzcFUl8BZ6jT3k5YBR6KK3pGCxDIry6czF7"

# testing with bitmex paper trading
bitmex_api_key = "yc9cECO9UjxiUYgEGNP8q7Yb"
bitmex_secret_key = "GTLDBMMXaPtM1-7g7ZQFl96Yqwqd5J1N_8CrPIbyRsgEJ2lg"
base_url = "https://testnet.bitmex.com/"


class BinanceClient:
    def __init__(self, public_key: str, secret_key: str):
        """
        compatible with bitmex api keys as well
        """
        self._public_key = public_key
        self._secret_key = secret_key

        self.client = Client(public_key, secret_key)

    """
    Quick Binance visualizations
    """
    def render_API_data(self):

        # starting at the point where training data ends
        historical_btc_usdt = self.client.get_historical_klines('BTCUSDT', Client.KLINE_INTERVAL_1HOUR, '1 Mar 2021')

        hist_df = pd.DataFrame(historical_btc_usdt)

        hist_df.columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume',
                           'Number of Trades', 'TB Base Volume', 'TB Quote Volume', 'Ignore']

        # comment next two lines out if you wish to retain date as timestamp rather than date
        hist_df['Open Time'] = pd.to_datetime(hist_df['Open Time'] / 1000, unit='s')
        hist_df['Close Time'] = pd.to_datetime(hist_df['Close Time'] / 1000, unit='s')

        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume', 'TB Base Volume',
                           'TB Quote Volume']

        hist_df[numeric_columns] = hist_df[numeric_columns].apply(pd.to_numeric, axis=1)

        """
        publish head and tail of data frame 
        
        open time, open, high, low, close, volume, quote asset volume, quote asset volume, number of trades, TB base vol, TB Quote Vol
        """

        # set of initial 1 hour historical price data
        print(hist_df.head())

        # set of latest 1 hour historical price data
        print(hist_df.tail())

        """
        publish statistics of data frame 
            
        open time, open, high, low, close, volume, quote asset volume, quote asset volume, number of trades, TB base vol, TB Quote Vol
        count
        mean
        std
        min
        25%
        50%
        75%
        max
        """

        # statistical data over the data frame
        print(hist_df.describe())

        # publish shape of historical data - how many rows and columns
        print("The shape of the data frame is: ", hist_df.shape)

        # publish information about historical data, number of entries, etc.
        print(hist_df.info())

        """
        visualization
        """

        # determine where to close visualization - take how many rows  of data for visualization
        # IN HOURS
        hist_df.set_index('Close Time').tail(14400)

        # time frame since Mar 01, 2021 - when kaggle training data terminates approxamitely 14420 hours ago
        # IN HOURS
        mpf.plot(hist_df.set_index('Close Time').tail(14420),
                 type='candle', style='charles',
                 volume=True,
                 title='BTCUSDT data since March 01, 2021 in hour increments',
                 mav=(10, 20, 30))

        return None

    def execute_sample_trade_binance(self):
        # client.binance - NOT POSSIBLE WITH BINANCE US
        pass


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

        self.client = bitmex.bitmex(test=True, api_key=public_key, api_secret=secret_key)

        self.bm = ccxt.bitmex({'apiKey': public_key,
                               'secret': secret_key})

        if 'test' in self.bm.urls:
            self.bm.urls['api'] = self.bm.urls['test']
            self.bm.urls['api'] = self.bm.urls['test']

        print("Bitmex Client successfully initialized")

    """
    Sample bitmex API methods 
    """
    def execute_bitmex_api_functions(self):
        # publish specific api user permissions / access
        for k, v in self.client.Position.Position_get(filter=json.dumps({'symbol': 'XBTUSD'})).result()[0][0].items():
            print(str(k) + "   " + str(v))

        # get specific BTC data - current bid price
        result = self.client.Quote.Quote_get(symbol="XBTUSD", reverse=True, count=1).result()
        print(result[0][0]['bidPrice'])

        # get the btc data, exchange stats for the pair
        for k, v in self.client.Instrument.Instrument_get(filter=json.dumps({'symbol': 'XBTUSDT'})).result()[0][
            0].items():
            print(str(k) + "      " + str(v))

        return None

    """
    Specific bot XBT / USDT trading function
    """

    def execute_bitmex_bot_trade(self):
        """
        Deprecated trade functions

        # limit buy
        self.client.Order.Order_new(symbol='XBTUSD', orderQty=100, price=18000).result()

        # spot buy
        self.client.Order.Order_new(symbol='XBTUSD', orderQty=100).result()

        # limit sell
        self.client.Order.Order_new(symbol='XBTUSD', orderQty=-100, price=25000).result()

        # spot sell
        self.client.Order.Order_new(symbol='XBTUSD', orderQty=-100).result()
        """

        # # may use cctx to execute trades
        # print(self.bm.fetch_trades('XBTUSD'))

        # # UNCOMMENT THIS TO EXECUTE TRADE - TO BE CONNECTED TO THE BOT
        # # spot buy with USDT 1000000 approx ~15 USDT token -arbitrary?
        # self.client.Order.Order_new(symbol='XBTUSDT', orderQty=1000000).result()

        # # cancel open orders - for limit orders
        # self.client.Order.Order_cancelAll().result()

        # publish all historical orders
        for i in self.client.Order.Order_getOrders(symbol="XBTUSD").result()[0]:
            print(i)

        return None


if __name__ == "__main__":
    binance_client = BinanceClient(binance_api_key, binance_secret_key)
    binance_client.render_API_data()

    bitmex_client = BitmexClient(bitmex_api_key, bitmex_secret_key, True)
    bitmex_client.execute_bitmex_api_functions()
    bitmex_client.execute_bitmex_bot_trade()
