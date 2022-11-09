# binance, ccxt and bitmex need separate pip install

import json
import bitmex
import ccxt
import time

# import AIML Bitcoin Trading Bot

# testing with bitmex paper trading
bitmex_api_key = "yc9cECO9UjxiUYgEGNP8q7Yb"
bitmex_secret_key = "GTLDBMMXaPtM1-7g7ZQFl96Yqwqd5J1N_8CrPIbyRsgEJ2lg"
base_url = "https://testnet.bitmex.com/"


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
            self.bm.urls["api"] = self.bm.urls["test"]
            self.bm.urls["api"] = self.bm.urls["test"]

        self.cleaned_btc_data = None

        print("Bitmex Client successfully initialized")

    """
    Sample bitmex API methods 
    """

    def execute_sample_bitmex_api_functions(self):
        # publish specific api user permissions / access
        for k, v in self.client.Position.Position_get(filter=json.dumps({'symbol': "XBTUSDT"})).result()[0][0].items():
            print(str(k) + "   " + str(v))

        # get specific BTC data - current bid price - derived from perpetual account
        # can be substituted with "XBTUSD"
        result = self.client.Quote.Quote_get(symbol="XBTUSDT", reverse=True, count=1).result()
        print(result[0][0]["bidPrice"])

        # get the btc data, exchange stats for the pair
        for k, v in self.client.Instrument.Instrument_get(filter=json.dumps(
                {'symbol': "XBTUSDT"})).result()[0][0].items():
            print(str(k) + "      " + str(v))

        return None

    """
    Specific bot XBT / USDT trading function
    """

    def get_latest_btc_1m_data(self):

        num_of_minutes = 10

        self.cleaned_btc_data = self.client.Trade.Trade_getBucketed(binSize='1h',
                                                                    count=num_of_minutes,
                                                                    symbol='XBTUSDT',
                                                                    reverse=True).result()[0][0]

        remove_keys = ["trades", "turnover", "homeNotional", "foreignNotional", "lastSize"]

        self.cleaned_btc_data = {key: self.cleaned_btc_data[key]
                     for key in self.cleaned_btc_data if key not in remove_keys}

        return self.cleaned_btc_data

    def execute_bitmex_sample_trade(self):
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
        # print(self.bm.fetch_trades('XBTUSDT'))

        # # UNCOMMENT THIS TO EXECUTE TRADE - TO BE CONNECTED TO THE BOT
        # # spot buy with USDT 1000000 approx ~15 USDT token -arbitrary?
        # self.client.Order.Order_new(symbol='XBTUSDT', orderQty=1000000).result()

        # # cancel open orders - for limit orders
        # self.client.Order.Order_cancelAll().result()

        # publish all historical orders
        # USE ETHAN IMPLEMENTATION
        for i in self.client.Order.Order_getOrders(symbol="XBTUSDT").result()[0]:
            print(i)

        return None

    def bitmex_bot_trade(self):

        start_time = time.time()
        while True:
            cur_time = time.time()
            # 10 mins == 600 secs
            if cur_time - start_time > 600:
                pass

    def save_funds_data(self):
        # print(self.client.User.User_getWallet(currency="USDT").result())
        # print(self.client.User.User_getWallet().result()[0][0])

        # problematic since this is pulling from perp account - something like this
        # to get overall pnl
        print(self.client.User.User_getWalletSummary().result()[0][0])

        # save to csv here


if __name__ == "__main__":
    bitmex_client = BitmexClient(bitmex_api_key, bitmex_secret_key, True)
    # bitmex_client.execute_sample_bitmex_api_functions()
    print(bitmex_client.get_latest_btc_1m_data())
    # bitmex_client.execute_bitmex_sample_trade()
    bitmex_client.save_funds_data()
