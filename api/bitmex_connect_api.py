# binance, ccxt and bitmex need separate pip install

import json
import bitmex
import ccxt

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

    bitmex_client = BitmexClient(bitmex_api_key, bitmex_secret_key, True)
    bitmex_client.execute_bitmex_api_functions()
    bitmex_client.execute_bitmex_bot_trade()
