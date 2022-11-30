# binance, ccxt and bitmex need separate pip install
import datetime
import json
import random

import bitmex
import ccxt
import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import talib
import sys
import bitmex_cleaning
import csv

from Get_Trade import get_trade, load_NN, close_NN

# import AIML Bitcoin Trading Bot

# testing with bitmex paper trading
bitmex_api_key = "3I0oOTNSOemyhuHPe8y2jdP4"
bitmex_secret_key = "N20IpfBQJxgKWXnoO2AH0u7Zfu0FOsom5oj75VDfDkm1QSRk"
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

        self.cleaned_btc_data = self.client.Trade.Trade_getBucketed(binSize='1m',
                                                                    count=num_of_minutes,
                                                                    symbol='XBTUSDT',
                                                                    reverse=True).result()[0][0]

        # foreignNotional = the amount that we've spent USDT, definitely not what we want
        remove_keys = ['symbol', "trades", "turnover", "lastSize", "homeNotional", "foreignNotional"]
        # 'timestamp', 'symbol',
        self.cleaned_btc_data = {key: self.cleaned_btc_data[key]
                                 for key in self.cleaned_btc_data if key not in remove_keys}
        # replace some keys with appropriate names # Need to make sure that this function is getting the right pieces
        # of data.
        replacement = {"volume": "volume_btc", "vwap": "weighted_price"}
        for k, v in list(self.cleaned_btc_data.items()):
            self.cleaned_btc_data[replacement.get(k, k)] = self.cleaned_btc_data.pop(k)

        # volume_currency is volume_btc*close
        self.cleaned_btc_data["volume_currency"] = self.cleaned_btc_data["volume_btc"] * self.cleaned_btc_data["close"]
        # reorder dictionary to match
        temp = self.cleaned_btc_data["weighted_price"]
        del self.cleaned_btc_data["weighted_price"]
        self.cleaned_btc_data["weighted_price"] = temp

        return self.cleaned_btc_data


class LiveTrading:
    def __init__(self):
        """ Intialize """
        self.bitmex_client = BitmexClient(bitmex_api_key, bitmex_secret_key, True)
        self.data_dictionary = {}
        self.ticker = "BTC"
        self.csv_file = 'get_data_bitmex.csv'
        self.data = None
        self.current_position = 0  # -1 for short, 0 for closed, +1 for long
        self.money_made_excluding_fees = 0
        self.money_made_excluding_fees_list = []

    def load_data(self):
        """ load the data from the h5 file. """
        idx = pd.IndexSlice
        with pd.HDFStore("bitmex_api.h5") as store:
            df = (store['quandl/wiki/prices']
                  .loc[idx[:, self.ticker],
                       ['open', 'high', 'low', 'close', 'volume_btc', 'volume_currency', 'weighted_price']]
                  .dropna()
                  .sort_index())
        df.columns = ['open', 'high', 'low', 'close', 'volume_btc', 'volume_currency', 'weighted_price']
        self.data = df

    def preprocess_data(self):
        """calculate returns and percentiles, then removes missing values"""
        from sklearn.preprocessing import scale

        # this is a good place to add other things that we can change, needs to match the function in RL training
        self.data['returns'] = self.data.close.pct_change()
        self.data['ret_2'] = self.data.close.pct_change(2)
        self.data['ret_5'] = self.data.close.pct_change(5)
        self.data['ret_10'] = self.data.close.pct_change(10)
        self.data['ret_21'] = self.data.close.pct_change(21)
        self.data['rsi'] = talib.STOCHRSI(self.data.close)[1]
        self.data['macd'] = talib.MACD(self.data.close)[1]
        self.data['atr'] = talib.ATR(self.data.high, self.data.low, self.data.close)
        slowk, slowd = talib.STOCH(self.data.high, self.data.low, self.data.close)
        self.data['stoch'] = slowd - slowk
        self.data['atr'] = talib.ATR(self.data.high, self.data.low, self.data.close)
        self.data['ultosc'] = talib.ULTOSC(self.data.high, self.data.low, self.data.close)

        # edited this line?
        self.data = (self.data.replace((np.inf, -np.inf), np.nan)
                     .drop(['open', 'high', 'low', 'close', 'volume_btc', 'volume_currency', 'weighted_price'], axis=1)
                     .dropna())
        r = self.data.returns.copy()
        normalize = False
        if normalize:
            try:
                self.data = pd.DataFrame(scale(self.data),
                                         columns=self.data.columns,
                                         index=self.data.index)
            except Exception as e:
                print(e)
        features = self.data.columns.drop('returns')
        self.data['returns'] = r  # don't scale returns
        self.data = self.data.loc[:, ['returns'] + list(features)]
        print(self.data)

    def get_data(self):
        """
        Calls API to get any additional data from last time we updated data.
        Subsequently, appends the items to the data dictionary, and saves as a csv file.
        """
        # Call API to get any additional data from last time we updated data.
        # Need to make sure that this function is getting the right pieces of data.
        single_data_dictionary = self.bitmex_client.get_latest_btc_1m_data()
        # append to dict
        for key, val in single_data_dictionary.items():
            try:
                self.data_dictionary[key].append(val)
            except KeyError:
                self.data_dictionary[key] = [val]
        # save to dataframe to csv
        data = pd.DataFrame.from_dict(self.data_dictionary)
        new = pd.concat([data], axis=1)
        new.to_csv(self.csv_file, index=False)

    def make_trade(self, Trade):
        """ Makes a trade to either hold, cash out, buy short, or buy long. """
        # four options: hold, cash out, long, short
        # note orderQty 1000000 = 1 BTC:
        # note: currently set to go -.001, 0, or .001 BTC
        if Trade == -1:
            # short
            if self.current_position == -1:
                pass
            elif self.current_position == 0:
                self.money_made_excluding_fees += self.bitmex_client.client.Order.Order_new(symbol='XBTUSDT', orderQty=-1000).result()[0]['price']
            elif self.current_position == 1:
                self.money_made_excluding_fees += self.bitmex_client.client.Order.Order_new(symbol='XBTUSDT', orderQty=-2000).result()[0]['price']*2
            # set current position appropriately
            self.current_position = -1
        elif Trade == 0:
            # close Position: that is, sell or buy everything back to 0
            if self.current_position == -1:
                self.money_made_excluding_fees -= self.bitmex_client.client.Order.Order_new(symbol='XBTUSDT', orderQty=1000).result()[0]['price']
            elif self.current_position == 1:
                self.money_made_excluding_fees += self.bitmex_client.client.Order.Order_new(symbol='XBTUSDT', orderQty=-1000).result()[0]['price']
            # set current position appropriately
            self.current_position = 0
            self.money_made_excluding_fees_list.append(self.money_made_excluding_fees)
        elif Trade == 1:
            # long
            if self.current_position == -1:
                self.money_made_excluding_fees -= self.bitmex_client.client.Order.Order_new(symbol='XBTUSDT', orderQty=2000).result()[0]['price']*2
            elif self.current_position == 0:
                self.money_made_excluding_fees -= self.bitmex_client.client.Order.Order_new(symbol='XBTUSDT', orderQty=1000).result()[0]['price']
            elif self.current_position == 1:
                pass
            # set current position appropriately
            self.current_position = 1
        print("Money Made Excluding Fees: ", self.money_made_excluding_fees)

    def run_live_trading(self):
        """
        Runs the live trading Bot.
        Note: This is proof of concept, ideally any trading would be done with the binance platform,
        and would be traded on a similar scale to that of the trained model
        NOTE: must run about 34 cycles before it starts trading
        """
        # set up
        num_seconds_between_query = 1  # will be 60 eventually to fit with the current model
        num_minutes_avg = 2  # would also be 60 to fit with our current model
        total_counter = 0
        counter = 0
        current_time = time.time() - num_seconds_between_query
        print("Starting...")
        ddqn, state_dim, trading_environment = load_NN()  # calls load_NN from Get_Trade.py
        while True:
            # note this number is the number of seconds between every query
            while time.time() - current_time >= num_seconds_between_query:
                current_time = time.time()
                self.get_data()
                if counter > num_minutes_avg:
                    # Pre-process to add the additional columns to the new data (ex rsi, macd)
                    bitmex_cleaning.clean_x_min_data_convert_to_h5(csv_in=self.csv_file, csv_out='bitmex_api.csv',
                                                                   h5_out='bitmex_api.h5', minutes=num_minutes_avg)
                    self.load_data()
                    self.preprocess_data()
                    # need to wait long enough for data to start getting put into dataframe
                    if not self.data.empty:  # ensure there is data as it takes a while before data is populated
                        # decide what the trade will be: buy, long, short
                        # test_data = np.array([0.03452198, -0.50615868, -2.82824341, -2.90283056, -3.45309186, 0.52710094, -11.19257768, 9.95488124, -0.2326905, -1.07325023])
                        current_state = self.data.tail(
                            1).to_numpy()  # gets most recent data and converts to numpy array
                        trade = get_trade(ddqn, state_dim, current_state)
                        print("trade is ", trade)
                        # make the trade
                        self.make_trade(trade)
                    # reset counter
                    counter = 0
                counter += 1
                total_counter += 1
            if total_counter >= 600:  # quit the function after # loops this means we'll run 600 times, which is currently about 10 minutes
                break

        close_NN(trading_environment)
        # close Position: that is, sell or buy everything back to 0
        if self.current_position == -1:
            self.money_made_excluding_fees -= self.bitmex_client.client.Order.Order_new(symbol='XBTUSDT', orderQty=1000).result()[0]['price']
        elif self.current_position == 1:
            self.money_made_excluding_fees += self.bitmex_client.client.Order.Order_new(symbol='XBTUSDT', orderQty=-1000).result()[0]['price']
        # set current position appropriately
        self.current_position = 0

        self.money_made_excluding_fees_list.append(self.money_made_excluding_fees)
        print("Money Made Excluding Fees: ", self.money_made_excluding_fees)
        # record details in a csv file
        self.obtain_record_portfolio()  # this function still doesn't seem to work properly

        # make a nice plot
        plt.plot(self.money_made_excluding_fees_list)
        # naming the x axis
        plt.xlabel('Closed Positions')
        # naming the y axis
        plt.ylabel('Gain/Loss (dollars)')
        plt.show()

    def obtain_record_portfolio(self):
        """
        Saves the PNL information to a csv file.
        Due to confusing bitmex API, this does not always work as hoped.
        """
        pnl_header = ["TransactionDateTime", "PNL", "TotalUSDAccountValue"]
        pnl_data = "pnl_data.csv"

        pnl_list = []
        date_list = []

        for i in self.bitmex_client.client.User.User_getWalletHistory().result()[0]:
            pnl_list.append(i["amount"])
            date_list.append(i['transactTime'])
        pnl_list = [round(i * 0.1, 2) for i in pnl_list]

        # save PNL to csv file here
        with open(pnl_data, "w+", encoding="UTF8", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(pnl_header)
            wal_hist = self.bitmex_client.client.User.User_getWalletHistory().result()[0]
            for i in range(len(wal_hist)):
                if i != len(wal_hist) - 1:
                    pnl_val = wal_hist[i]["amount"]
                    pnl_val *= 0.1
                    pnl_val = round(pnl_val, 2)
                    writer.writerow([str(date_list[i]), str(pnl_val), str(sum(pnl_list[i:]))])
                else:
                    pnl_val = wal_hist[i]["amount"]
                    pnl_val *= 0.1
                    pnl_val = round(pnl_val, 2)
                    writer.writerow([str(date_list[i]), str(pnl_val), str(pnl_val)])

        return None


if __name__ == "__main__":
    live_trading = LiveTrading()
    live_trading.run_live_trading()
