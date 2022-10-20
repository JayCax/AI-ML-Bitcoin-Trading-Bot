# Run the file, making sure to edit the csv file in , out, and h5 file out paths appropriately
# drop the resulting h5 file into the data folder, if not already completed.
# make sure that load data in trading_env.py is accessing the h5 file correctly
# Note: these files will be stored in the shared google drive
import sys


def main():
    # clean_1_min_data_convert_to_h5(csv_in="bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv", csv_out="cleaned_bitcoin_1min_with_ticker.csv", h5_out="1_min_btc.h5")
    clean_x_min_data_convert_to_h5(csv_in="bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv", csv_out="cleaned_bitcoin_60min_with_ticker.csv", h5_out="60_min_btc.h5", minutes=60)


def convert_csv_to_h5(csv_in, h5_out):
    """ Helper function to convert csv file to h5"""
    import warnings
    warnings.filterwarnings('ignore')
    from pathlib import Path
    import pandas as pd

    pd.set_option('display.expand_frame_repr', False)

    DATA_STORE = Path(h5_out)

    # how it was done originally
    # df = (pd.read_csv(csv_in,
    #                   parse_dates=['date'],
    #                   index_col=['date', 'ticker'],
    #                   infer_datetime_format=True)
    #       .sort_index())

    df = (pd.read_csv(csv_in,
                      index_col=['timestamp', 'ticker', 'minute'])
          .sort_index())

    print(df.info(null_counts=True))
    with pd.HDFStore(DATA_STORE) as store:
        store.put('quandl/wiki/prices', df)

    df = pd.read_csv(f"{csv_in}")

    print(df.info(null_counts=True))
    with pd.HDFStore(DATA_STORE) as store:
        store.put('quandl/wiki/stocks', df)


def clean_1_min_data_convert_to_h5(csv_in, csv_out, h5_out):
    """Adds a ticker, previously I had used to remove all nones from the data field"""
    i = 0
    minute = -1
    with open(csv_out, 'w') as sys.stdout:
        with open(csv_in) as f:
            for row in f:
                minute += 1
                if "NaN" not in row:
                    if i == 0:
                        print(f"ticker,minute,timestamp,open,high,low,close,volume_btc,volume_currency,weighted_price")
                        i = 1
                    else:
                        print(f"BTC,{minute},{row}", end="")
    sys.stdout = sys.__stdout__
    convert_csv_to_h5(csv_in=csv_out, h5_out=h5_out)


def clean_x_min_data_convert_to_h5(csv_in, csv_out, h5_out, minutes=60):
    """Adds a ticker, previously I had used to remove all nones from the data field"""
    i = 0
    minute = -1
    reset_counter = 0
    timestamp_list, open_list, high_list, low_list, close_list, volume_btc_list, volume_currency_list, weighted_price_list = [], [], [], [], [], [], [], []
    with open(csv_out, 'w') as sys.stdout:
        with open(csv_in) as f:
            for row in f:
                minute += 1
                if i == 0:
                    print(f"ticker,minute,timestamp,open,high,low,close,volume_btc,volume_currency,weighted_price")
                    i = 1
                else:
                    if reset_counter == minutes-1:
                        try:
                            timestamp = timestamp_list[0]
                            opening = open_list[0]
                            high = max(high_list)
                            low = min(low_list)
                            close = close_list[-1]
                            volume_btc = sum(volume_btc_list)
                            volume_currency = sum(volume_currency_list)
                            weighted_price = sum(weighted_price_list)/len(weighted_price_list)
                            print(f"BTC,{minute},{timestamp},{opening},{high},{low},{close},{volume_btc},{volume_currency},{weighted_price}")
                        except Exception:
                            pass
                        # reset
                        reset_counter = 0
                        timestamp_list, open_list, high_list, low_list, close_list, volume_btc_list, volume_currency_list, weighted_price_list = [], [], [], [], [], [], [], []
                    elif "NaN" in row:
                        reset_counter += 1
                    else:
                        timestamp_list.append(row.split(',')[0])
                        open_list.append(float(row.split(',')[1]))
                        high_list.append(float(row.split(',')[2]))
                        low_list.append(float(row.split(',')[3]))
                        close_list.append(float(row.split(',')[4]))
                        volume_btc_list.append(float(row.split(',')[5]))
                        volume_currency_list.append(float(row.split(',')[6]))
                        weighted_price_list.append(float(row.split(',')[7]))
                        reset_counter += 1
    sys.stdout = sys.__stdout__
    convert_csv_to_h5(csv_in=csv_out, h5_out=h5_out)


if __name__ == '__main__':
    main()

