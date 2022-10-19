# Run the file, making sure to edit the csv file in , out, and h5 file out paths appropriately
# drop the resulting h5 file into the data folder, if not already completed.
# make sure that load data in trading_env.py is accessing the h5 file correctly
# Note: these files will be stored in the shared google drive

def main():
    clean_1_min_data_convert_to_h5(csv_in="bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv", csv_out="cleaned_bitcoin_1min_with_ticker.csv", h5_out="1_min_btc.h5")


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
                      index_col=['timestamp', 'ticker'])
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
    import sys
    i = 0
    with open(csv_out, 'w') as sys.stdout:
        with open(csv_in) as f:
            for row in f:
                if "NaN" not in row:
                    if i == 0:
                        print(f"ticker,timestamp,open,high,low,close,volume_btc,volume_currency,weighted_price")
                        i = 1
                    else:
                        print(f"BTC,{row}", end="")
    sys.stdout = sys.__stdout__
    convert_csv_to_h5(csv_in=csv_out, h5_out=h5_out)


if __name__ == '__main__':
    main()
