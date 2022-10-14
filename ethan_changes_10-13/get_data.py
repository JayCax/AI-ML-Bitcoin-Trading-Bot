import warnings

warnings.filterwarnings('ignore')
from pathlib import Path
import requests
from io import BytesIO
from zipfile import ZipFile, BadZipFile

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from sklearn.datasets import fetch_openml

pd.set_option('display.expand_frame_repr', False)

DATA_STORE = Path('bitcoin.h5')

# df = (pd.read_csv(r"cleaned_bitcoin_1min.csv",
#                   parse_dates=['date'],
#                   index_col=['date', 'ticker'],
#                   infer_datetime_format=True)
#       .sort_index())


df = (pd.read_csv(r"cleaned_bitcoin_1min_with_ticker.csv",
                  index_col=['timestamp', 'ticker'])
      .sort_index())

# API key: KkHkttmBYaQyzDJ_SwQd

print(df.info(null_counts=True))
with pd.HDFStore(DATA_STORE) as store:
    store.put('quandl/wiki/prices', df)

df = pd.read_csv('cleaned_bitcoin_1min_with_ticker.csv')
# no longer needed
# df = pd.concat([df.loc[:, 'code'].str.strip(),
#                 df.loc[:, 'name'].str.split('(', expand=True)[0].str.strip().to_frame('name')], axis=1)

print(df.info(null_counts=True))
with pd.HDFStore(DATA_STORE) as store:
    store.put('quandl/wiki/stocks', df)
