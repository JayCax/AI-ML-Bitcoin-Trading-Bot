# AI-ML-Bitcoin-Trading-Bot
OSU 467 Capstone Project - AI/ML Bitcoin Trading Bot  

README
Hello and welcome to the AI/ML Bitcoin Trading Bot Repo

# how to create the requirements file:

#### *The AI/ML Bitcoin Trading Bot Repo must be built on a Conda / Anaconda interpreter and package manger*

Activate the conda enviroment / interpreter you want to use.
- We recommend installation of the latest conda package manager version conda 22.9.0

Within the repository, we have established an environment_setup folder with the 
requirements file already created 

If you wish to run requirements.txt from scratch:
- Create and cd to the directory you wish to put the requirements file.
- run this command: conda list -e > requirements.txt
- comment out # gym requirement

# how to run the requirements file:

Go to conda (base) by running: 
- ```conda deactivate```

Run this command: 
- ```conda config --append channels conda-forge```

cd to directory with requirements file

Create your new environment: 
- ```conda create --name <env> --file requirements.txt```

- example: 
  - ```conda create --name CryptoBot --file requirements.txt```
      - Note, this may take a while because the bot needs many packages and libraries installed

Activate environment via:
- ```conda activate CryptoBot```

Install pip to conda, if necessary

Finally, 

```
$ pip install gym==0.26.1
```

# Additional Packages that Need pip install 

- Certain exchange API modules and packages need to be pip installed 
  - These necessary pip installs include: 
    - binance ```$ pip install binance ```
    - bitmex ```$ pip install bitmex ```
    - ccxt ```$ pip install ccxt ```