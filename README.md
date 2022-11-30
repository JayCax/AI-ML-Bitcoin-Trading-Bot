# AI-ML-Bitcoin-Trading-Bot
OSU 467 Capstone Project - AI/ML Bitcoin Trading Bot  

Hello and welcome to the AI/ML Bitcoin Trading Bot Repo and README!

NOTE: GO to instructions.txt for more instructions.

# Required Environment Specifications:

### *The AI/ML Bitcoin Trading Bot Repo must be built on a Conda / Anaconda package & environment management system with Python 3.9 as the base interpreter.*

Activate the conda environment / interpreter you want to use.
- We recommend installation of the latest Conda package & environment manager version conda 22.9.0 that comes with Python 3.9 as its base interpreter.

Within the repository, we have established an environment_setup folder with the 
requirements.txt file already created. Change to this directory.

# How to run the requirements.txt file:

#### Proceed to the Anaconda / Conda terminal of your preferred IDE or the Anaconda / Conda powershell prompt.

Go to conda (base) by running: 
```
conda deactivate
```

Run this command: 
```
conda config --append channels conda-forge
```

cd to directory with requirements.txt file.

Create your new environment: 
```
conda create --name <env> --file requirements.txt
```

example: 
```
conda create --name CryptoBot --file requirements.txt
```
- Note, this may take a while because the AI/ML bot needs many packages and libraries installed.

Activate environment via:
```
conda activate CryptoBot
```

Conda install pip, if necessary.

Finally, There are some additional packages that need a pip install:

# Additional Packages that need pip install 

- Certain exchange API modules and packages need to be pip installed 
  - These necessary pip installs include: 
    - binance ```$ pip install gym==0.26.1 ```
    - bitmex ```$ pip install bitmex ```
    - ccxt ```$ pip install ccxt ```
