# AI-ML-Bitcoin-Trading-Bot
OSU 467 Capstone Project - AI/ML Bitcoin Trading Bot  

Hello and welcome to the AI/ML Bitcoin Trading Bot Repo and README!

# How to create the requirements file:

### *The AI/ML Bitcoin Trading Bot Repo must be built on a Conda / Anaconda interpreter & package manager*

Activate the conda environment / interpreter you want to use.
- We recommend installation of the latest conda package manager version conda 22.9.0

Within the repository, we have established an environment_setup folder with the 
requirements file already created. 

If you wish to run requirements.txt from scratch:
- Create and cd to the directory you wish to put the requirements file.
- run this command: 
```
conda list -e > requirements.txt
```
- comment out # gym requirement.

# How to run the requirements file:

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

Finally, 

```
$ pip install gym==0.26.1
```

# Additional Packages that need pip install 

- Certain exchange API modules and packages need to be pip installed 
  - These necessary pip installs include: 
    - binance ```$ pip install binance ```
    - bitmex ```$ pip install bitmex ```
    - ccxt ```$ pip install ccxt ```