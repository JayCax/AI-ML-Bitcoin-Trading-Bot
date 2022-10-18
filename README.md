# AI-ML-Bitcoin-Trading-Bot
OSU 467 Capstone Project - AI/ML Bitcoin Trading Bot  

README
Hello and welcome to the AI/ML Bitcoin Trading Bot Repo

To get started, Tensorflow must be PIP installed or built on Anaconda / Conda

# how to create the requirements file:

Activate the conda enviroment you want to use.
- We recommend installation of the latest conda package manager version conda 22.9.0

cd to the directory you want to put the requirements file.
- Within the repository, we have established an environment_setup folder with the 
requirements file already created 

If you wish to run it from scratch: 
-run this command: conda list -e > requirements.txt
-comment out # gym requirement

# how to run the requirements file:

go to conda (base) by running: conda deactivate

run this command: conda config --append channels conda-forge

cd to directory with requirements file

create your new environment: conda create --name <env> --file requirements.txt
- example: conda create --name CryptoBot --file requirements.txt
    - Note, this may take a while because the bot needs many packages and libraries installed

Activate environment, install pip to conda if necessary

Finally, pip install gym==0.26.1
