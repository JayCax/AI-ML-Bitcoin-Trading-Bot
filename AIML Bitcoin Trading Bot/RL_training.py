# Class CS 467
# Team Members: David Elrick, John-Francis Caccamo, Ethan Blake
# This code has been adapted from the book Machine Learning For Algorithmic Trading 2nd Edition
# See Github for base code from book.
# https://github.com/PacktPublishing/Machine-Learning-for-Algorithmic-Trading-Second-Edition/tree/master/22_deep_reinforcement_learning
#

#### IMPORTS  ####

import warnings

warnings.filterwarnings('ignore')
import matplotlib

from pathlib import Path
from time import time
from collections import deque
from random import sample

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import load_model

import gym
from gym.envs.registration import register
from datetime import datetime
import joblib
import csv

RESULT_SUMMARY = []

#### CLASSES ####

## DEFINE TRADING AGENT
class DDQNAgent:
    def __init__(self, state_dim,
                 num_actions,
                 learning_rate,
                 gamma,
                 epsilon,
                 epsilon_start,
                 epsilon_end,
                 epsilon_decay_steps,
                 epsilon_exponential_decay,
                 replay_capacity,
                 architecture,
                 l2_reg,
                 tau,
                 batch_size,
                 loaded):

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.replay_capacity = replay_capacity
        self.experience = deque([], maxlen=replay_capacity)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.architecture = architecture
        self.l2_reg = l2_reg

        self.online_network = self.build_model()
        self.target_network = self.build_model(trainable=False)
        self.update_target()

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon = epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.epsilon_exponential_decay = epsilon_exponential_decay
        self.epsilon_history = []  # appears to be unused

        self.total_steps = 0
        self.train_steps = self.train_episodes = 0   # these appear to be unused
        self.episodes = self.episode_length  = 0
        self.steps_per_episode = []
        self.episode_reward = 0
        self.rewards_history = []

        self.batch_size = batch_size
        self.tau = tau
        self.losses = []
        self.idx = tf.range(batch_size)
        self.train = True

    def build_model(self, trainable=True):
        layers = []
        n = len(self.architecture)
        for i, units in enumerate(self.architecture, 1):
            layers.append(Dense(units=units,
                                input_dim=self.state_dim if i == 1 else None,
                                activation='relu',
                                kernel_regularizer=l2(self.l2_reg),
                                name=f'Dense_{i}',
                                trainable=trainable))
        layers.append(Dropout(.1))
        layers.append(Dense(units=self.num_actions,
                            trainable=trainable,
                            name='Output'))
        model = Sequential(layers)
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target(self):
        self.target_network.set_weights(self.online_network.get_weights())

    def epsilon_greedy_policy(self, state):
        self.total_steps += 1
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)
        q = self.online_network.predict(state)
        return np.argmax(q, axis=1).squeeze()

    def memorize_transition(self, s, a, r, s_prime, not_done):
        if not_done:
            self.episode_reward += r
            self.episode_length += 1
        else:
            if self.train:
                if self.episodes < self.epsilon_decay_steps:
                    self.epsilon -= self.epsilon_decay
                else:
                    self.epsilon *= self.epsilon_exponential_decay

            self.episodes += 1
            self.rewards_history.append(self.episode_reward)
            self.steps_per_episode.append(self.episode_length)
            self.episode_reward, self.episode_length = 0, 0

        self.experience.append((s, a, r, s_prime, not_done))

    def experience_replay(self):
        if self.batch_size > len(self.experience):
            return
        minibatch = map(np.array, zip(*sample(self.experience, self.batch_size)))
        states, actions, rewards, next_states, not_done = minibatch

        next_q_values = self.online_network.predict_on_batch(next_states)
        best_actions = tf.argmax(next_q_values, axis=1)

        next_q_values_target = self.target_network.predict_on_batch(next_states)
        target_q_values = tf.gather_nd(next_q_values_target,
                                       tf.stack((self.idx, tf.cast(best_actions, tf.int32)), axis=1))

        targets = rewards + not_done * self.gamma * target_q_values

        q_values = self.online_network.predict_on_batch(states)
        q_values[[self.idx, actions]] = targets

        loss = self.online_network.train_on_batch(x=states, y=q_values)
        self.losses.append(loss)

        if self.total_steps % self.tau == 0:
            self.update_target()


#### FUNCTIONS ####
def format_time(t):
    m_, s = divmod(t, 60)
    h, m = divmod(m_, 60)
    return '{:02.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)


def set_up_gym(trading_periods):
    ## SET UP GYM ENVIRONMENT

    register(
        id='trading-v0',
        entry_point='trading_env:TradingEnvironment',  # this is where we call the trading_env.py
        max_episode_steps=trading_periods
    )


def init_agent(trading_periods, loaded_data={}):

    if not loaded_data:  #if no data loaded put our base values into loaded_data dictionary
        loaded_data = { 'trading_cost_bps': 0.0001, 'time_cost_bps': 1e-05, 'gamma': 0.99, 'tau': 100,
                        'architecture': [256, 256], 'learning_rate': 0.0001, 'l2_reg': 1e-06,
                        'replay_capacity': 1000000, 'batch_size': 4096, 'epsilon': 1.0, 'epsilon_start': 1.0,
                        'epsilon_end': 0.01, 'epsilon_decay_steps': 250, 'epsilon_exponential_decay': 0.99}


    ## INITIALIZING TRADING ENVIRONMENT
    trading_cost_bps = loaded_data["trading_cost_bps"]
    time_cost_bps = loaded_data["time_cost_bps"]

    f'Trading costs: {trading_cost_bps:.2%} | Time costs: {time_cost_bps:.0%}'

    trading_environment = gym.make('trading-v0',
                                   ticker='BTC',
                                   trading_periods=trading_periods,
                                   trading_cost_bps=trading_cost_bps,
                                   time_cost_bps=time_cost_bps)

    if 'episodes_run' in loaded_data:
        rseed = loaded_data["episodes_run"]  # for loaded values use episodes run as the seed
        loaded = True
    else:
        rseed = 42
        loaded = False
    #print("rseed is ", rseed)
    trading_environment.seed(rseed)
    #np.random.seed(rseed)
    #tf.random.set_seed(rseed)

    ## GET ENVIRONMENT PARAMS

    state_dim = trading_environment.observation_space.shape[0]
    num_actions = trading_environment.action_space.n
    max_episode_steps = trading_environment.spec.max_episode_steps

    ## DEFINE HYPERPARAMETERS

    gamma = loaded_data["gamma"]  # discount factor
    tau = loaded_data["tau"]     # target network update frequency

    ## NN ARCHITECTURE

    architecture = loaded_data["architecture"]   # units per layer
    learning_rate = loaded_data["learning_rate"]      # learning rate
    l2_reg = loaded_data["l2_reg"]              # L2 regularization

    ## EXPIRIENCE REPLAY

    replay_capacity = loaded_data["replay_capacity"]
    batch_size = loaded_data["batch_size"]

    ## e-GREEDY POLICY

    epsilon = loaded_data["epsilon"]
    epsilon_start = loaded_data["epsilon_start"]
    epsilon_end = loaded_data["epsilon_end"]
    epsilon_decay_steps = loaded_data["epsilon_decay_steps"]
    epsilon_exponential_decay = loaded_data["epsilon_exponential_decay"]

    ## CREATE DDQN AGENT
    tf.keras.backend.clear_session()

    ddqn = DDQNAgent(state_dim=state_dim,
                     num_actions=num_actions,
                     learning_rate=learning_rate,
                     gamma=gamma,
                     epsilon=epsilon,
                     epsilon_start=epsilon_start,
                     epsilon_end=epsilon_end,
                     epsilon_decay_steps=epsilon_decay_steps,
                     epsilon_exponential_decay=epsilon_exponential_decay,
                     replay_capacity=replay_capacity,
                     architecture=architecture,
                     l2_reg=l2_reg,
                     tau=tau,
                     batch_size=batch_size,
                     loaded=loaded)

    ddqn.online_network.summary()
    return (trading_environment, ddqn, state_dim)



def render_results(episode, navs, market_navs, diffs, trading_environment, render=False):
    """ calls the render function in trading_env.py
    This function should be edited to make any live graphs that we actualy care about. by default, rendering will be False.
    """
    if render:
        # copied from store analyze results
        results = pd.DataFrame({'Episode': list(range(1, episode + 1)),
                                'Agent': navs,
                                'Market': market_navs,
                                'Difference': diffs}).set_index('Episode')

        results['Strategy Wins (%)'] = (results.Difference > 0).rolling(100).sum()
        df1 = (results[['Agent', 'Market']]
               .sub(1)
               .rolling(100)
               .mean())
        df2 = results['Strategy Wins (%)'].div(100).rolling(50).mean()
        trading_environment.render(df1, df2)


def run_tests(trading_environment, ddqn, state_dim, max_episode_steps, max_episodes, loaded_data):
    ## RUN EXPERIMENT ##########################

    ## INITIALIZE VARIABLES

    episode_time, episode_eps = [], []
    # navs, market_navs, diffs, = [], [], []
    if not loaded_data:  # if we are not using loaded data
        navs, market_navs, diffs, = [], [], []
    else:
        navs = loaded_data["navs"]
        market_navs = loaded_data["market_navs"]
        diffs = loaded_data["diffs"]

    ## VISUALIZATION

    def track_results(episode, nav_ma_100, nav_ma_10,
                      market_nav_100, market_nav_10,
                      win_ratio, total, epsilon):
        time_ma = np.mean([episode_time[-100:]])
        T = np.sum(episode_time)

        template = '{:>4d} | {} | Agent: {:>6.1%} ({:>6.1%}) | '
        template += 'Market: {:>6.1%} ({:>6.1%}) | '
        template += 'Wins: {:>5.1%} | eps: {:>6.3f}'

        RESULT_SUMMARY.append([episode, format_time(total),
                              nav_ma_100 - 1, nav_ma_10 - 1,
                              market_nav_100 - 1, market_nav_10 - 1,
                              win_ratio, epsilon])
        print(template.format(episode, format_time(total),
                              nav_ma_100 - 1, nav_ma_10 - 1,
                              market_nav_100 - 1, market_nav_10 - 1,
                              win_ratio, epsilon))

    ## TRAIN AGENT

    start = time()

    for episode in range(1, max_episodes + 1):
        this_state = trading_environment.reset()
        for episode_step in range(max_episode_steps):
            action = ddqn.epsilon_greedy_policy(this_state.reshape(-1, state_dim))
            next_state, reward, done, _, _ = trading_environment.step(action)

            ddqn.memorize_transition(this_state,
                                     action,
                                     reward,
                                     next_state,
                                     0.0 if done else 1.0)
            if ddqn.train:
                ddqn.experience_replay()
            if done:
                break
            this_state = next_state

        # get DataFrame with seqence of actions, returns and nav values
        result = trading_environment.env.simulator.result()

        # get results of last step
        final = result.iloc[-1]

        # apply return (net of cost) of last action to last starting nav
        nav = final.nav * (1 + final.strategy_return)
        navs.append(nav)

        # market nav
        market_nav = final.market_nav
        market_navs.append(market_nav)

        # track difference between agent an market NAV results
        diff = nav - market_nav
        diffs.append(diff)

        if episode % 10 == 0:
            # set render to true if you want live rendering
            render_results(episode, navs, market_navs, diffs, trading_environment, render=False)
            track_results(episode,
                          # show mov. average results for 100 (10) periods
                          np.mean(navs[-100:]),
                          np.mean(navs[-10:]),
                          np.mean(market_navs[-100:]),
                          np.mean(market_navs[-10:]),
                          # share of agent wins, defined as higher ending nav
                          np.sum([s > 0 for s in diffs[-100:]]) / min(len(diffs), 100),
                          time() - start, ddqn.epsilon)
        if len(diffs) > 25 and all([r > 0 for r in diffs[-25:]]):
            print(result.tail())
            break

    return (episode, navs, market_navs, diffs, ddqn)


def print_result_summary(dir_name):

    rs = RESULT_SUMMARY
    template = '{:>4d} | {} | Agent: {:>6.1%} ({:>6.1%}) | '
    template += 'Market: {:>6.1%} ({:>6.1%}) | '
    template += 'Wins: {:>5.1%} | eps: {:>6.3f}'
    save_path = 'saved_model/' + dir_name + "/" + "results_summary.txt"

    with open(save_path, 'w') as f:
        for i in range(len(rs)):
            print(template.format(rs[i][0], rs[i][1], rs[i][2], rs[i][3], rs[i][4], rs[i][5], rs[i][6], rs[i][7]))
            f.writelines(template.format(rs[i][0], rs[i][1], rs[i][2], rs[i][3], rs[i][4], rs[i][5], rs[i][6], rs[i][7]) + "\n")


def store_analyze_results(episode, navs, market_navs, diffs, results_path):
    ## STORE RESULTS
    print("TYPES : ", type(episode), type(navs), type(market_navs), type(diffs))
    print(" episodes number is ", episode, "LENGTHS : ", len(navs), len(market_navs), len(diffs))
    ########################remove next line ###############
    episode += 100
    results = pd.DataFrame({'Episode': list(range(1, episode + 1)),
                            'Agent': navs,
                            'Market': market_navs,
                            'Difference': diffs}).set_index('Episode')

    results['Strategy Wins (%)'] = (results.Difference > 0).rolling(100).sum()
    results.info()

    results.to_csv(results_path / 'results.csv', index=False)

    with sns.axes_style('white'):
        sns.distplot(results.Difference)
        sns.despine()

    ## EVALUATE RESULTS

    results.info()

    fig, axes = plt.subplots(ncols=2, figsize=(14, 4), sharey=True)

    df1 = (results[['Agent', 'Market']]
           .sub(1)
           .rolling(100)
           .mean())
    df1.plot(ax=axes[0],
             title='Annual Returns (Moving Average)',
             lw=1)

    df2 = results['Strategy Wins (%)'].div(100).rolling(50).mean()
    df2.plot(ax=axes[1],
             title='Agent Outperformance (%, Moving Average)')

    for ax in axes:
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))
    axes[1].axhline(.5, ls='--', c='k', lw=1)

    sns.despine()
    fig.tight_layout()
    fig.savefig(results_path / 'performance', dpi=300)

def load_saved_data(dir_name):
    loaded_data = {}

    # read variables to save_data.csv
    load_path = 'saved_model/' + dir_name + "/" + "save_data.csv"
    with open(load_path, mode='r') as load_file:
        csv_reader = csv.reader(load_file, delimiter=',')
        for row in csv_reader:
            if len(row) > 2:
                #loaded_data[row[0]] = [x for x in row[1:]]
                loaded_data[row[0]] = [int(x) for x in row[1:]]
            else:
                value = float(row[1])         # convert to float first then switch to int if it should be an int
                loaded_data[row[0]] = (value, int(value))[row[1].find(".") == -1] # key is variable name
                #loaded_data[row[0]] = row[1]  # key is variable name, value is value od saved variable
    print(loaded_data["gamma"],  "  is gamma")
    # value = (value, int(value)) [row[1].find(".") == -1]
    print("Value is :", value)
    jl_load_path = 'saved_model/' + dir_name + "/"
    loaded_data["navs"] = joblib.load(jl_load_path + 'navs.sav')
    loaded_data["market_navs"] = joblib.load(jl_load_path + 'market_navs.sav')
    loaded_data["diffs"] = joblib.load(jl_load_path + 'diffs.sav')

    #print(loaded_data)
    #save_path = 'saved_model/'
    #savedModel = load_model(save_path)
    #savedModel.summary()
    return loaded_data

def save_file(final_ddqn, trading_env, dir_name, trading_periods, episodes_run, navs, market_navs, diffs):

    # save variables to save_data.csv
    save_path = 'saved_model/' + dir_name + "/" + "save_data.csv"
    with open(save_path, 'w') as f:
        f.writelines("gamma" + "," + str(final_ddqn.gamma) + "\n")
        f.writelines("tau" + "," + str(final_ddqn.tau) + "\n")
        f.writelines("architecture")
        for layer in final_ddqn.architecture:
            f.writelines("," + str(layer))
        f.writelines("\n")
        f.writelines("learning_rate" + "," + str(final_ddqn.learning_rate) + "\n")
        f.writelines("l2_reg" + "," + str(float(final_ddqn.l2_reg)) + "\n")
        f.writelines("replay_capacity" + "," + str(final_ddqn.replay_capacity) + "\n")
        f.writelines("batch_size" + "," + str(final_ddqn.batch_size) + "\n")
        f.writelines("epsilon" + "," + str(final_ddqn.epsilon) + "\n")
        f.writelines("epsilon_start" + "," + str(final_ddqn.epsilon_start) + "\n")
        f.writelines("epsilon_end" + "," + str(final_ddqn.epsilon_end) + "\n")
        f.writelines("epsilon_decay_steps" + "," + str(final_ddqn.epsilon_decay_steps) + "\n")
        f.writelines("epsilon_exponential_decay" + "," + str(final_ddqn.epsilon_exponential_decay) + "\n")
        f.writelines("trading_cost_bps" + "," + str(trading_env.trading_cost_bps) + "\n")
        f.writelines("time_cost_bps" + "," + str(trading_env.time_cost_bps) + "\n")
        f.writelines("trading_periods" + "," + str(trading_periods) + "\n")
        f.writelines("episodes_run" + "," + str(episodes_run))

    # save the nav, market_nav and diffs to seperate files as they are large lists (1 value per episode)
    jl_save_path = 'saved_model/' + dir_name + "/"
    joblib.dump(navs, jl_save_path + 'navs.sav')
    joblib.dump(market_navs, jl_save_path + 'market_navs.sav')
    joblib.dump(diffs, jl_save_path + 'diffs.sav')

    # save neural network weights to online_weights
    ow_save_path = 'saved_model/' + dir_name + "/online_weights/"
    final_ddqn.online_network.save(ow_save_path)
    tw_save_path = 'saved_model/' + dir_name + "/target_weights/"
    final_ddqn.target_network.save(tw_save_path)

def main():
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if gpu_devices:
        print('Using GPU')
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    else:
        print('Using CPU')
    sns.set_style('whitegrid')

    load = input("Do you want to load a saved file? (y to load) :")
    if load == "y" or load == "Y":
        loaded_data = load_saved_data("100steps")
        max_episodes = 10
        trading_periods = 252
        rseed = loaded_data["episodes_run"]   # use number of episodes run as the random seed
        np.random.seed(rseed)
        tf.random.set_seed(rseed)
        max_episode_steps = trading_periods
        set_up_gym(trading_periods)
        trading_environment, ddqn, state_dim = init_agent(trading_periods, loaded_data)
        #exit(0)
        episode, navs, market_navs, diffs, final_ddqn = run_tests(trading_environment, ddqn, state_dim,
                                                                 max_episode_steps, max_episodes, loaded_data)
    else:
        max_episodes = int(input("How many episodes do you want to run? (ex 100) :"))
        trading_periods = int(input("What length of trading period do you want to use (ex 252)? :"))
        np.random.seed(42)
        tf.random.set_seed(42)
        max_episode_steps = trading_periods
        set_up_gym(trading_periods)
        trading_environment, ddqn, state_dim = init_agent(trading_periods, {})
        episode, navs, market_navs, diffs, final_ddqn = run_tests(trading_environment, ddqn, state_dim,
                                                                  max_episode_steps, max_episodes, {})


    dir_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S") # create a unique directory name for the files
    save_dir = Path('saved_model', dir_name)
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    # after training store, print and save results
    store_analyze_results(episode, navs, market_navs, diffs, save_dir)
    print_result_summary(dir_name)
    save_file(final_ddqn, trading_environment, dir_name, trading_periods, max_episodes, navs, market_navs, diffs)

    trading_environment.close()

if __name__ == "__main__":
    main()
