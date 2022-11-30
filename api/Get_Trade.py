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
from keras.optimizers import SGD
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
                 experience,
                 load_path,
                 loaded,
                 total_steps,
                 episodes,
                 steps_per_episode,
                 rewards_history,
                 losses):

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.replay_capacity = replay_capacity
        self.experience = experience
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.architecture = architecture
        self.l2_reg = l2_reg
        if loaded:
            self.online_network = load_model(load_path + 'online_weights/')
            self.target_network = load_model(load_path + 'target_weights/')
        else:
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

        self.total_steps = total_steps
        self.episodes = episodes
        self.train_steps = self.train_episodes = 0  # these appear to be unused
        self.episode_length = 0
        self.steps_per_episode = steps_per_episode
        self.episode_reward = 0
        self.rewards_history = rewards_history

        self.batch_size = batch_size
        self.tau = tau
        self.losses = losses
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
                      optimizer=SGD(lr=self.learning_rate))
        return model

    def update_target(self):
        self.target_network.set_weights(self.online_network.get_weights())

    def epsilon_greedy_policy(self, state):
        self.total_steps += 1
        q = self.online_network.predict(state)
        print("q is ", q)
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
    if not loaded_data:  # if no data loaded put our base values into loaded_data dictionary
        loaded_data = {'trading_cost_bps': 0.0001, 'time_cost_bps': .00001, 'gamma': 0.99, 'tau': 100,
                       'architecture': [256, 256], 'learning_rate': 0.0001, 'l2_reg': 1e-06,
                       'replay_capacity': 1000000, 'batch_size': 4096, 'epsilon': 1.0, 'epsilon_start': 1.0,
                       'epsilon_end': 0.01, 'epsilon_decay_steps': 250, 'epsilon_exponential_decay': 0.99, }

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
        load_path = loaded_data["dir_location"]
        total_steps = loaded_data["total_steps"]
        episodes = loaded_data["episodes_run"]
        steps_per_episode = loaded_data["steps_per_episode"]
        rewards_history = loaded_data["rewards_history"]
        losses = loaded_data["losses"]
    else:
        rseed = 42
        loaded = False
        load_path = ""
        total_steps = episodes = 0
        steps_per_episode = []
        rewards_history = []
        losses = []
    print("seed ", rseed)
    trading_environment.seed(rseed)

    ## GET ENVIRONMENT PARAMS

    state_dim = trading_environment.observation_space.shape[0]
    num_actions = trading_environment.action_space.n
    max_episode_steps = trading_environment.spec.max_episode_steps

    ## DEFINE HYPERPARAMETERS

    gamma = loaded_data["gamma"]  # discount factor
    tau = loaded_data["tau"]  # target network update frequency

    ## NN ARCHITECTURE

    architecture = loaded_data["architecture"]  # units per layer
    learning_rate = loaded_data["learning_rate"]  # learning rate
    l2_reg = loaded_data["l2_reg"]  # L2 regularization

    ## EXPIRIENCE REPLAY

    replay_capacity = loaded_data["replay_capacity"]
    batch_size = loaded_data["batch_size"]
    if loaded:
        experience = loaded_data["experience"]
    else:
        experience = deque([], maxlen=replay_capacity)

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
                     experience=experience,
                     load_path=load_path,
                     loaded=loaded,
                     total_steps=total_steps,
                     episodes=episodes,
                     steps_per_episode=steps_per_episode,
                     rewards_history=rewards_history,
                     losses=losses
                     )

    ddqn.online_network.summary()
    return (trading_environment, ddqn, state_dim)


def get_trade(ddqn, state_dim, current_state):
    print("current state is ", current_state)
    action = ddqn.epsilon_greedy_policy(current_state.reshape(-1, state_dim)) - 1  # -1 is short, 0 is hold cash, 1 is long
    print("action is ", action)
    return action


def load_NN():
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if gpu_devices:
        print('Using GPU')
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    else:
        print('Using CPU')

    load_dir_name = "final_bot"  # directory that holds the final saved model for live trading
    loaded_data = load_saved_data(load_dir_name)
    trading_periods = loaded_data["trading_periods"]

    set_up_gym(trading_periods)
    trading_environment, ddqn, state_dim = init_agent(trading_periods, loaded_data)
    return (ddqn, state_dim, trading_environment)


def close_NN(trading_environment):
    trading_environment.close()


def load_saved_data(dir_name):
    loaded_data = {}

    # read variables to save_data.csv
    load_path = 'AIML Bitcoin Trading Bot/saved_model/' + dir_name + "/" + "save_data.csv"
    with open(load_path, mode='r') as load_file:
        csv_reader = csv.reader(load_file, delimiter=',')
        for row in csv_reader:
            if len(row) > 2:
                loaded_data[row[0]] = [int(x) for x in row[1:]]
            else:
                value = float(row[1])  # convert to float first then switch to int if it should be an int
                # key is variable name, check if decimal or scientific notation ex e-05
                loaded_data[row[0]] = (value, int(value))[row[1].find(".") == -1 and row[1].find("e-") == -1]

    jl_load_path = 'saved_model/' + dir_name + "/"
    loaded_data["dir_location"] = jl_load_path
    loaded_data["navs"] = joblib.load(jl_load_path + 'navs.sav')
    loaded_data["market_navs"] = joblib.load(jl_load_path + 'market_navs.sav')
    loaded_data["diffs"] = joblib.load(jl_load_path + 'diffs.sav')
    loaded_data["experience"] = joblib.load(jl_load_path + 'experience.sav')
    loaded_data["losses"] = joblib.load(jl_load_path + 'losses.sav')
    loaded_data["steps_per_episode"] = joblib.load(jl_load_path + 'steps_per_episode.sav')
    loaded_data["rewards_history"] = joblib.load(jl_load_path + 'rewards_history.sav')
    global RESULT_SUMMARY
    RESULT_SUMMARY = joblib.load(jl_load_path + 'result_summary.sav')
    return loaded_data


def main():
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if gpu_devices:
        print('Using GPU')
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    else:
        print('Using CPU')

    load_dir_name = "200_100_200_500_300steps_eds500v2"  # directory that holds the final saved model for live trading
    loaded_data = load_saved_data(load_dir_name)
    trading_periods = loaded_data["trading_periods"]

    set_up_gym(trading_periods)
    trading_environment, ddqn, state_dim = init_agent(trading_periods, loaded_data)
    print("DIM ", state_dim)

    with open('../AIML Bitcoin Trading Bot/test.npy', 'rb') as f:
        test_data = np.load(f)
    print("test data ", test_data)
    for hour in test_data:
        get_trade(ddqn, state_dim, hour)

    trading_environment.close()


if __name__ == "__main__":
    main()
