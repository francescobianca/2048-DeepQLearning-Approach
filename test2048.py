import os, fnmatch, pickle

import numpy as np
import os, fnmatch, pickle
import csv
import numpy as np
import random

from game2048 import Game2048Env # logica del gioco 2048

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv2D, Flatten, Input
from keras.layers.merge import concatenate
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy, GreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger

from callbacks2048 import TestCall2048
from processors2048 import OneHotNNInputProcessor

# Create the environment for the DQN agent:
ENV_NAME = '2048'
env = Game2048Env()

NB_STEPS_TRAINING = int(5e5)

path = ''
data_filepath = 'data/'
if not os.path.exists(data_filepath):
    os.makedirs(data_filepath)
csv_filepath = data_filepath + 'test/test_steps_'+ str(NB_STEPS_TRAINING) +'.csv'
if os.path.exists(csv_filepath):
    exit()

random.seed(123)
np.random.seed(123)
env.seed(123)
PREPROC="onehot2steps"

processor = OneHotNNInputProcessor(num_one_hot_matrices=16)

model = Sequential()
model.add(Flatten(input_shape=(1, 4+4*4, 16,) + (4, 4)))
model.add(Dense(units=1024, activation='relu'))
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=4, activation='linear'))
    
memory = SequentialMemory(limit=6000, window_length=1)

TRAIN_POLICY = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=0.05, value_min=0.05, value_test=0.01, nb_steps=100000)
TEST_POLICY = EpsGreedyQPolicy(eps=.01)
dqn = DQNAgent(model=model, nb_actions=4, test_policy=TEST_POLICY, policy=TRAIN_POLICY, memory=memory, processor=processor,
                nb_steps_warmup=5000, gamma=.99, target_model_update=1000, train_interval=4, delta_clip=1.)

dqn.compile(Adam(lr=.00025), metrics=['mse'])

weights_filepath = data_filepath + 'train/weights_steps_'+ str(NB_STEPS_TRAINING) +'.h5f'
dqn.load_weights(weights_filepath)
with open(csv_filepath, 'w', newline='') as file:
    writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC, delimiter=';')
    writer.writerow(['episode', 'episode_steps', 'highest_score', 'max_tile'])
_callbacks = [TestCall2048(csv_filepath)] 
dqn.test(env, nb_episodes=500, visualize=False, verbose=1, callbacks=_callbacks)
