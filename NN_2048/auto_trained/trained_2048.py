import gym
from game2048 import Game2048Env  # logica del gioco 2048
from train_model import train_model
from collect_data import model_data_preparation
import numpy as np
import os
import csv

ENV_NAME = '2048'
env = Game2048Env()
env.reset()
scores = []

csv_filepath = 'test.csv'
if os.path.exists(csv_filepath):
    print("ATTENZIONE: file csv già presente. Eliminarlo e rieseguire se serve")
    exit()
with open(csv_filepath, 'w', newline='') as file:
    writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
    writer.writerow(['episode', 'episode_steps', 'highest_score', 'max_tile'])

def AI_2048(trained_model, threshold, test_active):
    choices = []
    accepted_scores = []
    value = 0
    train_data = []
    episode = 0
    for each_game_ in range(200):
        episode += 1
        score = 0
        game_memory = []
        previous_state = []
        fallimento = []
        done = False
        episode_steps = 0
        while not done:
            episode_steps += 1
            if len(previous_state) == 0 or np.array_equal(fallimento, previous_state):
                action = env.np_random.choice(range(4), 1).item()
            else:
                action = np.argmax(trained_model.predict(previous_state.reshape(-1, 16))[0])

            choices.append(action)
            next_state, reward, done, info = env.step(action)

            if len(previous_state) > 0:
                game_memory.append([previous_state, action])

            fallimento = previous_state
            previous_state = next_state
            score += reward
            if done:
                break

        if score >= threshold:
            print(score)
            accepted_scores.append(score)
            '''
            Valore mosse          
            LEFT = 0
            UP = 1
            RIGHT = 2
            DOWN = 3
            '''
            if test_active:
                with open(csv_filepath, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([episode, episode_steps, score, info])
            # For every step perform an hot encoding
            for data in game_memory:
                if data[1] == 0:
                    output = [1, 0, 0, 0]
                elif data[1] == 1:
                    output = [0, 1, 0, 0]
                elif data[1] == 2:
                    output = [0, 0, 1, 0]
                else:
                    output = [0, 0, 0, 1]
                train_data.append([data[0], output])
                if value == 0:
                    value = score
                elif value > score:
                    value = score
        env.reset()

    scores.append(value)
    return train_data

nb_train = 50
trained_data = model_data_preparation()
for x in range(nb_train):
    if x+1 < nb_train:
        trained_model = train_model(trained_data)
        trained_data = AI_2048(trained_model, 1000, False)
    else:
        trained_model = train_model(trained_data)
        trained_data = AI_2048(trained_model, 1000, True)

