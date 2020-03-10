import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import os

def build_model():
    model = Sequential()
    model.add(Dense(16, input_dim=16, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(4, activation='linear'))
    model.compile(loss='mse', optimizer="sgd")

    return model


def train_model():
    data = np.load('play_game/giocata0.npy')
    x_load = []
    y_load = []
    count = 0
    while True:
        filepath = 'play_game/giocata'+str(count)+'.npy'
        if not os.path.exists(filepath):
            break
        count += 1
        data = np.load(filepath)
        print(data)
        for c in data[0]:
            x_load.append(c[0])
            y_load.append(c[1][0])
    X = np.array(x_load).reshape(-1, 16)
    y = np.array(y_load)
    model = build_model()

    model.fit(X, y, epochs=100)
    return model
