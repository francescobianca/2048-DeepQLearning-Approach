import numpy as np
from keras.layers import Dense
from keras.models import Sequential

def build_model():
    model = Sequential()
    model.add(Dense(16, input_dim=16, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(4, activation='linear'))
    model.compile(loss='mse', optimizer="sgd")

    return model


def train_model(training_data):
    X = np.array([i[0] for i in training_data]).reshape(-1, 16)
    y = np.array([i[1] for i in training_data])
    model = build_model()

    model.fit(X, y, epochs=100)
    return model
