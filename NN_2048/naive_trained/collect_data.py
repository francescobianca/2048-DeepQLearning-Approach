import gym_2048
import gym

env = gym.make('2048-v0')
env.reset()
score_requirement = 1000
intial_games = 100

def model_data_preparation():
  training_data = []
  accepted_scores = []
  for game_index in range(intial_games):
      score = 0
      game_memory = []
      previous_state = []
      done = False
      while not done:
          action = env.np_random.choice(range(4), 1).item()
          next_state, reward, done, info = env.step(action)

          if len(previous_state) > 0:
              game_memory.append([previous_state, action])

          previous_state = next_state
          score += reward
          if done:
              break

      if score >= score_requirement:
          accepted_scores.append(score)
          '''
          Valore mosse          
          LEFT = 0
          UP = 1
          RIGHT = 2
          DOWN = 3
          '''

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
              training_data.append([data[0], output])

      env.reset()

  return training_data
