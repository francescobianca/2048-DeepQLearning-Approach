from os.path import exists
import csv
import numpy as np
import matplotlib.pyplot as plt
from rl.callbacks import Callback, TestLogger

class TestCall2048(TestLogger):
    def __init__(self, filePath):
        self.path = filePath

    def on_episode_end(self, episode, logs):       
        grid = self.env.get_board()
        template = 'episode: {episode}, max tile: {max_tile}, episode reward: {episode_reward:.3f}, episode steps: {nb_steps}'
        variables = {
            'episode': episode + 1,
            'max_tile': np.amax(grid),
            'episode_reward': logs['episode_reward'],
            'nb_steps': logs['nb_steps']
        }
        with open(self.path, 'a', newline='') as file:
            writer=csv.writer(file)
            writer.writerow([episode, logs['nb_steps'], logs['episode_reward'], np.amax(grid)])

    
class TrainCall2048(Callback):

    def __init__(self, filePath):
        self.observations = {}
        self.rewards = {}
        self.max_tile = {}
        self.step = 0
                
        self.episodes = []
        
        self.max_tiles = []
        self.episodes_rewards = []
        self.max_tiles_means = 0
        self.episodes_rewards_means = 0
        
        self.nb_episodes_for_mean = 50 
        self.episode_counter = 0

        # CSV file:
        if exists(filePath):
            csv_file = open(filePath, "a") # a = append
            self.csv_writer = csv.writer(csv_file, delimiter=',')
        else:
            csv_file = open(filePath, "w") # w = write (clear and restart)
            self.csv_writer = csv.writer(csv_file, delimiter=',')
            headers = ['episode', 'episode_steps', 'episode_reward', 'max_tile']
            self.csv_writer.writerow(headers)
        
    def on_episode_begin(self, episode, logs):
        self.observations[episode] = []
        self.rewards[episode] = []
        self.max_tile[episode] = 0
    
    def on_episode_end(self, episode, logs):
        self.episode_counter += 1
        self.episodes = np.append(self.episodes, episode + 1)
        self.max_tiles = np.append(self.max_tiles, self.max_tile[episode])
        self.episodes_rewards = np.append(self.episodes_rewards, np.sum(self.rewards[episode]))
        
        nb_step_digits = str(int(np.ceil(np.log10(self.params['nb_steps']))) + 1)
        template = '{step: ' + nb_step_digits + 'd}/{nb_steps} - episode: {episode}, episode steps: {episode_steps}, highest_score: {highest_score:.3f}, max tile: {max_tile}'
        variables = {
            'step': self.step,
            'nb_steps': self.params['nb_steps'],
            'episode': episode + 1,
            'episode_steps': len(self.observations[episode]),
            'highest_score': self.episodes_rewards[-1],
            'max_tile': self.max_tiles[-1]
        }
        print(template.format(**variables))
        
        # Save CSV:
        self.csv_writer.writerow((episode + 1, len(self.observations[episode]), self.episodes_rewards[-1], self.max_tiles[-1]))

        if self.episode_counter % self.nb_episodes_for_mean == 0 :
            self.max_tiles_means = np.append(self.max_tiles_means,np.mean(self.max_tiles[-self.nb_episodes_for_mean:]))
            
            self.episodes_rewards_means = np.append(self.episodes_rewards_means,np.mean(self.episodes_rewards[-self.nb_episodes_for_mean:]))

        del self.observations[episode]
        del self.rewards[episode]
        del self.max_tile[episode]

    def on_step_end(self, step, logs):
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.max_tile[episode] = logs['info']['max_tile']
        self.step += 1