import numpy as np
from game2048 import Game2048Env
from rl.core import Processor

    
class OneHotNNInputProcessor(Processor):
    
    def __init__(self, num_one_hot_matrices=16, window_length=1, model="dnn"):
        
        self.num_one_hot_matrices = num_one_hot_matrices
        self.window_length = window_length
        self.model = model
        
        self.game_env = Game2048Env() 
        self.table = {2**i:i for i in range(1,self.num_one_hot_matrices)} 
        self.table[0] = 0 
    
    def one_hot_encoding(self, grid):
        
        grid_onehot = np.zeros(shape=(self.num_one_hot_matrices, 4, 4))
        for i in range(4):
            for j in range(4):
                grid_element = grid[i, j]
                grid_onehot[self.table[grid_element],i, j]=1
        return grid_onehot


    def get_grids_next_step(self, grid):
        
        grids_list = [] 
        for movement in range(4):
            grid_before = grid.copy()
            self.game_env.set_board(grid_before)
            try:
                _ = self.game_env.move(movement) 
            except:
                pass
            grid_after = self.game_env.get_board()
            grids_list.append(grid_after)
        return grids_list
    
    def process_observation(self, observation):
        observation = np.reshape(observation, (4, 4))
        
        grids_list_step1 = self.get_grids_next_step(observation)
        grids_list_step2 =[]
        for grid in grids_list_step1:
            grids_list_step2.append(grid) 
            grids_temp = self.get_grids_next_step(grid)
            for grid_temp in grids_temp:
                grids_list_step2.append(grid_temp)
        grids_list = np.array([self.one_hot_encoding(grid) for grid in grids_list_step2])
        
        return grids_list