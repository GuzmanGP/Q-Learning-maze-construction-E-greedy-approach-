
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 09:31:26 2018

@author: Guzman GP
"""

from copy import deepcopy
import numpy as np
import random
import pickle
import seaborn as sns
from matplotlib import pyplot as plt



class State:
    '''
    This class stores the actualized situation of the grid and the current agent position. 
    This class owns hash properties in order to accelarate the comparing and searching process within the Q-table.
    In case it is necessary, there could be added additional attributes of certain circunstancial information to be
    associated with its respective quality values.

    Attributes:
        grid: Representation of the environment (as a bunch of linked states) to dealt with.
        agent_pos: state of the agent in each step.
    '''
    
    def __init__(self, grid, agent_pos):
        self.grid = grid
        self.agent_pos = agent_pos

    def __eq__(self, other):
        return isinstance(other, State) and self.grid == other.grid and self.agent_pos == other.agent_pos
    
    def __hash__(self):
        return hash(str(self.grid) + str(self.agent_pos))
    
    def __str__(self): 
        return f"State(grid={self.grid}, agent_pos={self.agent_pos})" 
    
    
class Qlearning:
    '''
    This class encompasses all the necessary tools to achieve the learning and inferring process through the
    Bellman's equation. 
    It will be necessary to instance this class in the module in which the environment will be created, that is,
    where the Q-learing algorithm will be applied.

    Attributes:
        grid_resources: This parameter contains the information of each string symbol of the grid. There can be also
            added any additional object with further utility along the Qlearning process, for example, any information 
            of the environment that cannot be described as an string-reward pair.
            The structure of this dictionary is as follows: {symbol_1: [associated_reward, ends_the_episode?], ..., otherObject}
                symbol_1(key): The string of the symbol to which the information of the value's list is associated.
                associated_reward (value1): The reward as a float, being positive if suitable or negative if avoidable.
                ends_the_episode?(value2): Boolean; True if stepping into this symbol (state) ends the episode, False otherwise.
        start_state: Is an instance of the State class created in the application module containing the grid and the position from
            which the learning process is going to start in each episode. It migth be necessary to rotate this state for a overall
            perspective learning of environment by the agent.
        inference_state: Similar to the start_state, but this one appears once completed the q-table, and represents the initial state
            from where to infer the optimal policy of the environment.
        q_table: Dictionary which stores as {State_instance: [action's quality value's list]}. Represents the suitability of each action
            in each state of the envirionment with a float obtained as a weighing of the collected experience through Bellman's equation.

    Methods:
        warn_if_obstacle: warns if the position (state) in which the agent starts is an obstacle and therefore, not valid initial state.
        observe_reward_value: Recieves the symbol stepped on and gives the respective reward and the boolean's possible endding episode.
        extract_possible_actions: Actions along the grid are defined here. Currently as spatially movements directions. If necessary they 
            re-designed for other applications. 
        choose_action: This applies the E-greedy policy strategy. It could be possible to be changed for further utilities, for example, 
            to Boltzmanns strategy.
        infer_path: Deduction of the optimal policy from the Q-table with a given initial state and maximum range of steps.
        visualize_inferenced_path: String logged representation of the states resulted as optimal policy.
        visualize_max_quality_action: Seaborn and matplotlib colored representation of the maximum Qvalue in each state.
        learn: Bellman's equation process. This function coordinates most of the previously described ones and fills the q_table attribute
            with the experience collected through the specified trial and error method.
    '''
    
    def __init__(self, grid_resources, start_state = None, inference_state = None):

        self.start_state = start_state
        self.inference_state = inference_state
        self.grid_resources = grid_resources
        self.q_table = dict() 
        
        self.warn_if_obstacle()
    
    def warn_if_obstacle (self):
        '''
        Warns if the position (state) in which the agent starts is an obstacle and therefore, not valid initial state.
        '''
        if self.inference_state is not None:
            x, y = self.inference_state.agent_pos
            if self.inference_state.grid[y][x] == self.grid_resources['obstacle']:
                raise ValueError (f"Current position in inferencing process ({x,y}) is an obstacle.")
        elif self.start_state is not None:
            x, y = self.start_state.agent_pos
            if self.start_state.grid[y][x] == self.grid_resources['obstacle']:
                raise ValueError (f"Current position in learning process ({x,y}) is an obstacle.")
                
    def observe_reward_value(self, state, action):  
        '''
        Recieves the symbol stepped on and gives the respective reward and the boolean's possible endding episode.
        '''  
        x,y = action
        grid_item = state.grid[y][x]          
        new_grid = deepcopy(state.grid) # in case you want to modify the current grid
        values = self.grid_resources.get(grid_item, f"Unknown grid item {grid_item} appeared in cell {action}")
        if isinstance(grid_item, float) is True:
                reward = grid_item 
                is_done = False
        elif isinstance(values, str) is True:
            raise ValueError(f"{values}")
        else:
            reward, is_done = values
                    
        return State(grid=new_grid, agent_pos=(x,y)), reward, is_done
    
    possible_actions = []
    
    def extract_possible_actions(self, state):
        '''
        Actions along the grid are defined here. Currently as spatially movements directions. If necessary they 
        can be re-designed for other applications. 
        '''
        x,y = state.agent_pos
        around_area = [(x, y-1), (x, y+1), (x-1, y), (x+1, y), (x+1,y+1), (x-1,y-1), (x+1,y-1), (x-1,y+1)]
        self.possible_actions.clear()
        for _x, _y in around_area:
                aux = False
                if _x > (len(state.grid[0]) - 1) or _y > (len(state.grid) - 1): 
                    continue
                elif _x < 0 or _y < 0: 
                    continue
                elif state.grid[_y][_x] == self.grid_resources['obstacle']: 
                    continue
                self.possible_actions.append((_x, _y))
        if len(self.possible_actions) == 0:
            raise ValueError(f"Can't go anywhere from cell {x,y}.")
        return self.possible_actions
    
                
    def choose_action(self, state):
        '''
        This applies the E-greedy policy strategy. It could be possible to be changed for further utilities, for example, 
        to Boltzmanns strategy.
        '''
        if random.uniform(0, 1) < self.eps: 
            return random.choice(list(range(len(self.q_table[state]))))
        else: 
            return np.argmax(self.q_table[state])
        
        
    def infer_path (self, n_episode_steps, trained_agent_start):
        '''
        Deduction of the optimal policy from the Q-table with a given initial state and maximum range of steps.
        '''
        state = trained_agent_start        
        path = []
        tot_reward = 0
        for _ in range(n_episode_steps):
            action = np.argmax(self.q_table[state])
            new_state, reward, done = self.observe_reward_value(state, self.extract_possible_actions(state)[action])
            state = new_state
            tot_reward += reward
            path.append(new_state.agent_pos)
            if done == True:
                return path, tot_reward
            
    def q_value_ascii_action (self, q_table, grid):
        '''
        Arrow Unicode representation of the quality values per state
        '''
        visualization_grid = deepcopy(grid)
        action = 0

        for x in range(len(visualization_grid[0])):
            for y in range(len(visualization_grid)):
                ascii_per_action = {(x, y-1): '\u21D1', (x, y+1): '\u21D3', (x-1, y):'\u21D0', (x+1, y):'\u21D2', (x+1,y+1):'\u21D8',   (x-1,y-1):'\u21D6', (x+1,y-1):'\u21D7', (x-1,y+1):'\u21D9'}
                if visualization_grid[y][x] == self.grid_resources['obstacle']:
                    visualization_grid[y][x] = '\u2587'
                elif  visualization_grid[y][x] == self.grid_resources['goal']:
                    visualization_grid[y][x] = '\u25C9'
                else:
                    state = State(grid=grid, agent_pos=(x,y))
                    try:
                        action = np.argmax(self.q_table[state])
                    except Exception as e:
                        print(x,y)
                    act = self.extract_possible_actions(state)[action]
                    visualization_grid[y][x] = ascii_per_action[act] 
        for row in visualization_grid: #reversed mode> reversed(visualization_grid):
            print(' '.join(map(str, row))) 
    
    def visualize_inferenced_path (self, path):
        '''
        String logged representation of the states resulted as optimal policy.
        '''
        visualization_grid = deepcopy(self.inference_state.grid)
        for x,y in path:
            visualization_grid[y][x] = "\u2638"
        for row in visualization_grid: #reversed mode> reversed(visualization_grid):
            print(' '.join(map(str, row)))
            
    def visualize_max_quality_action (self, q_table, grid):
        '''
        Seaborn and matplotlib colored representation of the maximum Qvalue in each state.
        '''
        min_value = -1 * self.grid_resources[self.grid_resources['goal']][0]
        visualization_grid = np.full((len(grid), len(grid[0])), min_value)
        for k,v in q_table.items():
            max_q_x, max_q_y = k.agent_pos
            max_q_action  = np.argmax(v)
            visualization_grid[max_q_y][max_q_x] = max_q_action
        plt.figure(figsize=(8, 8))
        sns.heatmap(visualization_grid, linewidths=.2)
        plt.show();plt.close()    
    
    def convert_to_pickle (self, pickable_object, file_name):
        outfile = open(file_name,'wb')
        pickle.dump(pickable_object, outfile) 
        outfile.close()
        
    def extract_from_pickle(self,file_name):
        infile = open(file_name, 'rb')
        pickable_object = pickle.load(infile)
        infile.close()
        return pickable_object
    
    gamma = 1.0
    eps = 1.0 
    
    def learn (self, n_episodes, n_episode_steps):
        '''
        Bellman's equation process. This function coordinates most of the previously described ones and fills the q_table attribute
            with the experience collected through the specified trial and error method.
        '''
        #alphas = np.linspace(1.0, min_alpha, n_episodes)
        alphas = np.linspace(1.0, 0.02, n_episodes)
        
        # The four auxiliar variables below are created for plotting prusposes, they do not affect the learning process.
        sample_total_reward = 0
        previous_sample_reward = 0
        l = []
        goal_reached = False
          
        for e in range(n_episodes): 
            state = self.start_state 
            total_reward = 0
            alpha = alphas[e]

            for _ in range(n_episode_steps):
                possible_actions = self.extract_possible_actions(state)
                if state not in self.q_table:
                    self.q_table[state] = np.zeros(len(possible_actions))
                action = self.choose_action(state) 
                next_state, reward, done = self.observe_reward_value(state, possible_actions[action])
                total_reward += reward 
                if next_state not in self.q_table:
                    self.q_table[next_state] = np.zeros(len(self.extract_possible_actions(next_state)))
                self.q_table[state][action] = self.q_table[state][action] + \
                         alpha * (reward + self.gamma *  np.max(self.q_table[next_state]) - self.q_table[state][action])
                state = next_state
                if done:
                    break                   
              
            #eps = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*e) 
            self.eps = 0.01 + (1.0 - 0.01)*np.exp(-0.01*e) 
    
            print(f"#   Episode {e}/{n_episodes-1} with total reward {total_reward} ", end="\r", flush=True)
            
            # All the code below is just for plotting purposes, it does not affect the learning process.
            try:
                sample_path, sample_total_reward = self.infer_path(n_episode_steps, self.start_state)
                previous_sample_reward = sample_total_reward
            except Exception as x:
                sample_total_reward = previous_sample_reward
            l.append(sample_total_reward)
            
        x,y = sample_path[len(sample_path)-1]
        if self.start_state.grid[y][x] == self.grid_resources['goal']:
            goal_reached = True
        
        return l, goal_reached   

       
    
