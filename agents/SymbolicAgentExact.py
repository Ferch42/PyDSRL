import random
from collections import namedtuple
from collections import deque
#from scipy.spatial.distance import euclidean
import pickle

import pygame
import numpy as np

from tqdm import tqdm
# namedtuple definitions
EntityType = namedtuple('EntityType', ['activation_spectra', 'type_number'])
Entity = namedtuple('Entity', ['position', 'entity_type'])
Interaction = namedtuple('Interaction', ['entity_type1', 'entity_type2', 'x_dist', 'y_dist'])

# Auxiliary_functions

euclidean = lambda x, y: np.sqrt(np.sum(np.square(np.subtract(x,y))))

np.set_printoptions(threshold=np.inf)


class SymbolicAgentExact:

	def __init__(self,action_size: int):

		self.action_size  = action_size

		# RL
		self.gamma = 0.99
		self.lr = 0.001
		self.epsilon = 1
		self.epsilon_decay = 0.999995
		self.batch_size = 32

		# Auxiliary data structures		
		self.state_Q_functions = {}
		self.experience_replay_buffer = deque(maxlen = 1_000)

		self.viewer = None


	def act(self, state, random_act = True):
		
		Q_values = self.get_q_value_function(self.get_state_representation(state))

		#print(Q_values)
		if random_act:
			if np.random.random() < self.epsilon:
				return np.random.choice(range(self.action_size))
		
		Q_max = Q_values.max()
		Q_max_indexes = [j for j in range(self.action_size) if Q_values[j]==Q_max] 
		
		return np.random.choice(Q_max_indexes)

	def get_q_value_function(self, i: str):

		if i not in self.state_Q_functions.keys():

			self.state_Q_functions[i] = np.zeros(self.action_size)

		return self.state_Q_functions[i]

	def get_state_representation(self, state):
		
		state_string = hash(str(state))

		return state_string

	def update(self, state, action, reward, next_state, done):

		self.experience_replay_buffer.append((self.get_state_representation(state),action, reward, self.get_state_representation(next_state), done))

		if len(self.experience_replay_buffer)> self.batch_size:
			batch = random.sample(self.experience_replay_buffer, self.batch_size)

			for experience in batch:
				self.remember(*experience)

		self.epsilon = max(0.1, self.epsilon*self.epsilon_decay)


	def remember(self, state, action, reward, next_state, done):
		
		Q_before = self.get_q_value_function(state)
		Q_after = self.get_q_value_function(next_state)

		td = reward + Q_after.max()- Q_before[action]

		Q_before[action] = Q_before[action] + self.lr* td


	def reset(self):
		pygame.display.quit()

	def save(self, path):

		pickle.dump(self.state_Q_functions, open(path + '_Q_values', "wb+"))


