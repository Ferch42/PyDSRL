import random
from collections import namedtuple
#from scipy.spatial.distance import euclidean

import pygame
from skimage.transform import resize, rescale
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError

from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise

# namedtuple definitions
EntityType = namedtuple('EntityType', ['activation_spectra', 'type_number'])
Entity = namedtuple('Entity', ['position', 'entity_type'])
Interaction = namedtuple('Interaction', ['entity_type1', 'entity_type2', 'x_dist', 'y_dist'])

# Auxiliary_functions

euclidean = lambda x, y: np.sqrt(np.sum(np.square(np.subtract(x,y))))

class SymbolicAgentv2:

	def __init__(self, action_size: int):

		# RL
		self.gamma = 0.99
		self.lr = 0.00001
		self.epsilon = 1
		self.epsilon_decay = 0.99998

		# Auxiliary data structures
		self.type_transition_matrix = {}
		self.tracked_entities = []
		self.entity_types = [EntityType(None, 0), EntityType('agent', 1), \
		EntityType('cross', 2), EntityType('circle', 3)] # Initializes with null-type
		self.interactions_Q_functions = {}
		self.states_dict = {}

		## max distances
		self.interaction_max_distance = 10

		self.viewer = None


	def act(self, state, random_act = True):
		
		Q_values = self.get_Q_total(self.get_state_representation(state))

		#print(Q_values)
		if random_act:
			if np.random.random() < self.epsilon:
				return np.random.choice(range(self.action_size))
		
		Q_max = Q_values.max()
		Q_max_indexes = [j for j in range(self.action_size) if Q_values[j]==Q_max] 
		
		self.epsilon = max(0.1, self.epsilon*self.epsilon_decay)
		
		return np.random.choice(Q_max_indexes)

	def get_q_value_function(self, i: Interaction):

		if i not in self.interactions_Q_functions.keys():

			self.interactions_Q_functions[i] = np.zeros(self.action_size)

		return self.interactions_Q_functions[i]

	def get_state_representation(self, state):

		state_string = str(state)

		if state_string not in self.states_dict.keys():
			self.states_dict[state_string] = self.build_state_representation(state)

		return self.states_dict[state_string]
	

	def get_Q_total(self, interactions):

		Q_values = np.zeros(self.action_size)
		for i in interactions:
			Q_values += self.get_q_value_function(i)

		return Q_values


	def update(self, state, action, reward, next_state, done):
		
		interactions_before = self.get_state_representation(state)
		interactions_after = self.get_state_representation(next_state)
		
		Q_before = self.get_Q_total(interactions_before)
		Q_after = self.get_Q_total(interactions_after)

		td = reward + Q_after.max() *(1- int(done)) - Q_before[action]

		if done:
			print(Q_before.mean())

		for ib in interactions_before:
			# Interactions
			Q_int = self.get_q_value_function(ib)
			Q_int[action] = Q_int[action] + self.lr* td

	def extract_entities(self, state: np.array, render_extracted = False):
		

	def reset(self):
		pygame.display.quit()

	def save(self, path):
		pass

