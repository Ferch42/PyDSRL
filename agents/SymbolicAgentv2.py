import random
from collections import namedtuple
#from scipy.spatial.distance import euclidean

import pygame
import numpy as np

# namedtuple definitions
EntityType = namedtuple('EntityType', ['activation_spectra', 'type_number'])
Entity = namedtuple('Entity', ['position', 'entity_type'])
Interaction = namedtuple('Interaction', ['entity_type1', 'entity_type2', 'x_dist', 'y_dist'])

# Auxiliary_functions

euclidean = lambda x, y: np.sqrt(np.sum(np.square(np.subtract(x,y))))

class SymbolicAgentv2:

	def __init__(self, action_size: int):

		self.action_size = action_size
		# RL
		self.gamma = 0.99
		self.lr = 0.001
		self.epsilon = 1
		self.epsilon_decay = 0.9999

		# Auxiliary data structures
		self.entity_types = [EntityType(None, 0), EntityType('agent', 1), \
		EntityType('cross', 2), EntityType('circle', 3)] # Initializes with null-type
		self.interactions_Q_functions = {}
		self.states_dict = {}

		## max distances
		self.interaction_max_distance = 40

		self.viewer = None


	def act(self, state, random_act = True):
		
		Q_values = self.get_Q_total(self.get_state_representation(state))

		#print(Q_values)
		if random_act:
			if np.random.random() < self.epsilon:
				return np.random.choice(range(self.action_size))
		
		Q_max = Q_values.max()
		Q_max_indexes = [j for j in range(self.action_size) if Q_values[j]==Q_max] 
		
		
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

		td = reward + Q_after.max()- Q_before[action]


		for ib in interactions_before:
			# Interactions
			Q_int = self.get_q_value_function(ib)
			Q_int[action] = Q_int[action] + self.lr* td

		self.epsilon = max(0.1, self.epsilon*self.epsilon_decay)

	def extract_entities(self, state):
		
		agent_type = self.entity_types[1]
		cross_type = self.entity_types[2]
		circle_type = self.entity_types[3]

		agent_pos_x, agent_pos_y = int(state['agent'].left), int(state['agent'].top)
		detected_entities = [Entity(np.array([agent_pos_x, agent_pos_y]), agent_type)]

		for cross in state['entities']['cross']:

			if cross.alive:
				cross_x, cross_y = int(cross.left), int(cross.top)
				detected_entities.append(Entity(np.array([cross_x, cross_y]), cross_type))

		for circle in state['entities']['circle']:

			if circle.alive:
				circle_x, circle_y = int(circle.left), int(circle.top)
				detected_entities.append(Entity(np.array([circle_x, circle_y]), circle_type))

		return detected_entities

	def build_state_representation(self, state):
		"""
		Builds the state representation

		input: 
			state: np.array

		return: 
			interactions: [Interaction]
		"""
		detected_entities = self.extract_entities(state)
		
		n_entities = len(detected_entities)

		interactions = set()
		for i in range(n_entities-1):
			for j in range(i+1, n_entities):

				e1 = detected_entities[i]
				e2 = detected_entities[j]

				if euclidean(e1.position, e2.position) < self.interaction_max_distance:
					# Valid interaction
					# Sorting entities by their type in order to mantain consistency
					se1, se2 = sorted([e1, e2], key = lambda x: x.entity_type.type_number)
					x_dist, y_dist = se1.position - se2.position

					interactions.add(Interaction(se1.entity_type.type_number, \
						se2.entity_type.type_number, x_dist, y_dist))
		#print(interactions)
		return interactions

	def reset(self):
		pygame.display.quit()

	def save(self, path):
		pass

