import random
from collections import namedtuple
from collections import deque
import gc

import pickle
import pygame
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError

# namedtuple definitions
EntityType = namedtuple('EntityType', ['activation_spectra', 'type_number'])
Entity = namedtuple('Entity', ['position', 'entity_type'])
Interaction = namedtuple('Interaction', ['entity_type1', 'entity_type2', 'x_dist', 'y_dist'])

# Auxiliary_functions

euclidean = lambda x, y: np.sqrt(np.sum(np.square(np.subtract(x,y))))

class SymbolicAgentDQN:

	def __init__(self, action_size: int):

		self.action_size = action_size
		# RL
		self.gamma = 0.99
		self.lr = 0.001
		self.epsilon = 1
		self.epsilon_min = 0.1
		self.epsilon_decay = 0.999995

		# Auxiliary data structures
		self.entity_types = [EntityType(None, 0), EntityType('agent', 1), \
		EntityType('cross', 2), EntityType('circle', 3)] # Initializes with null-type
		self.interactions_Q_functions = {}
		self.states_dict = {}

		## max distances
		self.interaction_max_distance = 40

		self.viewer = None

		self.experience_replay_buffer = deque(maxlen = 1_000)
		self.batch_size = 32
		self.max_number_of_interactions = 10

		# DQN 
		self.memory = deque(maxlen=1_000)
		self.update_frequency = 4
		self.model = self._build_model()
		self.target_model = self._build_model()
		self.update_target_model()
		self.step = 0
		self.C = 10_000
		self.batch_size = 32
		self.loss_function = MeanSquaredError()
		self.optimizer = Adam(learning_rate=0.00025, clipnorm=1.0)

	def _build_model(self):
		# Neural Net for Deep-Q learning Model
		state_input = Input(shape = (self.max_number_of_interactions *4,))

		dense1 = Dense(128, activation = 'relu')(state_input)
		dense2 = Dense(128, activation = 'relu')(dense1)
		dense3 = Dense(128, activation = 'relu')(dense2)

		action_output = Dense(self.action_size, activation='linear')(dense3)

		model = Model(inputs = state_input, outputs = action_output)

		return model
		
	def update_target_model(self):
		# copy weights from model to target_model
		self.target_model.set_weights(self.model.get_weights())

	def act(self, state,random_act=True):

		s = self.get_state_representation(state)
		if random_act:
			if np.random.rand() <= self.epsilon:
				return random.randrange(self.action_size)

		act_values = self.model.predict(np.array([s]))

		return np.argmax(act_values[0])  # returns action


	def get_state_representation(self, state):

		state_string = str(state)

		if state_string not in self.states_dict.keys():
			self.states_dict[state_string] = self.build_state_representation(state)

		return self.states_dict[state_string]
	

	def replay(self, batch_size):

		minibatch = random.sample(self.memory, batch_size)

		states, rewards, next_states, actions, dones = [], [], [], [], []

		for s, a, r, ns, d in minibatch:

			states.append(s)
			actions.append(a)
			rewards.append(r)
			next_states.append(ns)
			dones.append(int(d))

		states, next_states = np.array(states), np.array(next_states)
		rewards = np.array(rewards)
		dones = np.array(dones)
		future_rewards = self.target_model.predict(next_states)

		updated_q_values = rewards + (1-dones) * (self.gamma* np.max(future_rewards, axis = 1))

		action_masks = tf.one_hot(actions, self.action_size)

		with tf.GradientTape() as tape:
			q_values = self.model(states)

			q_action = tf.reduce_sum(tf.multiply(q_values, action_masks), axis=1)

			loss = self.loss_function(updated_q_values, q_action)


		grads = tape.gradient(loss, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
		gc.collect()

	def update(self, state, action, reward, next_state, done):


		self.remember(self.get_state_representation(state), action, reward, self.get_state_representation(next_state), done)
		
		if len(self.memory) > self.batch_size and self.step % self.update_frequency==0:
			self.replay(self.batch_size)
  
		self.step +=1

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

		if self.step%self.C == 0:
			self.update_target_model()


	def remember(self, state, action, reward, next_state, done):

		self.memory.append((state, action, reward, next_state, done))


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

				if not (e1.entity_type.type_number == 1 or e2.entity_type.type_number == 1):

					continue

				if euclidean(e1.position, e2.position) < self.interaction_max_distance:
					# Valid interaction
					# Sorting entities by their type in order to mantain consistency
					se1, se2 = sorted([e1, e2], key = lambda x: x.entity_type.type_number)
					x_dist, y_dist = se1.position - se2.position

					interactions.add(Interaction(se1.entity_type.type_number, \
						se2.entity_type.type_number, x_dist, y_dist))
		#print(interactions)
		return self.create_vector_representation(interactions)

	def create_vector_representation(self, interactions):

		vector_representation = []

		sorted_interactions = sorted(interactions, key = lambda x: abs(x.x_dist) + abs(x.y_dist))

		count = 0

		for si in sorted_interactions:

			if si.entity_type2 == 2:
				vector_representation += [1,0]
			elif si.entity_type2 == 3:
				vector_representation += [0,1]
			else:
				raise Exception("invalid format")

			vector_representation += [si.x_dist/self.interaction_max_distance, si.y_dist/self.interaction_max_distance]
			count+=1

			if count == self.max_number_of_interactions:
				break

		if count!= self.max_number_of_interactions:

			number_of_remaining_zeros = [0]* (self.max_number_of_interactions- count) *4
			vector_representation += number_of_remaining_zeros

		#print(vector_representation)
		assert(len(vector_representation) == self.max_number_of_interactions*4)
		
		return np.array(vector_representation)

	def reset(self):
		pygame.display.quit()

	def save(self, path):
		self.model.save_weights(path+ '.h5')

