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

np.set_printoptions(threshold=np.inf)


def duplicate_matrix(mat):

	mat_i, mat_j = mat.shape
	dup_mat = np.zeros(shape = (mat_i*2, mat_j*2))

	for i in range(mat_i):
		for j in range(mat_j):
			dup_mat[2*i][2*j] = mat[i][j]
			dup_mat[2*i+1][2*j] = mat[i][j]
			dup_mat[2*i][2*j+1] = mat[i][j]
			dup_mat[2*i+1][2*j+1] = mat[i][j]

	return dup_mat

class SymbolicAgent:

	def __init__(self, state_dim: tuple, action_size: int, pre_training_images: np.array):

		self.state_dim = state_dim
		self.action_size  = action_size

		# Number of convolutions
		self.number_of_convolutions = 8

		# RL
		self.gamma = 0.99
		self.lr = 0.00001
		self.epsilon = 1
		self.epsilon_decay = 0.99998

		# Auxiliary data structures
		self.entity_types = [EntityType(np.full(self.number_of_convolutions , np.inf), 0)] # Initializes with null-type
		self.interactions_Q_functions = {}
		self.states_dict = {}

		# Entity tracking
		## thresholds
		self.activation_threshold = 0
		self.type_distance_threshold = 0.5
		

		## max distances
		self.interaction_max_distance = 10
		
		# Autoencoder
		self.build_autoencoder()
		self.train_autoencoder(pre_training_images)

		self.viewer = None

	def build_autoencoder(self):
		"""
		Builds the model for the autoencoder
		"""
		input_state = Input(shape = self.state_dim)
		conv = Conv2D(self.number_of_convolutions, (5,5), activation = 'relu')(input_state)
		max_pool = MaxPooling2D((2,2))(conv)
		up_sample = UpSampling2D((2,2))(max_pool)
		conv_trans = Conv2DTranspose(1, (5,5), activation = 'sigmoid')(up_sample)

		self.encoder = Model(input_state, max_pool)
		self.autoencoder = Model(input_state, conv_trans)
		self.autoencoder.compile(optimizer=Adam(learning_rate=0.0001, clipnorm=1.0), loss='binary_crossentropy', metrics=['accuracy'])

	def train_autoencoder(self, pre_training_images: np.array):
		"""
		Trains the autoencoder given the pre_training_images
		"""
		print("Training autoencoder...")
		train_data, validation_data = train_test_split(pre_training_images, test_size=0.1)
		self.autoencoder.fit(train_data, train_data, validation_data=(validation_data, validation_data), verbose = 1, epochs=10, batch_size = 64)

		# Using the validation data in order to estimate activation threshold
		thresholds_estimate_list = []

		for v in validation_data:
			encoded_filters = self.encoder.predict(np.expand_dims(v, axis = 0))[0]
			highest_activation_features = np.max(encoded_filters, axis = 2)
			thresholds_estimate_list.append(np.percentile(highest_activation_features, 90))

		self.activation_threshold = np.mean(thresholds_estimate_list)


	def render_image(self, state):


		#if self.viewer is None:
		pygame.init()
		self.viewer = pygame.display.set_mode((350, 350))
		self.clock = pygame.time.Clock()
		s = state
		if len(state.shape)==3:
			s = duplicate_matrix(duplicate_matrix(np.squeeze(state, axis = 2)))
		else:
			s = duplicate_matrix(duplicate_matrix(duplicate_matrix(s)))
		squeezed_combined_state =  s*255
		surf = pygame.surfarray.make_surface(squeezed_combined_state).convert_alpha()
		pygame.event.poll()
		self.viewer.blit(surf, (0, 0))
		self.clock.tick(60)
		pygame.display.update()


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

		return interactions


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

		td = reward + Q_after.max()  - Q_before[action]

		for ib in interactions_before:
			# Interactions
			Q_int = self.get_q_value_function(ib)
			Q_int[action] = Q_int[action] + self.lr* td
			
		self.epsilon = max(0.1, self.epsilon*self.epsilon_decay)

	def extract_entities(self, state: np.array, render_extracted = False):
		"""
		Extracts the entities and their locations 

		input:
			state: np.array

		return: 
			entities = [Entities]
		"""
		# Predict Encoded latent spaces
		encoded_filters = self.encoder.predict(np.expand_dims(state, axis = 0))[0]

		highest_activation_features = np.max(encoded_filters, axis = 2)
		#self.activation_threshold = (np.percentile(highest_activation_features, 80) +self.activation_threshold)/2
		sufficient_salient_positions = np.argwhere(highest_activation_features > self.activation_threshold)

		detected_entities = []

		for position in sufficient_salient_positions:

			x, y = position
			activation_spectra = encoded_filters[x, y, :]

			new_entity_type_flag = True
			for entity_type in self.entity_types:
				if euclidean(entity_type.activation_spectra, activation_spectra) < self.type_distance_threshold:

					detected_entities.append(Entity(position, entity_type))
					new_entity_type_flag = False
					break

			if new_entity_type_flag:

				# New Entity type detected
				new_entity_type = EntityType(activation_spectra , len(self.entity_types))
				self.entity_types.append(new_entity_type)

				detected_entities.append(Entity(position, new_entity_type))

		detected_entities_img = np.zeros(shape = (highest_activation_features.shape))

		if render_extracted:

			print('original state')
			self.render_image(state)
			input()
			for de in detected_entities:

				x,y = de.position
				detected_entities_img[x][y] = de.entity_type.type_number* 0.1

			print('enitites')
			self.render_image(detected_entities_img)
			input()

			p = self.autoencoder.predict(np.array([state]))[0]
			print('reconstructed image')
			self.render_image(p)
			input()

		return detected_entities

	def reset(self):
		pygame.display.quit()

	def save(self, path):
		pass

