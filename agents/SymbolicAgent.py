import random
from collections import namedtuple
#from scipy.spatial.distance import euclidean

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
EntitySketch = namedtuple('EntitySketch',['position', 'entity_type'])
Entity = namedtuple('Entity', ['position', 'entity_type', 'number_of_neighboors'])
TimeExtendedEntity = namedtuple('TimeExtendedEntity', ['entity_before', 'entity_after'])
Interaction = namedtuple('Interaction', ['tee1', 'tee2', 'x_dist', 'y_dist'])

# Auxiliary_functions
euclidean = lambda x, y: np.sqrt(np.sum(np.square(np.subtract(x,y))))

class SymbolicAgent:

	def __init__(self, state_dim: tuple, action_size: int, pre_training_images: np.array):

		self.state_dim = state_dim
		self.action_size  = action_size

		# RL
		self.gamma = 0.99

		# Autoencoder
		self.number_of_convolutions = 8
		self.build_autoencoder()
		self.train_autoencoder(pre_training_images)

		# Entity tracking
		## thresholds
		self.entity_likelihood_threshold = 0.5
		self.activation_threshold = 0
		self.type_distance_threshold = 1
		
		## same entity weights
		self.spatial_proximity_weight = 1
		self.type_transition_weight = 1
		self.neighboor_weight = 1
		
		## max distances
		self.neighboor_max_distance = 10
		self.interaction_max_distance = 3
		
		self.type_transition_matrix = {}
		self.tracked_entities = []
		self.entity_types = [EntityType(np.full(self.number_of_convolutions , np.inf), 0)] # Initializes with null-type
		self.interactions_Q_functions = {}
		self.states_dict = {}

		# epsilon greedy
		self.epsilon = 0.1

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
		self.autoencoder.fit(train_data, train_data, validation_data=(validation_data, validation_data), verbose = 1, epochs=10 , batch_size = 64)


	def act(self, state, random_act = True):

		interactions = self.get_state_representation(state)

		Q_values = np.zeros(self.action_size)
		for i in interactions:
			Q_values += self.get_q_value_function(i)

		if random_act:
			if np.random.random() < self.epsilon:
				return np.random.choice(range(self.action_size))

		Q_max = Q_values.max()
		Q_max_indexes = [j for j in range(self.action_size) if Q_values[j]==Q_max] 
		
		return np.random.choice(Q_max_indexes)

	def get_q_value_function(self, i: Interaction):

		interaction_key = (i.tee1.entity_after.entity_type.type_number, \
							i.tee2.entity_after.entity_type.type_number, \
							i.x_dist, i.y_dist)

		if interaction_key not in self.interactions_Q_functions.keys():

			self.interactions_Q_functions[interaction_key] = np.zeros(self.action_size)

		return self.interactions_Q_functions[interaction_key]

	def update_q_value_function(self, i:Interaction, q_value: np.array):

		interaction_key = (i.tee1.entity_after.entity_type.type_number, \
							i.tee2.entity_after.entity_type.type_number, \
							i.x_dist, i.y_dist)

		self.interactions_Q_functions[interaction_key] = q_value

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
		
		if not self.tracked_entities:
			self.tracked_entities = detected_entities

		temporally_extended_entities = []

		for te in self.tracked_entities:

			likely_same_entity_list = self.same_entities_likelihoods(te, detected_entities)
			likely_same_entity_list = [e for e in likely_same_entity_list if e[1] > self.entity_likelihood_threshold] # Removing the entities with likelihood bellow the threshold
			likely_same_entity_list = sorted(likely_same_entity_list, key = lambda e: e[1], reverse = True)

			if not likely_same_entity_list:
				# It means that there are no matches for the tracked entity te and therefore it has disappeared
				null_entity_type = self.entity_types[0]
				most_likely_entity = Entity(te.position, null_entity_type, None)

			else:
				most_likely_entity =  likely_same_entity_list[0][0]

			# Update type transition matrix
			self.update_type_transition_matrix(te.entity_type, most_likely_entity.entity_type)
			
			if te.entity_type.type_number != most_likely_entity.entity_type.type_number:
				print("NOT EQUAL ENTITY TYPES")
				print('types before', te.entity_type.type_number)
				print('types after', most_likely_entity.entity_type.type_number)
			
			temporally_extended_entities.append(TimeExtendedEntity(te, most_likely_entity))

		# Updates the tracked entities to be the entites most recently detected
		self.tracked_entities = detected_entities

		interactions = []
		n_tee = len(temporally_extended_entities)

		for i in range(n_tee-1):
			for j in range(i+1, n_tee):
				tee1 = temporally_extended_entities[i]
				tee2 = temporally_extended_entities[j]
				interactions += self.get_interactions(tee1, tee2)

		return interactions

	def get_interactions(self, tee1: TimeExtendedEntity, tee2: TimeExtendedEntity):

		# check for spatial proximity (if the final distance between them is bellow a certain
		# distance threshold)
		if euclidean(tee1.entity_after.position, tee2.entity_after.position) < self.interaction_max_distance:
			# There is an interaction between the two entitites
			stee1, stee2 = sorted([tee1, tee2], key = lambda t: t.entity_after.entity_type.type_number)
			x_dist, y_dist = stee1.entity_after.position - stee2.entity_after.position
			return [Interaction(stee1, stee2, x_dist, y_dist)]
		else:
			return []
		

	def update(self, state, action, reward, next_state, done):

		interactions_before = self.get_state_representation(state)
		interactions_after = self.get_state_representation(next_state)

		print(len(interactions_before))
		interactions_after_dict = self.build_interactions_after_dict(interactions_after)

		for ib in interactions_before:

			ia = self.find_corresponding_interaction(ib, interactions_after_dict)

			Q_ib = self.get_q_value_function(ib).copy()
			Q_ia = self.get_q_value_function(ia).copy()

			Q_ib[action] = reward + self.gamma * Q_ia.max() - Q_ib[action]

			self.update_q_value_function(ib, Q_ib)

	def build_interactions_after_dict(self, interactions_after):

		interactions_after_dict = {}

		for ia in interactions_after:
			interaction_after_fingerprint = (ia.tee1.entity_before.entity_type.type_number, \
				ia.tee1.entity_before.position[0], ia.tee1.entity_before.position[1],\
				ia.tee2.entity_before.entity_type.type_number,\
				ia.tee2.entity_before.position[0], ia.tee2.entity_before.position[1])
			interactions_after_dict[interaction_after_fingerprint] = ia

		return interactions_after_dict

	def find_corresponding_interaction(self, i: Interaction, iad: dict):

		# Entity fingerprint after
		interaction_before_fingerprint = (i.tee1.entity_after.entity_type.type_number, \
			i.tee1.entity_after.position[0], i.tee1.entity_after.position[1],\
			i.tee2.entity_after.entity_type.type_number,\
			i.tee2.entity_after.position[0], i.tee2.entity_after.position[1])

		possible_interactions = []

		for ii in ia:
			# Possible interaction fingerprint
			interaction_after_fingerprint = (ii.tee1.entity_before.entity_type.type_number, \
				ii.tee1.entity_before.position[0], ii.tee1.entity_before.position[1],\
				ii.tee2.entity_before.entity_type.type_number,\
				ii.tee2.entity_before.position[0], ii.tee2.entity_before.position[1])

			if interaction_before_fingerprint == interaction_after_fingerprint:
				possible_interactions.append(ii)

		assert(len(possible_interactions)==1)
		return possible_interactions.pop()

	def same_entities_likelihoods(self, entity: Entity, detected_entities : [Entity]):

		entities_likelihood = []

		for de in detected_entities:
			# Compute the likelihood of the entity and detected_entity being the same

			spatial_proximity = (1/ (1+ euclidean(entity.position, de.position)))

			type_transition_probability = self.get_type_transition_probability(entity.entity_type, de.entity_type)

			neighboors_difference = (1/ (1 + abs(entity.number_of_neighboors - de.number_of_neighboors)))

			entity_likelihood = (self.spatial_proximity_weight*spatial_proximity + \
			 self.type_transition_weight *type_transition_probability + \
			 self.neighboor_weight * neighboors_difference)/ \
			(self.spatial_proximity_weight + self.type_transition_weight + self.neighboor_weight)

			entities_likelihood.append(entity_likelihood)

		return zip(detected_entities, entities_likelihood)

	def get_type_transition_probability(self, type1: EntityType, type2: EntityType):

		type1_number = type1.type_number
		type2_number = type2.type_number

		if type1_number not in self.type_transition_matrix:

			self.type_transition_matrix[type1_number] = {
				type1_number: 1, # Counts the number of times that type1 has transitioned to type1
				'n':1,      # Counts the number of type transitions starting from type1
				}

		total_number_of_transitions = self.type_transition_matrix[type1_number]['n']
		type2_count_number = self.type_transition_matrix[type1_number].get(type2_number, 0) # Gets the type2 count or returns 0 as default

		return type2_count_number/total_number_of_transitions

	def update_type_transition_matrix(self, type1: EntityType, type2: EntityType):
		
		type1_number = type1.type_number
		type2_number = type2.type_number
		
		if type2_number not in self.type_transition_matrix[type1_number].keys():
			self.type_transition_matrix[type1_number][type2_number] = 1
		else:
			self.type_transition_matrix[type1_number][type2_number] += 1
		
		self.type_transition_matrix[type1_number]['n'] += 1

	def extract_entities(self, state: np.array):
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
		self.activation_threshold = (np.percentile(highest_activation_features, 80) +self.activation_threshold)/2
		sufficient_salient_positions = np.argwhere(highest_activation_features > self.activation_threshold)

		detected_entities_sketches = []

		for position in sufficient_salient_positions:

			x, y = position
			activation_spectra = encoded_filters[x, y, :]

			new_entity_type_flag = True
			for entity_type in self.entity_types:
				if euclidean(entity_type.activation_spectra, activation_spectra) < self.type_distance_threshold:

					detected_entities_sketches.append(EntitySketch(position, entity_type))
					new_entity_type_flag = False
					break

			if new_entity_type_flag:

				# New Entity type detected
				new_entity_type = EntityType(activation_spectra , len(self.entity_types))
				self.entity_types.append(new_entity_type)

				detected_entities_sketches.append(EntitySketch(position, new_entity_type))

		entities_positions = [es.position for es in detected_entities_sketches]
		pairwise_neighboors_distances = pairwise.euclidean_distances(entities_positions, entities_positions)

		detected_entities = []

		for i, es in enumerate(detected_entities_sketches):

			number_of_neighboors = np.count_nonzero((pairwise_neighboors_distances[i] - self.neighboor_max_distance) < 0) - 1
			detected_entities.append(Entity(es.position, es.entity_type, number_of_neighboors))


		return detected_entities

	def reset(self):
		self.states_dict = {}


