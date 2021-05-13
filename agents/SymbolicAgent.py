import random

from scipy.spatial.distance import euclidean

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError

from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise

class SymbolicAgent:

	def __init__(self, state_dim: tuple, action_size: int, pre_training_images: np.array):

		self.state_dim = state_dim
		self.action_size  = action_size

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
		self.interaction_max_distance = 10
		
		self.type_transition_matrix = {}
		self.tracked_entities = []
		self.entity_types = [EntityType(np.full(self.number_of_convolutions , np.inf), 0)] # Initializes with null-type
		self.interactions_Q_fucntions = {}

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
		self.autoencoder.fit(train_data, train_data, validation=validation_data, verbose = 1, epochs=10 , batch_size = 64)


	def act(self, state, random_act = True);

		interactions = self.build_state_representation(state)

		for i in interactions:

			


	def build_state_representation(self, state: np.array):
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

			temporally_extended_entities.append(TimeExtendedEntity(te, most_likely_entity))

		# Updates the tracked entities to be the entites most recently detected
		self.tracked_entities = detected_entities

		interactions = []
		for tee in temporally_extended_entities:

			for tee2 in temporally_extended_entities:
				
				if tee!=tee2:
				# If both temporally extended entities are not equal check for interaction
				if euclidean(tee1.entity_before.position, tee2.entity_before.position) < self.interaction_max_distance and euclidean(tee1.entity_after.position, tee2.entity_after.position) < self.interaction_max_distance:
					if tee1.entity_before.entity_type.type_number == tee1.entity_after.entity_type.type_number and tee2.entity_before.entity_type.type_number == tee2.entity_after.entity_type.type_number:
						interactions.append(self.get_type_time_consistent_interaction(tee1, tee2))

		return interactions

	def get_interaction(self, tee1: TimeExtendedEntity, tee2 : TimeExtendedEntity):
		# Interaction within interaction radius

		types_before = [tee1.entity_before.entity_type, tee2.entity_before.entity_type]
		types_after = [tee1.entity_after.entity_type, tee2.entity_after.entity_type]
		# Defined as changes in relative distance between objects i.e. tee1.position
		distance_before = tee1.entity_before.position - tee2.entity_after.position
		distance_after = tee1.entity_after.position - tee2.entity_after.position
		loc_diff = distance_after - distance_before

		return Interaction(types_before, types_after, loc_diff)

	def get_type_time_consistent_interaction(self, tee1: TimeExtendedEntity, tee2 : TimeExtendedEntity):
		# Interaction where the enitities types before and after remain the same

		assert(tee1.entity_before.entity_type.type_number == tee1.entity_after.entity_type.type_number)
		assert(tee2.entity_before.entity_type.type_number == tee2.entity_after.entity_type.type_number)

		entity_types = sorted([tee1.entity_before.entity_type, tee2.entity_before.entity_type], key = lambda t: t.type_number)
		
		# Defined as changes in relative distance between objects i.e. tee1.position
		distance_before = tee1.entity_before.position - tee2.entity_after.position
		distance_after = tee1.entity_after.position - tee2.entity_after.position
		loc_diff = distance_after - distance_before

		return TypeTimeConsistentInteraction(entity_types, loc_diff)

	def same_entities_likelihoods(self, entity: Entity, detected_entities : [Entity]):

		entities_likelihood = []

		for de in detected_entities:
			# Compute the likelihood of the entity and detected_entity being the same

			spatial_proximity = (1/ (1+ euclidean(entity.position, de.position)))

			type_transition_probability = self.get_type_transition_probability(entity.entity_type, de.entity_type)

			neighboors_difference = (1/ (1 + abs(entity.number_of_neighboors - de.number_of_neighboors)))

			entity_likelihood = (self.spatial_proximity_weight*spatial_proximity + self.type_transition_weight *type_transition_probability + self.neighboor_weight * neighboors_difference)/ (self.spatial_proximity_weight + self.type_transition_weight + self.neighboor_weight)

			entities_likelihood.append(entity_likelihood)

		return zip(detected_entities, entities_likelihood)

	def get_type_transition_probability(self, type1: EntityType, type2: EntityType):


		if type1 not in self.type_transition_matrix:

			self.type_transition_matrix[type1] = {
				type1: 1, # Counts the number of times that type1 has transitioned to type1
				n:1,      # Counts the number of type transitions starting from type1
				}

		total_number_of_transitions = self.type_transition_matrix[type1][n]
		type2_count_number = self.type_transition_matrix[type1].get(type2, 0) # Gets the type2 count or returns 0 as default

		return type2_count_number/total_number_of_transitions

	def update_type_transition_matrix(self, type1: EntityType, type2: EntityType):

		if type2 not in self.type_transition_matrix[type1].keys():
			self.type_transition_matrix[type1][type2] = 1
		else:
			self.type_transition_matrix[type1][type2] += 1
		
		self.type_transition_matrix[type1][type2] += 1

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

		self.activation_threshold = (self.activation_threshold + np.mean(encoded_filters))/2
		highest_activation_features = np.max(encoded_filters, axis = 2)

		sufficient_salient_positions = np.argwhere(highest_activation_features > self.activation_threshold)

		detected_entities_sketches = []

		for position in sufficient_salient_positions:

			x, y = position
			activation_spectra = encoded_filters[x, y, :]

			new_entity_type_flag = True
			for entity_type in self.entity_types:
				if entity_type.check_same_type(activation_spectra, self.type_distance_threshold):

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



class Interaction:

	def __init__(self, types_before: [EntityType], types_after: [EntityType], loc_diff: np.array):

		self.types_before = self.ordered_types(types_before)
		self.types_after = self.ordered_types(types_after)
		self.loc_diff = loc_diff

	def ordered_types(self, entity_types: [EntityType]):

		return sorted(entity_types, key = lambda t: t.type_number)



class EntitySketch:
	"""
	This class represents only the position and entity type detected. It does not 
	have the number_of_neighboors attributes 
	"""

	def __init__(self, position : tuple, entity_type: EntityType):

		self.position = position
		self.entity_type = entity_type


class Entity:

	def __init__(self, position : tuple, entity_type: EntityType, number_of_neighboors: int):

		self.position = position
		self.entity_type = entity_type
		self.number_of_neighboors = number_of_neighboors

	def __eq__(self, other):

		if not isinstance(other, Entity):
			return False
		
		return self.position == other.position and self.entity_type == other.entity_type and self.number_of_neighboors = other.number_of_neighboors


class EntityType:

	def __init__(self, activation_spectra : np.array, type_number: int):

		self.type_spectra = activation_spectra
		self.type_number = type_number


	def check_same_type(activation_spectra: np.array, type_max_distance: float):

		return euclidean(self.type_spectra, activation_spectra) < type_max_distance

class TimeExtendedEntity:
	"""
	This class represents the a entity during two subsequent timesteps
	"""
	def __init__(self, entity_before: Entity, entity_after: Entity):

		self.entity_before = entity_before
		self.entity_after = entity_after

	def __eq__(self, other):

		if not isinstane(other, TimeExtendedEntity):
			return False
		return self.entity_before == other.entity_before and self.entity_after == other.entity_after

class TypeTimeConsistentInteraction:

	def __init__(self, entity_types: [EntityType], loc_diff : np.array):

		self.entity_types = entity_types
		self.loc_diff = loc_diff

	def __hash__(self):

		entity_type1, entity_type2 = self.entity_types
		unique_interaction_string = str(entity_type1.type_number) + "_" + str(entity_type2.type_number)

		pos_x, pos_y = self.loc_diff
		unique_interaction_string = unique_interaction_string + str(pos_x) + "_" + str(pos_y)
		
		return hash(unique_interaction_string)

	def __eq__(self, other):

		if not isinstance(other, TypeTimeConsistentInteraction):
			return False

		return self.entity_types[0].type_number == other.entity_types[0].type_number and self.entity_types[1].type_number == other.entity_types[1].type_number and self.loc_diff == other.loc_diff