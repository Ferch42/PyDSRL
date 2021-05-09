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

class SymbolicAgent:

	def __init__(self, state_dim: tuple, action_size: int, pre_training_images: np.array):

		self.state_dim = state_dim
		self.action_size  = action_size

		# Autoencoder
		self.build_autoencoder()
		self.train_autoencoder(pre_training_images)

		# Entity tracking
		self.activation_threshold = 0
		self.type_distance_threshold = 1
		self.tracked_entities = {}
		self.entity_types = []

	def build_autoencoder(self):
		"""
		Builds the model for the autoencoder
		"""
		input_state = Input(shape = self.state_dim)
		conv = Conv2D(8, (5,5), activation = 'relu')(input_state)
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
		self.autoencoder.fit(train_data, train_data, validation=validation_data,`verbose = 1, epochs=10 , batch_size = 64)


	def act(self, state, random_act = True);




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

			self.tracked_entities = set(detected_entities)

	def extract_entities(self, state):
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

		detected_entities = []

		for position in sufficient_salient_positions:

			x, y = postion
			activation_spectra = encoded_filters[x, y, :]

			if not self.entity_types:

				# Creating first type
				first_entity_type = EntityType(activation_spectra , 0)
				self.entity_types.append(first_entity_type)

				detected_entities.append(Entity(position, first_entity_type))

			else:

				# There are already some entity types detected
				new_entity_type_flag = True
				for entity_type in self.entity_types:

					if entity_type.check_same_type(activation_spectra, self.type_distance_threshold):

						detected_entities.append(Entity(position, entity_type))
						new_entity_type_flag = False
						break

				if new_entity_type_flag:

					# New Entity type detected
					new_entity_type = EntityType(activation_spectra , len(self.entity_types))
					self.entity_types.append(new_entity_type)

					detected_entities.append(Entity(position, new_entity_type))

		return detected_entities




class Entity:

	def __init__(self, position : tuple, entity_type: EntityType):

		self.position = position
		self.entity_type = entity_type


class EntityType:

	def __init__(self, activation_spectra : np.array, type_number: int):

		self.type_spectra = activation_spectra
		self.type_number = type_number


	def check_same_type(activation_spectra: np.array, type_max_distance: float):

		return euclidean(self.type_spectra, activation_spectra) < type_max_distance
