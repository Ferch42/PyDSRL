'''
MIT License

Copyright (c) 2017 Keon Kim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

SOURCE: https://github.com/keon/deep-q-learning/blob/master/ddqn.py
SOURCE: https://keras.io/examples/rl/deep_q_network_breakout/

'''
import pickle
import random
from collections import deque
import sys
import gc

from scipy import sparse
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError


class DQNAgent:
    def __init__(self, state_dim, action_size, memory_size = 1_000, 
        gamma = 0.99, init_epsilon = 1.5, final_epsilon = 0.1, epsilon_decay = 0.99999,
        lr = 0.00025, update_frequency = 4, batch_size = 32, C = 10_000):
        self.state_dim = state_dim
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma    # discount rate
        self.epsilon = init_epsilon  # exploration rate
        self.epsilon_min = final_epsilon
        self.epsilon_decay = epsilon_decay #0.99998
        self.learning_rate = lr
        self.update_frequency = update_frequency
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.step = 0
        self.C = C
        self.batch_size = batch_size
        self.loss_function = MeanSquaredError()
        self.optimizer = Adam(learning_rate=0.00025, clipnorm=1.0)
        self.states_table = dict()

        print(self.model.summary())


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        state_input = Input(shape = self.state_dim)

        conv1 = Conv2D(8, (5,5), activation = 'relu')(state_input)
        conv2 = Conv2D(8 , (3,3), activation = 'relu') (conv1)
        max_pool = MaxPooling2D((2,2))(conv2)
        flatten = Flatten()(max_pool)
        dense1 = Dense(32, activation = 'relu')(flatten)

        action_output = Dense(self.action_size, activation='linear')(dense1)

        model = Model(inputs = state_input, outputs = action_output)

        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())


    def remember(self, state, action, reward, next_state, done):

        self.memory.append((state, action, reward, next_state, done))

    def act(self, state,random_act=True):

        if random_act:
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)

            act_values = self.model.predict(np.array([state]))
        else:
            act_values = self.model.predict(np.array([state]))
        return np.argmax(act_values[0])  # returns action


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


        self.remember(state, action, reward, next_state, done)
        
        if len(self.memory) > self.batch_size and self.step % self.update_frequency==0:
            self.replay(self.batch_size)
  
        self.step +=1

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.step%self.C == 0:
            self.update_target_model()


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)