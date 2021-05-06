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


'''
import pickle
import random
from collections import deque

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError


class DQNAgent:
    def __init__(self, state_dim, action_size, memory_size = 10000, 
        gamma = 0.99, init_epsilon = 1.0, final_epsilon = 0.1, epsilon_decay = 0.999,
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

        print(self.model.summary())


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        state_input = Input(shape = self.state_dim)

        conv1 = Conv2D(8, (5,5), activation = 'relu')(state_input)
        conv2 = Conv2D(8 , (3,3), activation = 'relu') (conv2)
        max_pool = MaxPooling2D((2,2))(conv2)
        flatten = Flatten()(max_pool)
        dense1 = Dense(32, activation = 'relu')(flatten)
        """
        dense2 = Dense(128 , activation = 'relu')(state_input)
        dense3 = Dense(128, activation = 'relu')(dense2)
        dense4 = Dense(128, activation = 'relu')(dense3)        
        """
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

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update(self, state, action, reward, next_state, done):


        self.remember(state, action, reward, next_state, done)
        
        if len(self.memory) > self.batch_size and self.step % self.update_frequency==0:
            self.replay(self.batch_size)
        
        self.step +=1
        
        if self.step%self.C == 0:
            self.update_target_model()


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class TabularAgent:
    '''RL agent as described in the DSRL paper'''
    def __init__(self, action_size,alpha,epsilon_decay,neighbor_radius=25):
        self.action_size = action_size
        self.alpha = alpha
        self.epsilon = 1
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.1
        self.gamma = 0.95
        self.neighbor_radius=neighbor_radius
        self.offset = neighbor_radius*2
        self.tables = {}

    def act(self, state,random_act=True):
        '''
        Determines action to take based on given state
        State: Array of interactions
               (entities in each interaction are presorted by type for consistency)
        Returns: action to take, chosen e-greedily
        '''
        if not random_act:
            return np.argmax(self._total_rewards(state))
        if np.random.rand() <= self.epsilon:
            #print('random action, e:', self.epsilon)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            return random.randrange(self.action_size)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return np.argmax(self._total_rewards(state))  # returns action

    def update(self, state, action, reward, next_state, done):
        '''Update tables based on reward and action taken'''



        for interaction in state:
            type_1, type_2 = interaction['types_after'] # TODO resolve: should this too be types_before?
            table = self.tables.setdefault(type_1, {}).setdefault(type_2, self._make_table())
            id1,id2 = interaction['interaction']
            interaction_next_state = [inter for inter in next_state if inter['interaction']==(id1,id2)]
            if len(interaction_next_state)==0:
                continue
            elif len(interaction_next_state)>1:
                raise ValueError('This should not happen')
            else:
                #print('Now we should update the Q-values')
                #print(f'The current reward is {reward}')
                interaction_next_state = interaction_next_state[0]
                interaction['loc_difference'] = (interaction['loc_difference'][0]+self.offset,interaction['loc_difference'][1]+self.offset)
                interaction_next_state['loc_difference'] = (interaction_next_state['loc_difference'][0]+self.offset,interaction_next_state['loc_difference'][1]+self.offset)
                #print(interaction_next_state['loc_difference'])
                #print(interaction['loc_difference'])
                next_action_value = table[interaction_next_state['loc_difference']]
                #print(f'The next action value {next_action_value}')
                if done:
                    table[interaction['loc_difference']][action] = reward
                else:
                    #print(f'Q-value before update {table[interaction["loc_difference"]][action]}')
                    #print(f'Location {interaction["loc_difference"]}')
                    #print(f"The new value should be {table[interaction['loc_difference']][action] + self.alpha*(reward + self.gamma * np.max(next_action_value) - table[interaction['loc_difference']][action])}")
                    #print(interaction['loc_difference'])
                    table[interaction['loc_difference']][action] = table[interaction['loc_difference']][action] + self.alpha*(reward + self.gamma * np.max(next_action_value) - table[interaction['loc_difference']][action])
                    #print(f'Q-value after update {table[interaction["loc_difference"]][action]}')

    def _total_rewards(self, interactions):
        action_rewards = np.zeros(self.action_size)
        for interaction in interactions:
            type_1, type_2 = interaction['types_before']
            table = self.tables.setdefault(type_1, {}).setdefault(type_2, self._make_table())
            action_rewards += table[interaction['loc_difference']]  # add q-value arrays
        return action_rewards

    def _make_table(self):
        '''
        Makes table for q-learning
        3-D table: rows = loc_difference_x, cols = loc_difference_y, z = q-values for actions
        Rows and cols added to as needed
        '''
        return np.zeros((self.neighbor_radius * 8, self.neighbor_radius * 8, self.action_size),
                        dtype=float)

    def save(self, filename):
        '''Save agent's tables'''
        with open(filename, 'wb') as f_p:
            pickle.dump(self.tables, f_p)

    @staticmethod
    def from_saved(filename, action_size):
        '''Load agent from filename'''
        with open(filename, 'rb') as f_p:
            tables = pickle.load(f_p)
            ret = TabularAgent(action_size)
            assert len(tables.values()[0].values()[0][0, 0]) == action_size, \
                   'Action size given to from_saved doesn\'t match the one in the tables'
            ret.tables = tables
        return ret
