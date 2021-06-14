import argparse
import os
from collections import deque
from datetime import datetime
import numpy as np
import tensorflow as tf
import tqdm
import cross_circle_gym
from agents import DQNAgent, SymbolicAgent, SymbolicAgentv2, SymbolicAgentDQN, SymbolicAgentDQNv2, SymbolicAgentExact
from utils import make_autoencoder_train_data
import gym

parser = argparse.ArgumentParser(description=None)

# Experiment variables
parser.add_argument('--episodes', '-e', type=int, default=10_000,
					help='number of DQN training episodes')
parser.add_argument('--evaluation_frequency', type=int, default=1_000,
					help='How often to evaluate the agent')
parser.add_argument('--agent', type = str, default = 'symbdqn', 
					help='What agent do you want to evaluate (dqn or symb)')
# Saving and logging config
parser.add_argument('--experiment_name', type=str, default='default', help='Name of the experiment')
parser.add_argument('--logdir',type=str,default='./logs', help='Log directory')
parser.add_argument('--name',type=str,default='exp', help='name')

args = parser.parse_args()

args.logdir = os.path.join(args.logdir,args.experiment_name,args.agent, args.name)

env = gym.make("CrossCircle-MixedGrid-v0")

action_size = env.action_space.n

symbv2flag = args.agent =='symbv2' or 'symbdqn' in args.agent
state_dim = None
if False:
	state_dim = env.reset().shape


agent = None
if args.agent=='dqn':
	agent = DQNAgent(state_dim, action_size)
elif args.agent=='symb':
	agent = SymbolicAgent(state_dim, action_size, make_autoencoder_train_data(5000))
elif args.agent=='symbv2':
	agent = SymbolicAgentv2(action_size)
elif args.agent=='symbdqn':
	agent = SymbolicAgentDQN(action_size)
elif args.agent=='symbdqnv2':
	agent = SymbolicAgentDQNv2(action_size)
elif args.agent== 'symbexact':
	agent = SymbolicAgentExact(action_size)
else:
	raise Exception('agent type not found')


number_of_evaluations = 0
time_steps = 100
buffered_rewards = deque(maxlen=200)
summary_writer = tf.summary.create_file_writer(args.logdir)

for e in tqdm.tqdm(range(args.episodes)):
	#state_builder.restart()
	state = env.reset()
	agent.reset()
	#state = state_builder.build_state(*autoencoder.get_entities(state))
	total_reward = 0

	for t in range(time_steps):
		
		#env.render()
		action = agent.act(state)
		#print(action)
		next_state, reward, done, _ = env.step(action)
		total_reward += reward

		agent.update(state, action, reward, next_state, done)

		state = next_state
		if done:
			break

	env.close()
	buffered_rewards.append(total_reward)

	with summary_writer.as_default():
		tf.summary.scalar('Averaged Reward',np.mean(buffered_rewards),e)
		tf.summary.scalar('Epsilon',agent.epsilon,e)

	if e%args.evaluation_frequency ==0:
		agent.save(os.path.join(args.logdir,'agent'))
	"""
	if e % args.evaluation_frequency == 0:
		number_of_evaluations += 1
		agent.save(os.path.join(args.logdir,'dqn_agent.h5'))
		evaluation_reward = []
		with summary_writer.as_default():
			for i in range(1):
				done = False
				#state_builder.restart()
				image = env.reset()
				agent.reset()
				#state = state_builder.build_state(*autoencoder.get_entities(image))
				total_reward = 0
				for t in range(time_steps):
					
					action = agent.act(image,random_act=False)
					#env.render()
					next_image, reward, done, _ = env.step(action)
					if i==0:
						tf.summary.image(f'Agent Behaviour {number_of_evaluations}',np.reshape(image,(1,)+image.shape),t)
						
					total_reward += reward
					#next_state = state_builder.build_state(*autoencoder.get_entities(next_image))
					#state = next_image
					image = next_image
				evaluation_reward.append(total_reward)

				env.close()

			tf.summary.scalar('Evaluation Reward',np.mean(evaluation_reward),number_of_evaluations)

	"""