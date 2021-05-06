import gym 
import cross_circle_gym
from components.agent import  DQNAgent
import numpy as np
from tqdm import tqdm

from matplotlib import pyplot as plt

env = gym.make("CartPole-v1")

s = env.reset()
agent = DQNAgent(s.shape, env.action_space.n)

rewards = []
epsilons = []
total_rewards = 0
for i in tqdm(range(1_000_000)):
	
	a = agent.act(s)
	ns, r, d, info = env.step(a)
	#env.render()
	total_rewards+= r
	agent.update(s,a,r,ns,d)
	s = ns
	epsilons.append(agent.epsilon)
	if i%10_000 == 0:
		plt.title('rewards')
		plt.plot(rewards)
		plt.show()
		plt.title('episilons')
		plt.plot(epsilons)
		plt.show()
	if d:
		rewards.append(total_rewards)
		total_rewards = 0
		s = env.reset()