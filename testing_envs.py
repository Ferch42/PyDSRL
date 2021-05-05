import gym 
import cross_circle_gym


env = gym.make("CrossCircle-MixedGrid-v0")

env.reset()

for i in range(10000):
	
	action = int(input())
	observation, reward, done, info = env.step(action)
	print(f"Reward = {reward}")
	env.render()