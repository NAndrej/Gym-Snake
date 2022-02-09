import gym
import keras
import gym_snake


env = gym.make('snake-v0')
instance = env.reset()

while True:
    observation, reward, done, info = env.step(env.action_space.sample())
    print(observation)
    env.render()