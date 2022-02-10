import gym
import keras
import gym_snake


env = gym.make('snake-v0')
instance = env.reset()
game_controller = env.controller

while True:
    observation, reward, done, info = env.step(env.action_space.sample())
    snake_state = game.controller.get_state(0)

    env.render()