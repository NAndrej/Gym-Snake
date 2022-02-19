import gym
import keras
import gym_snake
import numpy as np
from deep_q_network import DQN, DQModel, DDQN

env = gym.make('snake-v0')
instance = env.reset()
game_controller = env.controller

state_space = game_controller.get_state()
action_space = env.action_space.shape[0]

main_model = DQModel(state_space, action_space).get_model()
target_model = DQModel(state_space, action_space).get_model()

batch_size = 64
memory_size = 512

dqn_agent = DDQN(state_space.shape,
                    action_space,
                    model=main_model,
                    target_model=target_model,
                    batch_size=batch_size,
                    memory_size=memory_size)

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0
exploration_rate_decay = 0.01

rewards_per_episode = []

for num_episode in range(500):
    env.reset()
    state = game_controller.get_state()

    reward_this_episode = 0
    for step_number in range(100):
        action = dqn_agent.get_action(state, exploration_rate)

        new_state, reward, done, info = env.step([action])
        dqn_agent.update_memory(state, action, reward, new_state, done)

        state = new_state
        reward_this_episode += reward
        if done:
            break

        exploration_rate = min_exploration_rate  + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_rate_decay * num_episode)
        dqn_agent.discount_factor = exploration_rate
        dqn_agent.train()

        print(num_episode, "rewards this episode ", reward_this_episode, ' exploration rate ', exploration_rate)
        rewards_per_episode.append(reward_this_episode)

        env.render()

main_model.save('models/model1.h5')