# %%
import gym
import numpy as np
from collections import defaultdict

# %%
env = gym.make('Blackjack-v0')

# %%
print(env.observation_space)
print(env.action_space)

# %%
for episode in range(3):
    print('\n', '='*45, ' Episode : {}'.format(str(episode)), '='*45)
    state = env.reset()
    done = False

    while not done:
        print('Current Sum: {},\tDealers Face Up Card: {},\tPlayer Has Useable Ace: {}'.format(str(state[0]), str(state[1]), bool(state[2])))
        action = env.action_space.sample()
        print('Selected action: ', 'Hit' if action==1 else 'Stick')
        state, reward, done, _ = env.step(action)
        # print('Current Sum: {},\tDealers Face Up Card: {},\tPlayer Has Useable Ace: {}'.format(str(state[0]), str(state[1]), bool(state[2])))
    print('Game Ended. Reward: ', int(reward))
    print('You Won') if reward > 0 else print('You Lost')
print('\n', '='*100)

# %%
