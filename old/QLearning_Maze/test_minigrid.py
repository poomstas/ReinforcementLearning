# %%
import gym
import matplotlib.pyplot as plt
from gym_minigrid.wrappers import * 		# pip install gym-minigrid

env = gym.make('MiniGrid-FourRooms-v0')
env.reset()

before_img = env.render('rgb_array')
action = env.actions.forward
obs, reward, done, info = env.step(action)
after_img = env.render('rgb_array')

plt.imshow(np.concatenate([before_img, after_img], 1))
plt.show()

# %%