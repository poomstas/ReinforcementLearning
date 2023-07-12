# %%
import os
import gym
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3 import SAC

# %%
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

env = gym.make('Pendulum-v1', render_mode='human')

# %%
model = SAC(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=50000, log_interval=10)
model.save('sac_pendulum')

del model # remove to demonstrate saving and loading

# %%
model = SAC.load('sac_pendulum')

obs = env.reset()[0]

while True:
    action = model.predict(obs)
    obs, rewards, dones, info, _ = env.step(action[0])
    print(rewards)
    env.render()
