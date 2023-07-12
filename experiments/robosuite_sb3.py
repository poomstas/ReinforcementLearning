# %%
import os
import robosuite as suite

from datetime import datetime
from robosuite.wrappers.gym_wrapper import GymWrapper
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy, CnnPolicy # TODO Use CNN policy for camera obs

# %% Camera observation
env = suite.make(
    env_name                = 'Lift',
    robots                  = 'Panda',
    has_renderer            = True,
    has_offscreen_renderer  = True,
    use_camera_obs          = False,
    render_camera           = 'frontview', # ['frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand']
    reward_shaping          = True, # Sparse binary reward if False (indicating Success/Fail), more informing if True. True is easier to train.
)
env = GymWrapper(env)

# %%
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

obs = env.reset()
done = False
tb_logger_name = datetime.now().strftime('SAC_%Y-%m-%d_%H-%M')

model = SAC(policy='MlpPolicy', env=env, verbose=1)
model.learn(total_timesteps=3000000, log_interval=1, tb_log_name=tb_logger_name)
model.save('PandaLift'); print('Model saved.')
del model

# %%
model = SAC.load('20230712_PandaLift_3000000', env=env)
episodes = 10
rewards = []

for ep in range(episodes):
    obs = env.reset()
    episode_reward, timestep, done = 0, 0, False

    while not done:
        action = model.predict(obs)
        obs, reward, dones, info = env.step(action[0])
        episode_reward += reward
        env.render()
        print('Timestep: {}\t Reward: {:.5f}\t CumReward: {:.3f}'.format(timestep, reward, episode_reward))
        timestep += 1
    rewards.append(episode_reward)
    print('='*90)
    print('Episode: {}, Reward: {:.5f}'.format(ep, episode_reward))
    print('='*90)

env.close()

# %%
# Last run started 20230712 10:28 AM
# Run completed 14:53 20230712 (4.5 hours)
# total_timesteps = 1000000

# Last run started 20230712 18:00
# Run completed  20230713 07:43 (13.75 hours)
# total_timesteps = 3000000

# %%
# Try using the camera inputs from the model's head camera
# Replace the observation space with the camera observation space
# Replace MlpPolicy with CnnPolicy
