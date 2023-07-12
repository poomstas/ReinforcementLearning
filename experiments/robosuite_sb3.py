# %%
import os
import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy

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

model = SAC(policy='MlpPolicy', env=env, verbose=1)
model.learn(total_timesteps=100000, log_interval=1)
model.save('PandaLift'); print('Model saved.')
del model

# %%
model = SAC.load('PandaLift', env=env)
episodes = 10
rewards = []

for ep in range(episodes):
    obs = env.reset()
    episode_reward, timestep, done = 0, 0, False

    while not done:
        print('Timestep: ', timestep); timestep += 1
        action = model.predict(obs)
        obs, reward, dones, info = env.step(action[0])
        episode_reward += reward
        env.render()
        # print('reward:', reward)
    rewards.append(episode_reward)
    print('='*90, '\n' 'Episode: {}, Reward: {}'.format(ep, episode_reward), '\n', '='*90)

env.close()
