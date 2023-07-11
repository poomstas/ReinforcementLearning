# %%
import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
from stable_baselines3 import SAC

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

# %%
env = GymWrapper(env)

obs = env.reset()
done = False

model = SAC('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000, log_interval=1)

# %%
episodes = 10
for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render() 
env.close()

# %%
# while not done:
#     action = env.action_space.sample()  # Replace with your own action selection logic
#     obs, reward, done, info = env.step(action)

#     print(obs.shape)
#     print('obs: ', obs)
#     print('reward: ', reward)

#     env.render()  # Remove this line if has_renderer=False

# env.close()  # Close the environment