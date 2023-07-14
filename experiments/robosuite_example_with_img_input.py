# %%
import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper

import numpy as np
import matplotlib.pyplot as plt

# %% Camera observation
env = suite.make(
    env_name                = 'Lift',
    robots                  = 'Panda',
    has_renderer            = True,
    has_offscreen_renderer  = True,
    use_camera_obs          = True,
    render_camera           = 'agentview', # ['frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand'] for env.render()
    reward_shaping          = True,
    camera_names            = ['agentview'], # ['frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand'] for use_camera_obs=True
)

# %%
# 42 for use_camera_obs = False
# 393258 for ['frontview', 'agentview']
# 196650 for ['frontview'] or ['agentview']

# 393258 - 196650 = 196650 - 42 = 196608
# Each image frame has 196608 frames. Just not sure how it's organized

# 196608 / 3 = 65536
# Total of 65536 pixels, assuming RGB channel

# 256 * 256 = 65536

# %%
env = GymWrapper(env)

obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Replace with your own action selection logic
    obs, reward, done, info = env.step(action)

    print(obs.shape)
    print('obs: ', obs)
    print('reward: ', reward)

    # Render camera observation
    img = obs[-256*256*3:] + 1
    img = np.reshape(img, (256, 256, 3)).astype(np.uint8)
    plt.imshow(img, origin='lower')
    plt.savefig('img_agentcamera.png'); print('Saved camera data to image file. Exiting...')
    break

    env.render()  # Remove this line if has_renderer=False
