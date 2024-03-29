# %%
import time
import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
import numpy as np
import argparse

from datetime import datetime
from sac_torch import Agent

import torch
from torch.utils.tensorboard import SummaryWriter

# %%
def parse_arguments(parser):
    parser.add_argument('--alpha',          type=float, default=0.001,  help='Learning Rate for the Actor (float)')
    parser.add_argument('--beta',           type=float, default=0.001,  help='Learning Rate for the Critic (float')
    parser.add_argument('--tau',            type=float, default=0.005,  help='Controls soft updating the target network')
    parser.add_argument('--reward_scale',   type=int,   default=1,      help='Reward scaling factor')
    parser.add_argument('--batch_size',     type=int,   default=100,    help='Batch Size for Actor & Critic training')
    parser.add_argument('--layer1_size',    type=int,   default=256,    help='Layer 1 size (same for actor & critic)')
    parser.add_argument('--layer2_size',    type=int,   default=256,    help='Layer 2 size (same for actor & critic)')
    parser.add_argument('--n_games',        type=int,   default=10000,  help='Total number of episodes')
    parser.add_argument('--patience',       type=int,   default=500,    help='Patience for plateau checking')
    parser.add_argument('--cuda_index',     type=int,   default=0,      help='GPU Index, default at 0')
    parser.add_argument('--TB_note',        type=str,   default='',     help='Note on TensorBoard')

    args = parser.parse_args()

    return args

# %%
def get_writer_name(args):
    writer_name = \
        "SAC_PandaLift_alpha_{}_beta_{}_tau_{}_RewScale_{}_batchsize_{}_layer1size_{}_layer2size_{}_nGames_{}_patience_{}_{}".format(
            args.alpha, args.beta, args.tau, args.reward_scale, args.batch_size,
            args.layer1_size, args.layer2_size, args.n_games, args.patience, 
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )

    if args.TB_note != "":
        writer_name += "_" + args.TB_note

    writer_name = './TB/' + writer_name
    print('TensorBoard Name: {}'.format(writer_name))

    return writer_name

# %%
def add_hparams_to_writer(writer, args, best_reward, last_100_avg_reward):
    parameter_dict = {}
    for hyperparameter in dir(args):
        if hyperparameter != 'TB_note' and not hyperparameter.startswith('_'): 
            parameter_dict[hyperparameter] = getattr(args, hyperparameter)

    metric_dict = {'best_reward':best_reward, 'last_100_avg_reward':last_100_avg_reward}
    writer.add_hparams(parameter_dict, metric_dict)
    writer.add_scalar('best_reward', best_reward, 0) # To include it in the same group. Bug in PyTorch + TensorBoard; see: https://stackoverflow.com/questions/63830848/hparams-in-tensorboard-run-ids-and-naming

    return writer

# %%
def has_plateaued(reward_history, patience=100):
    ''' Simple function that checks for plateau. '''
    single_patience_mean = np.mean(reward_history[-patience:])
    double_patience_mean = np.mean(reward_history[-2*patience:])

    if len(reward_history) < 2*patience:
        return False

    plateau_bool = np.abs((single_patience_mean - double_patience_mean) / single_patience_mean)*100 < 0.1

    return plateau_bool

# %%
def train_SAC(args, writer):
    train_begin_time = time.time()

    env = suite.make(
        env_name                = 'Lift',
        robots                  = 'Panda',
        has_renderer            = False,
        has_offscreen_renderer  = False,
        use_camera_obs          = False,
        render_camera           = None, # ['frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand']
        reward_shaping          = True,
        camera_names            = None, # ['frontview', 'agentview', None, ?]
    )
    env = GymWrapper(env)
    obs = env.reset()

    TB_name = writer.log_dir.split('/')[-1]

    agent = Agent(env=env, input_dims=env.observation_space.shape, n_actions=env.action_space.shape[0],
                  TB_name=TB_name, alpha=args.alpha, beta=args.beta, tau=args.tau, batch_size=args.batch_size,
                  layer1_size=args.layer1_size, layer2_size=args.layer2_size, reward_scale=args.reward_scale, cuda_index=args.cuda_index)

    best_reward = env.reward_range[0]
    reward_history = []
    mean_reward_history = []

    for i in range(args.n_games):
        obs = env.reset()
        done = False
        reward_sum = 0

        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, _ = env.step(action)
            agent.remember(obs, action, reward, obs_, done)
            agent.learn()
            reward_sum += reward
            obs = obs_

        reward_history.append(reward_sum)
        writer.add_scalar('Reward', reward_sum, i)
        avg_reward_100 = np.mean(reward_history[-100:])
        mean_reward_history.append(avg_reward_100)
        writer.add_scalar('last_100_reward_avg', avg_reward_100, i)

        if reward_sum > best_reward:
            best_reward = reward_sum
            agent.save_models()
        
        writer.add_scalar('best_reward_so_far', best_reward, i)

        if i % 20 == 0:
            print('Episode: {:<6s}\tReward: {:<10s}\tLast 100 Episode Avg.: {:<15s}\tTrain Time: {:.1f} sec'.format(
                str(i), str(np.round(reward_sum, 2)), str(np.round(avg_reward_100, 2)), time.time()-train_begin_time))

        if has_plateaued(mean_reward_history, patience=args.patience):
            print("\nReached Plateau; Terminating Simulations.\n")
            print("Writer: {}".format(writer.log_dir))
            break

    return writer, best_reward, avg_reward_100
    
# %%
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Hyperparameters for SAC') # Parse hyperparameter arguments from CLI
    args = parse_arguments(parser) # Reference values like so: args.alpha 

    writer = SummaryWriter(get_writer_name(args))
    writer, best_reward, last_100_avg_reward = train_SAC(args, writer)
    writer = add_hparams_to_writer(writer, args, best_reward, last_100_avg_reward)
