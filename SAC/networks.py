# %%
import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

# %%
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, TB_name, 
                 fc1_dims=256, fc2_dims=256, name='critic', chkpt_dir='./TB/', cuda_index=0):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, TB_name+'_'+name)

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:{}'.format(str(cuda_index)) if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, model_fullpath, gpu_indx=0):
        self.load_state_dict(T.load(model_fullpath, map_location='cuda:{}'.format(str(gpu_indx))))

# %%
class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, TB_name, fc1_dims=256, fc2_dims=256, name='value', chkpt_dir='./TB/', cuda_index=0):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, TB_name+'_'+name)

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:{}'.format(str(cuda_index)) if T.cuda.is_available() else 'cpu')

        self.to(self.device)
    
    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)
        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, model_fullpath, gpu_indx=0):
        self.load_state_dict(T.load(model_fullpath, map_location='cuda:{}'.format(str(gpu_indx))))

# %%
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, TB_name,
                 fc1_dims=256, fc2_dims=256, n_actions=2, name='actor', chkpt_dir='./TB/', cuda_index=0):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, TB_name+'_'+name)
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:{}'.format(str(cuda_index)) if T.cuda.is_available() else 'cpu')

        self.to(self.device)
    
    def forward(self, state):
        prob = self.fc1(state.float())
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma
    
    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample() # sample with noise
        else:
            actions = probabilities.sample()
        
        action = T.tanh(actions)*T.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions) # for calculation of loss function. not used for calculating which action to take
        log_probs -= T.log(1-action.pow(2) + self.reparam_noise) # From paper's appendix
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, model_fullpath, gpu_indx=0):
        self.load_state_dict(T.load(model_fullpath, map_location='cuda:{}'.format(str(gpu_indx))))