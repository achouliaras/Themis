import numpy as np
import torch
import torch.nn.functional as F
import lib.utils as utils

from torch import nn

class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_space, obs_dim, action_dim, action_type, policy, hidden_dim, hidden_depth, mode = 0):
        super().__init__()
        self.policy = policy
        self.action_type = action_type

        # If obs is image-like use feature extraction
        if self.policy =='CNN':
            self.cnn, self.flatten = utils.cnn(obs_space, obs_dim[2], mode = mode)
            obs_dim = self.flatten

        if self.action_type == 'Cont':
            self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
            self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        elif self.action_type == 'Discrete':    # Calculate Q-value for every Action on discrete action spaces
            self.Q1 = utils.mlp(obs_dim, hidden_dim, action_dim, hidden_depth)
            self.Q2 = utils.mlp(obs_dim, hidden_dim, action_dim, hidden_depth)

        self.outputs = dict()
        #self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        if self.policy =='CNN':
            obs = self.cnn(obs.permute(0, 3, 1, 2))
        
        if self.action_type == 'Cont':
            input_data = torch.cat([obs, action], dim=-1)       # Add Action on continuous action spaces 
        elif self.action_type == 'Discrete':
            input_data = obs

        q1 = self.Q1(input_data)
        q2 = self.Q2(input_data)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)