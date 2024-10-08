import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import lib.utils as utils
import hydra

from agent import Agent
from agent.critic import DoubleQCritic
from agent.actor import DiagGaussianActor, CategoricalActor
from stable_baselines3.common.distributions import CategoricalDistribution

def compute_state_entropy(obs, full_obs, k, action_type):
    batch_size = 100
    with torch.no_grad():
        dists = []
        for idx in range(len(full_obs) // batch_size + 1):
            start = idx * batch_size
            end = (idx + 1) * batch_size
            if action_type == 'Cont':
                dist = torch.norm(obs[:, None, :] - full_obs[None, start:end, :], dim=-1, p=2)
            else:
                #print(full_obs[None, start:end, :].shape)
                #print(obs[:, None, :].shape)
                dist = torch.norm(obs[:, None, :] - full_obs[None, start:end, :], dim=(-1,-2,-3), p=2)
            dists.append(dist)

        dists = torch.cat(dists, dim=1)
        knn_dists = torch.kthvalue(dists, k=k + 1, dim=1).values
        state_entropy = knn_dists

    return state_entropy.unsqueeze(1)

class SACAgent(Agent):
    """SAC algorithm."""
    def __init__(self, obs_space, obs_dim, action_dim, action_range, device, critic_cfg,
                 actor_cfg, discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, policy, learnable_temperature, mode=0,
                 normalize_state_entropy=True):
        super().__init__()

        self.obs_space = obs_space
        self.obs_dim = obs_dim
        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature
        self.critic_cfg = critic_cfg
        self.critic_lr = critic_lr
        self.critic_betas = critic_betas
        self.s_ent_stats = utils.TorchRunningMeanStd(shape=[1], device=device)
        self.normalize_state_entropy = normalize_state_entropy
        self.init_temperature = init_temperature
        self.alpha_lr = alpha_lr
        self.alpha_betas = alpha_betas
        self.actor_cfg = actor_cfg
        self.actor_betas = actor_betas
        self.alpha_lr = alpha_lr
        self.policy = policy
        self.action_type = self.actor_cfg.action_type
        self.mode = mode

        #self.critic = hydra.utils.instantiate(critic_cfg, _convert_="all").to(self.device)
        self.critic = self.create_critic()
        #self.critic_target = hydra.utils.instantiate(critic_cfg, _convert_="all").to(self.device)
        self.critic_target = self.create_critic()
        
        self.critic_target.load_state_dict(self.critic.state_dict())

        #self.actor = hydra.utils.instantiate(actor_cfg, _convert_="all").to(self.device)
        self.actor = self.create_actor()
        
        self.log_alpha = torch.tensor(np.log(init_temperature), dtype=torch.float32).to(self.device)
        self.log_alpha.requires_grad = True
        
        # set target entropy to -|A|
        #self.target_entropy = -action_dim
        self.target_entropy =  -np.log((1.0 / action_dim)) * 0.98

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr,
            betas=actor_betas)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_lr,
            betas=critic_betas)
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=alpha_lr,
            betas=alpha_betas)
        
        # change mode
        self.train()
        # self.actor.train()
        # self.critic.train()
        # self.critic_target.train()
    
    def create_critic(self):
        critic = DoubleQCritic(obs_space = self.obs_space, 
            obs_dim= self.obs_dim,
            action_dim= self.critic_cfg.action_dim,
            action_type= self.critic_cfg.action_type,
            policy= self.critic_cfg.policy,
            hidden_dim= self.critic_cfg.hidden_dim,
            hidden_depth= self.critic_cfg.hidden_depth,
            mode= self.mode).to(self.device)
        return critic
    
    def create_actor(self):
        if self.actor_cfg.action_type == 'Cont':
            #self.actor = hydra.utils.instantiate(actor_cfg, _convert_="all").to(self.device)
            actor = DiagGaussianActor(obs_dim = self.actor_cfg.obs_dim, 
                action_dim = self.actor_cfg.action_dim,
                #action_type = self.actor_cfg.action_type,
                policy = self.actor_cfg.policy,
                hidden_dim = self.actor_cfg.hidden_dim, 
                hidden_depth = self.actor_cfg.hidden_depth,
                log_std_bounds = self.actor_cfg.log_std_bounds).to(self.device)
        elif self.actor_cfg.action_type == 'Discrete':
            actor = CategoricalActor(obs_space = self.obs_space, 
                obs_dim = self.actor_cfg.obs_dim, 
                action_dim = self.actor_cfg.action_dim,
                #action_type = self.actor_cfg.action_type, 
                policy = self.actor_cfg.policy, 
                hidden_dim = self.actor_cfg.hidden_dim, 
                hidden_depth = self.actor_cfg.hidden_depth,
                log_std_bounds = self.actor_cfg.log_std_bounds,
                mode= self.mode).to(self.device)
            self.categorical = CategoricalDistribution(self.actor_cfg.action_dim)
        return actor
    
    def reset_critic(self):
        # The CNN feature shouldn't reset...
        # copy them and paste on top of the reseted critics
        self.critic = self.create_critic()
        self.critic_target = self.create_critic()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            betas=self.critic_betas)
    
    def reset_actor(self):
        # reset log_alpha
        self.log_alpha = torch.tensor(np.log(self.init_temperature), dtype=torch.float32).to(self.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=self.alpha_lr,
            betas=self.alpha_betas)
        
        # reset actor
        #self.actor = hydra.utils.instantiate(self.actor_cfg).to(self.device)
        self.actor = self.create_actor()
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.actor_lr,
            betas=self.actor_betas)
        
    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False, determ=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor.forward(obs)
        if self.action_type == 'Cont':
            # Action is a vector of float numbers
            action = dist.sample() if sample else dist.mean         
            action = action.clamp(*self.action_range)
            assert action.ndim == 2 and action.shape[0] == 1
            return utils.to_np(action[0])
        elif self.action_type == 'Discrete':
            # Action is a flat integer
            dist = self.categorical.proba_distribution(action_logits=dist)
            action = dist.get_actions(deterministic=False)
            return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, 
                      not_done, logger, step, print_flag=True):
        
        output = self.actor.forward(next_obs)
        if self.action_type == 'Cont':
            dist = output
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)

            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
            
            target_Q = reward + (not_done * self.discount * target_V)
            target_Q = target_Q.detach()
        elif self.action_type == 'Discrete':
            action_probs = output
            next_action = self.categorical.actions_from_params(action_logits=output)
            z = action_probs == 0.0
            z = z.float() * 1e-8
            log_prob = torch.log(action_probs + z)
            
            #print(action_probs)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = action_probs * (torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob)
            target_V = target_V.sum(1).unsqueeze(-1)
            target_Q = reward + (not_done * self.discount * target_V)
            target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        current_Q1 = current_Q1.gather(1, action.long())
        current_Q2 = current_Q2.gather(1, action.long())
        
        qf1_loss = F.mse_loss(current_Q1, target_Q)
        qf2_loss = F.mse_loss(current_Q2, target_Q)
        
        critic_loss =  qf1_loss + qf2_loss
        # Use action to take the suitable Q value
        
        if print_flag:
            logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        qf1_loss.backward(retain_graph=True)
        qf2_loss.backward(retain_graph=True)
        self.critic_optimizer.step()
        self.critic.log(logger, step)
        
    def update_critic_state_ent(
        self, obs, full_obs, action, next_obs, not_done, logger,
        step, K=5, print_flag=True):
        
        output = self.actor.forward(next_obs)
        if self.action_type == 'Cont':
            dist = output
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        elif self.action_type == 'Discrete':
            action_probs = output
            next_action = self.categorical.actions_from_params(action_logits=output)
            z = action_probs == 0.0
            z = z.float() * 1e-8
            log_prob = torch.log(action_probs + z)
        
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob

        # compute state entropy
        state_entropy = compute_state_entropy(obs, full_obs, k=K, action_type=self.action_type)
        #print('State entropy = ', state_entropy.shape)
        if print_flag:
            logger.log("train_critic/entropy", state_entropy.mean(), step)
            logger.log("train_critic/entropy_max", state_entropy.max(), step)
            logger.log("train_critic/entropy_min", state_entropy.min(), step)
        
        self.s_ent_stats.update(state_entropy)
        norm_state_entropy = state_entropy / self.s_ent_stats.std
        
        if print_flag:
            logger.log("train_critic/norm_entropy", norm_state_entropy.mean(), step)
            logger.log("train_critic/norm_entropy_max", norm_state_entropy.max(), step)
            logger.log("train_critic/norm_entropy_min", norm_state_entropy.min(), step)
        
        if self.normalize_state_entropy:
            state_entropy = norm_state_entropy
        
        target_Q = state_entropy + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        current_Q1 = current_Q1.gather(1, action.long())
        current_Q2 = current_Q2.gather(1, action.long())
        
        qf1_loss = F.mse_loss(current_Q1, target_Q)
        qf2_loss = F.mse_loss(current_Q2, target_Q)
        critic_loss =  qf1_loss + qf2_loss
        
        if print_flag:
            logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        qf1_loss.backward(retain_graph=True)
        qf2_loss.backward(retain_graph=True)
        self.critic_optimizer.step()
        self.critic.log(logger, step)
    
    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic_target.state_dict(), '%s/critic_target_%s.pt' % (model_dir, step)
        )
        
    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        self.critic_target.load_state_dict(
            torch.load('%s/critic_target_%s.pt' % (model_dir, step))
        )
    
    def update_actor_and_alpha(self, obs, logger, step, print_flag=False):
        output = self.actor.forward(obs)
        if self.action_type == 'Cont':
            dist = output
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)

            actor_Q1, actor_Q2 = self.critic(obs, action)
            actor_Q = torch.min(actor_Q1, actor_Q2)
            actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
        elif self.action_type == 'Discrete':
            action_probs = output
            action = self.categorical.actions_from_params(action_logits=output)
            z = action_probs == 0.0
            z = z.float() * 1e-8
            log_prob = torch.log(action_probs + z)
            
            actor_Q1, actor_Q2 = self.critic(obs, action)
            actor_Q = torch.min(actor_Q1, actor_Q2)
            inside_term = (self.alpha.detach() * log_prob) - actor_Q
            actor_loss = (action_probs*inside_term).sum(dim=1).mean()
            log_prob = torch.sum(log_prob * action_probs, dim=1)        # CHECK AGAIN
            #print('actor_loss', actor_loss)
            
        if print_flag:
            logger.log('train_actor/loss', actor_loss, step)
            logger.log('train_actor/target_entropy', self.target_entropy, step)
            logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = -(self.alpha * 
                          (log_prob + self.target_entropy).detach()).mean()
            if print_flag:
                logger.log('train_alpha/loss', alpha_loss, step)
                logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
            
    def update(self, replay_buffer, logger, step, gradient_update=1):
        for index in range(gradient_update):
            obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
                self.batch_size)

            print_flag = False
            if index == gradient_update -1:
                logger.log('train/batch_reward', reward.mean(), step)
                print_flag = True
                
            self.update_critic(obs, action, reward, next_obs, not_done_no_max,
                               logger, step, print_flag)

            if step % self.actor_update_frequency == 0:
                self.update_actor_and_alpha(obs, logger, step, print_flag)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)
            
    def update_after_reset(self, replay_buffer, logger, step, gradient_update=1, policy_update=True):
        for index in range(gradient_update):
            obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
                self.batch_size)

            print_flag = False
            if index == gradient_update -1:
                logger.log('train/batch_reward', reward.mean(), step)
                print_flag = True
                
            self.update_critic(obs, action, reward, next_obs, not_done_no_max,
                               logger, step, print_flag)

            if index % self.actor_update_frequency == 0 and policy_update:
                self.update_actor_and_alpha(obs, logger, step, print_flag)

            if index % self.critic_target_update_frequency == 0:
                utils.soft_update_params(self.critic, self.critic_target,
                                         self.critic_tau)
            
    def update_state_ent(self, replay_buffer, logger, step, gradient_update=1, K=5):
        for index in range(gradient_update):
            obs, full_obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample_state_ent(self.batch_size)

            print_flag = False
            if index == gradient_update -1:
                logger.log('train/batch_reward', reward.mean(), step)
                print_flag = True
                
            self.update_critic_state_ent(
                obs, full_obs, action, next_obs, not_done_no_max,
                logger, step, K=K, print_flag=print_flag)

            #print('Here 2')
            if step % self.actor_update_frequency == 0:
                self.update_actor_and_alpha(obs, logger, step, print_flag)

        #print('Here 3')
        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)