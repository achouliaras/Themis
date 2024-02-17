#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
from pathlib import Path
import time
import pickle as pkl
import tqdm
import copy

from lib.logger import Logger
from agent.sac import SACAgent
from lib.replay_buffer import ReplayBuffer
from lib.reward_model import RewardModel
from collections import deque

import lib.utils as utils
import hydra
from omegaconf import DictConfig
from gymnasium.spaces import utils as gym_utils

class Workspace(object):
    def __init__(self, cfg, work_dir):
        self.work_dir = work_dir
        print(f'Workspace: {self.work_dir}')

        folder = work_dir / cfg.checkpoints_dir    
        folder.mkdir(exist_ok=True, parents=True)
        self.checkpoints_dir = cfg.checkpoints_dir

        self.cfg = cfg
        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.algorithm.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.log_success = False
        
        # make envs
        if 'Control' in cfg.domain:
            self.env, self.eval_env, self.sim_env = utils.make_control_env(cfg, cfg.render_mode)
            self.log_success = True

            # Setup Agent
            self.action_type = 'Cont'
            self.policy = 'MLP'
            self.mode = 0
            self.obs_space = self.env.observation_space
            action_space = self.env.action_space.shape
            cfg.agent.obs_dim = self.env.observation_space.shape[0]
            cfg.agent.action_dim = self.env.action_space.shape[0]
            cfg.agent.action_range = [
                float(self.env.action_space.low.min()),
                float(self.env.action_space.high.max())
            ]
            #cfg.agent.actor_cfg = '${diag_gaussian_actor}' find another way to do this
            critic_cfg = cfg.double_q_critic,
            actor_cfg = cfg.diag_gaussian_actor,
        
            critic_cfg[0].action_type = self.action_type
            critic_cfg[0].policy = self.policy
            
            actor_cfg[0].action_type = self.action_type
            actor_cfg[0].policy = self.policy
        elif 'ALE' in cfg.domain:
            self.env, self.eval_env, self.sim_env = utils.make_atari_env(cfg, cfg.render_mode)
            self.log_success = True

            self.action_type = 'Discrete'
            self.state_type = 'pixels'
            self.policy = 'CNN'
            self.mode = 0
            self.obs_space = self.env.observation_space 
            self.obs_space_shape = self.env.observation_space.shape
            
            action_space = [1]
            cfg.agent.obs_dim = self.obs_space_shape
            
            cfg.agent.action_dim = int(self.env.action_space.n)
            cfg.agent.batch_size = 256
            cfg.agent.action_range = [0,1]
            critic_cfg = cfg.double_q_critic
            actor_cfg = cfg.categorical_actor
            
            critic_cfg.action_type = self.action_type
            critic_cfg.policy = self.policy

            actor_cfg.action_type = self.action_type
            actor_cfg.policy = self.policy
        elif 'Box2D' in cfg.domain:
            self.env, self.eval_env, self.sim_env = utils.make_box2d_env(cfg, cfg.render_mode)
            self.log_success = True
        elif 'MiniGrid' in cfg.domain or 'BabyAI' in cfg.domain:
            self.env, self.eval_env, self.sim_env = utils.make_minigrid_env(cfg, cfg.render_mode)
            self.log_success = True

            self.action_type = 'Discrete'
            self.state_type = 'pixel-grid'
            self.policy = 'CNN'
            self.mode = 0
            self.obs_space = self.env.observation_space 
            sp = list(self.env.observation_space.shape) # Reorder first 2 dimensions to match state shape
            self.obs_space_shape = sp[1], sp[0], sp[2]
            
            action_space = [1]
            cfg.agent.obs_dim = self.obs_space_shape
            cfg.agent.action_dim = int(self.env.action_space.n)
            cfg.agent.batch_size = 256
            cfg.agent.action_range = [0,1]
            critic_cfg = cfg.double_q_critic
            actor_cfg = cfg.categorical_actor
            
            critic_cfg.action_type = self.action_type
            critic_cfg.policy = self.policy

            actor_cfg.action_type = self.action_type
            actor_cfg.policy = self.policy
        else:
            raise NotImplementedError
        
        #self.agent = hydra.utils.instantiate(cfg.agent, _recursive_=False, _convert_="all")
        self.agent = SACAgent(obs_space = self.obs_space,
            obs_dim = cfg.agent.obs_dim, 
            action_dim = cfg.agent.action_dim, 
            action_range = cfg.agent.action_range, 
            device = cfg.agent.device, 
            critic_cfg = critic_cfg,
            actor_cfg = actor_cfg, 
            discount = cfg.agent.discount, 
            init_temperature = cfg.agent.init_temperature, 
            alpha_lr = cfg.agent.alpha_lr, 
            alpha_betas = cfg.agent.alpha_betas,
            actor_lr = cfg.agent.actor_lr, 
            actor_betas = cfg.agent.actor_betas, 
            actor_update_frequency = cfg.agent.actor_update_frequency, 
            critic_lr = cfg.agent.critic_lr,
            critic_betas = cfg.agent.critic_betas, 
            critic_tau = cfg.agent.critic_tau, 
            critic_target_update_frequency = cfg.agent.critic_target_update_frequency,
            batch_size =cfg.agent.batch_size,
            policy = self.policy,
            mode= self.mode, 
            learnable_temperature = cfg.agent.learnable_temperature,
            normalize_state_entropy = True)

        self.replay_buffer = ReplayBuffer(
            self.obs_space,
            self.obs_space_shape,
            action_space,
            self.action_type,
            int(cfg.replay_buffer_capacity), 
            self.device)
        
        # for logging
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0
        self.episode=0
        self.interactions=0
        
        # instantiating the reward model
        self.reward_model = RewardModel(
            obs_space=self.obs_space,
            ds=gym_utils.flatdim(self.obs_space),
            da=action_space[0],
            action_type=self.action_type,
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation,
            capacity=cfg.reward_model_capacity, 
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch, 
            large_batch=cfg.large_batch, 
            label_margin=cfg.label_margin, 
            teacher_beta=cfg.teacher_beta, 
            teacher_gamma=cfg.teacher_gamma, 
            teacher_eps_mistake=cfg.teacher_eps_mistake, 
            teacher_eps_skip=cfg.teacher_eps_skip, 
            teacher_eps_equal=cfg.teacher_eps_equal)
        
        print('INIT COMPLETE')
        
    @property
    def global_step(self):
        return self.step

    @property
    def global_episode(self):
        return self.episode

    @property
    def global_frame(self):
        return self.step * self.cfg.action_repeat

    def evaluate(self):
        average_episode_reward = 0
        average_true_episode_reward = 0
        success_rate = 0
        
        for episode in range(self.cfg.num_eval_episodes):
            obs, info = self.eval_env.reset(seed = self.cfg.seed)
            if self.action_type == 'Discrete' and self.state_type == 'grid':
                obs = obs['image']
            self.agent.reset()
            terminated = False
            truncated = False
            episode_reward = 0
            true_episode_reward = 0
            if self.log_success:
                episode_success = 0

            while not (terminated or truncated):
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False, determ=False) # set determ=True in experiments
                    #print(action)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                
                episode_reward += reward
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, terminated)
                if self.action_type == 'Discrete' and self.state_type == 'grid':
                    obs = obs['image']
                
            average_episode_reward += episode_reward
            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success
            
        average_episode_reward /= self.cfg.num_eval_episodes
        average_true_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0
        
        self.logger.log('eval/episode_reward', average_episode_reward, self.step)
        self.logger.log('eval/true_episode_reward', average_true_episode_reward, self.step)
        if self.log_success:
            self.logger.log('eval/success_rate', success_rate, self.step)
            self.logger.log('train/true_episode_success', success_rate, self.step)
        self.logger.dump(self.step)
    
    def run(self):
        self.episode, episode_reward, terminated, truncated = 0, 0, True, False
        if self.log_success:
            episode_success = 0
        true_episode_reward = 0
        
        # store train returns of recent 10 episodes
        avg_train_true_return = deque([], maxlen=10) 
        total_time=0
        start_time = time.time()

        interact_count = 0
        while self.step != (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
            if terminated or truncated:
                if self.step > 0:
                    episode_time = time.time() - start_time
                    self.logger.log('train/duration', episode_time, self.step)
                    total_time += episode_time
                    self.logger.log('train/total_duration', total_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))
                
                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)
                
                if self.log_success:
                    self.logger.log('train/episode_success', episode_success, self.step)
                    self.logger.log('train/true_episode_success', episode_success, self.step)
                
                obs, info = self.env.reset(seed = self.cfg.seed)

                if self.action_type == 'Discrete' and self.state_type == 'grid':
                    obs = obs['image']

                self.agent.reset()
                terminated = False
                truncated = False
                episode_reward = 0
                avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                self.episode += 1

                self.logger.log('train/episode', self.episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                if self.action_type == 'Discrete':
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.act(obs, sample=True, determ=False)
            else:
                #with utils.eval_mode(self.agent):
                action = self.agent.act(obs, sample=False, determ=False) # set determ=True in experiments

            # unsupervised exploration
            if self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step, gradient_update=1, K=self.cfg.topK)
                #print('OK')
                
            
            # For Video generation
            if self.state_type == 'grid' or self.state_type == 'tabular':
                env_snapshot = [] # Not yet supported
            elif self.state_type == 'grid':
                env_snapshot = [] # Not yet supported
            elif self.state_type == 'pixels':
                env_snapshot = self.env.get_state()
            else:
                env_snapshot = [] # Not yet supported

            next_obs, reward, terminated, truncated, info = self.env.step(action)

            if self.action_type == 'Discrete':
                next_obs = next_obs['image'] if self.state_type == 'grid' else next_obs
                action = np.array([action], dtype=np.uint8)

            obs_flat = gym_utils.flatten(self.obs_space, obs)

            reward_hat = self.reward_model.r_hat(np.concatenate([obs_flat, action], axis=-1))

            # allow infinite bootstrap
            terminated = float(terminated)
            episode_reward += reward_hat
            true_episode_reward += reward
            
            if self.log_success:
                episode_success = max(episode_success, terminated)
                
            # adding data to the reward training data
            self.reward_model.add_data(obs_flat, action, reward, terminated, truncated, env_snapshot)
            self.replay_buffer.add(obs, action, reward_hat, next_obs, terminated, truncated)

            # Save model checkpoint for State Explanation
            if self.cfg.xplain_state == True and self.step % self.cfg.checkpoint_frec == 0:
                checkpoint_name = "-".join(["checkpoint", str(self.step) + ".pt"])
                torch.save(
                    {
                        "epoch": self.step,
                        "model_state_dict": self.agent.actor.state_dict(),
                        "optimizer_state_dict": self.agent.actor_optimizer.state_dict()
                    },
                    os.path.join(self.checkpoints_dir, checkpoint_name),
                )
            
            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1
            

        # evaluate agent at the end
        self.logger.log('eval/episode', self.episode, self.step)
        self.evaluate()

    def save_snapshot(self):
        snapshot_dir = self.cfg.snapshot_dir        
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        self.agent.save(snapshot_dir, self.global_frame)
        self.replay_buffer.save(snapshot_dir, self.global_frame)
        self.reward_model.save(snapshot_dir, self.global_frame)
        snapshot = snapshot_dir / f'snapshot_{self.global_frame}.pt'
        keys_to_save = ['step', 'episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        torch.save(payload, snapshot, pickle_protocol=4)
        
@hydra.main(version_base=None, config_path="config", config_name='train_themis')
def main(cfg : DictConfig):
    work_dir = Path.cwd()
    workspace = Workspace(cfg, work_dir)
    cfg.snapshot_dir = work_dir / cfg.snapshot_dir
    snapshot = cfg.snapshot_dir / f'snapshot_{cfg.num_seed_steps + cfg.num_unsup_steps}.pt'
    if snapshot.exists():
        print(f'Snapshot seems to already exist at {cfg.snapshot_dir}')
        print('Do you want to overwrite it?\n')
        answer = input('[y]/n \n')
        if answer in ['n','no','No']: exit()
    workspace.run()
    if snapshot.exists():
        print(f'Overwriting models at: {cfg.snapshot_dir}')
    else:
        print(f'Creating models at: {cfg.snapshot_dir}')
    workspace.save_snapshot()

if __name__ == '__main__':
    main()