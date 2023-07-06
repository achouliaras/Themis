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

from logger import Logger
from replay_buffer import ReplayBuffer
from reward_model import RewardModel
from collections import deque

import utils
import hydra
from omegaconf import DictConfig, OmegaConf


class Workspace(object):
    def __init__(self, cfg, work_dir):
        self.work_dir = work_dir
        print(f'Workspace: {self.work_dir}')

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
            cfg.agent.obs_dim = self.env.observation_space.shape[0]
            cfg.agent.action_dim = self.env.action_space.shape[0]
            cfg.agent.action_range = [
                float(self.env.action_space.low.min()),
                float(self.env.action_space.high.max())
            ]

        elif 'ALE' in cfg.domain:
            self.env, self.eval_env, self.sim_env = utils.make_atari_env(cfg, cfg.render_mode)
            self.log_success = True
        elif 'Box2D' in cfg.domain:
            self.env, self.eval_env, self.sim_env = utils.make_box2d_env(cfg, cfg.render_mode)
            self.log_success = True
        elif 'MiniGrid' in cfg.domain or 'BabyAI' in cfg.domain:
            self.env, self.eval_env, self.sim_env = utils.make_minigrid_env(cfg, cfg.render_mode)
            self.log_success = True

            # Setup Agent
            cfg.agent.obs_space = self.env.observation_space       #Hydra doesn't let me pass a dict 
            cfg.agent.obs_dim = self.env.observation_space['image'].shape
            cfg.agent.action_dim = int(self.env.action_space.n)
            cfg.agent.batch_size = 256
            cfg.agent.policy = 'CNN'
            cfg.agent.action_range = [0,1]
        else:
            raise NotImplementedError
        
        self.agent = hydra.utils.instantiate(cfg.agent, _recursive_=False, _convert_="all")
        
        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity), self.device)
        
        # for logging
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0
        self.episode=0
        self.interactions=0
        
        # instantiating the reward model
        self.reward_model = RewardModel(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation, 
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
            self.agent.reset()
            terminated = False
            truncated = False
            episode_reward = 0
            true_episode_reward = 0
            if self.log_success:
                episode_success = 0

            while not (terminated or truncated):
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                
                episode_reward += reward
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, terminated)
                
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
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # unsupervised exploration
            if self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step, gradient_update=1, K=self.cfg.topK)
                
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))

            # allow infinite bootstrap
            terminated = float(terminated)
            episode_reward += reward_hat
            true_episode_reward += reward
            
            if self.log_success:
                episode_success = max(episode_success, terminated)
                
            # adding data to the reward training data
            self.reward_model.add_data(obs, action, reward, terminated, truncated)
            self.replay_buffer.add(obs, action, reward_hat, next_obs, terminated, truncated)

            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1

        # evaluate agent at the end
        self.logger.log('eval/episode', self.episode, self.step)
        self.evaluate()

    def save_snapshot(self):
        snapshot_dir = self.cfg.snapshot_dir        
        snapshot = snapshot_dir / f'snapshot_{self.global_frame}.pt'
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        self.agent.save(snapshot_dir, self.global_frame)
        self.reward_model.save(snapshot_dir, self.global_frame)
        keys_to_save = ['replay_buffer', 'step', 'episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        torch.save(payload, snapshot)
        
@hydra.main(version_base=None, config_path="config", config_name='train_PEBBLE')
def main(cfg : DictConfig):
    work_dir = Path.cwd()
    workspace = Workspace(cfg, work_dir)
    cfg.snapshot_dir = work_dir / cfg.snapshot_dir
    snapshot = cfg.snapshot_dir / f'snapshot_{cfg.num_seed_steps + cfg.num_unsup_steps}.pt'
    if snapshot.exists():
        print(f'Snapshot seems to already exist at {cfg.snapshot_dir}')
        print('Do you want to overwrite it?\n')
        answer = input('[y]/n')
        if answer in ['n','no','No']: exit()
    workspace.run()
    if snapshot.exists():
        print(f'Overwriting models at: {cfg.snapshot_dir}')
    else:
        print(f'Creating models at: {cfg.snapshot_dir}')
    workspace.save_snapshot()

if __name__ == '__main__':
    main()