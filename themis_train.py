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

import logging
import lib.utils as utils
import hydra
from omegaconf import DictConfig, OmegaConf
from gymnasium.spaces import utils as gym_utils
from lib.human_interface import Xplain, ReplayDataset

log = logging.getLogger()

class Workspace(object):
    def __init__(self, cfg, work_dir):
        self.work_dir = work_dir
        print(f'Workspace: {self.work_dir}')

        snapshot_dir = cfg.snapshot_dir
        snapshot = snapshot_dir / f'snapshot_{(cfg.num_seed_steps + cfg.num_unsup_steps)*cfg.action_repeat}.pt'
        
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
            self.state_type = 'tabular'
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
        
        payload = torch.load(snapshot)
        keys_to_load = ['replay_buffer', 'step', 'episode']
        self.replay_buffer, self.step, self.episode = [payload[k] for k in keys_to_load]

        # Setup Agent
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
        
        self.agent.load(snapshot_dir, self.global_frame)
        
        ui_module= Xplain(self.agent, 
                          self.action_type, 
                          xplain_action = cfg.xplain_action, 
                          xplain_state = cfg.xplain_state,
                          checkpoints_dir = self.checkpoints_dir,
                          replay_buffer = self.replay_buffer,
                          debug = cfg.debug)

        # for logging
        self.start_step=self.step
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.interactions=0
        
        if cfg.learn_reward == True:
            # instantiating the reward model
            self.reward_model = RewardModel(
                obs_space=self.obs_space,
                ds=gym_utils.flatdim(self.obs_space),
                da=action_space[0],
                action_type=self.action_type,
                ensemble_size=cfg.ensemble_size,
                size_segment=cfg.segment,
                env = self.sim_env,
                activation=cfg.activation, 
                capacity=cfg.reward_model_capacity,
                lr=cfg.reward_lr,
                mb_size=cfg.reward_batch, 
                large_batch=cfg.large_batch, 
                label_margin=cfg.label_margin,
                reward_scale=cfg.reward_scale, 
                reward_intercept=cfg.reward_intercept,
                human_teacher = cfg.human_teacher, 
                teacher_beta=cfg.teacher_beta, 
                teacher_gamma=cfg.teacher_gamma, 
                teacher_eps_mistake=cfg.teacher_eps_mistake, 
                teacher_eps_skip=cfg.teacher_eps_skip, 
                teacher_eps_equal=cfg.teacher_eps_equal,
                ui_module=ui_module)
        
            self.reward_model.load(snapshot_dir, self.global_frame)
        
        print('INIT COMPLETE')
        print('Models Restored')
    
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
            # Add if minigrid has grid state
            # if self.action_type == 'Discrete' and  self.state_type == 'grid': 
            #     obs = obs['image']
            self.agent.reset()
            terminated = False
            truncated = False
            episode_reward = 0
            true_episode_reward = 0
            if self.log_success:
                episode_success = 0

            while not (terminated or truncated):
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False, determ=False)

                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                
                episode_reward += reward
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, terminated)
                # Add if minigrid has grid state
                # if self.action_type == 'Discrete' and  self.state_type == 'grid': 
                #     obs = obs['image']
                
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
    
    def learn_reward(self, first_flag=False):    
        # get feedbacks
        labeled_queries, noisy_queries = 0, 0
        if first_flag == True:
            # if it is first time to get feedback, need to use random sampling
            labeled_queries = self.reward_model.uniform_sampling(first_flag=True)
        else:
            if self.cfg.feed_type == 0:
                labeled_queries = self.reward_model.uniform_sampling()
            elif self.cfg.feed_type == 1:
                labeled_queries = self.reward_model.disagreement_sampling()
            elif self.cfg.feed_type == 2:
                labeled_queries = self.reward_model.entropy_sampling()
            elif self.cfg.feed_type == 3:
                labeled_queries = self.reward_model.kcenter_sampling()
            elif self.cfg.feed_type == 4:
                labeled_queries = self.reward_model.kcenter_disagree_sampling()
            elif self.cfg.feed_type == 5:
                labeled_queries = self.reward_model.kcenter_entropy_sampling()
            else:
                raise NotImplementedError
        
        if labeled_queries == self.reward_model.mb_size:
            self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries
        self.interactions+=1
        print(f'Feedback No {self.interactions}: {self.total_feedback}/{self.cfg.max_feedback}')

        train_acc = 0
        if self.labeled_feedback > 0:
            # update reward
            for epoch in range(self.cfg.reward_update):
                if self.cfg.label_margin > 0 or self.cfg.teacher_eps_equal > 0 or self.cfg.human_teacher == True:
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    train_acc = self.reward_model.train_reward()
                total_acc = np.mean(train_acc)
                
                if total_acc > 0.97:
                    break
                    
            print("Reward function is updated!! ACC: " + str(total_acc))
        return labeled_queries

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
        while self.step < self.cfg.num_train_steps:
            if terminated or truncated:
                if self.step > self.start_step:
                    episode_time = time.time() - start_time
                    self.logger.log('train/duration', episode_time, self.step)
                    total_time += episode_time
                    self.logger.log('train/total_duration', total_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > self.start_step and self.episode % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', self.episode, self.step)
                    self.evaluate()
                
                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)
                
                if self.log_success:
                    self.logger.log('train/episode_success', episode_success, self.step)
                    self.logger.log('train/true_episode_success', episode_success, self.step)
                
                obs, info = self.env.reset(seed = self.cfg.seed)

                #print(obs[0,0,0].dtype)
                # Add if minigrid has grid state
                # if self.action_type == 'Discrete' and  self.state_type == 'grid': 
                #     obs = obs['image']

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
                    # Action is a vector of floats
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.act(obs, sample=True, determ=False)
            else:
                with utils.eval_mode(self.agent):
                    # Action is a vector of flat integer
                    action = self.agent.act(obs, sample=False, determ=False) # Sample from the action distribution
                #print(action)

            # run training update (until the end)
            if self.step > (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
                # Check if reward model is used
                if self.cfg.learn_reward == True:
                    
                    # update reward function
                    if self.total_feedback < self.cfg.max_feedback:
                        if interact_count == self.cfg.num_interact:
                            # update schedule
                            if self.cfg.reward_schedule == 1:
                                frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                                if frac == 0:
                                    frac = 0.01
                            elif self.cfg.reward_schedule == 2:
                                frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                            else:
                                frac = 1
                            self.reward_model.change_batch(frac)
                            
                            # update margin --> not necessary / will be updated soon
                            new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                            self.reward_model.set_teacher_thres_skip(new_margin * self.cfg.teacher_eps_skip)
                            self.reward_model.set_teacher_thres_equal(new_margin * self.cfg.teacher_eps_equal)
                            
                            # corner case: new total feed > max feed
                            if self.reward_model.mb_size + self.total_feedback > self.cfg.max_feedback:
                                self.reward_model.set_batch(self.cfg.max_feedback - self.total_feedback)
                                
                            self.learn_reward()
                            self.replay_buffer.relabel_with_predictor(self.reward_model)
                            interact_count = 0
                        
                self.agent.update(self.replay_buffer, self.logger, self.step, 1)

            #run training update (at the end of the unsupervised phase)
            elif self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
                # Check if reward model is used
                if self.cfg.learn_reward == True:

                    # update schedule
                    if self.cfg.reward_schedule == 1:
                        frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                        if frac == 0:
                            frac = 0.01
                    elif self.cfg.reward_schedule == 2:
                        frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                    else:
                        frac = 1
                    self.reward_model.change_batch(frac)
                    # update margin --> not necessary / will be updated soon
                    new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                    self.reward_model.set_teacher_thres_skip(new_margin)
                    self.reward_model.set_teacher_thres_equal(new_margin)
                    
                    # first learn reward
                    self.learn_reward(first_flag=1)
                    
                    # relabel buffer
                    self.replay_buffer.relabel_with_predictor(self.reward_model)

                    # reset interact_count
                    interact_count = 0
                
                if self.cfg.num_unsup_steps>0:
                    # reset Q due to unsuperivsed exploration
                    self.agent.reset_critic()
                
                # update agent
                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step, 
                    gradient_update=self.cfg.gradient_update, 
                    policy_update=True)
            
            # For State Explanation
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
                # if obs space contains more that an image
                # Add if minigrid has grid state
                next_obs = next_obs['image'] if self.state_type == 'grid' else next_obs
                # Convert action to int as the obs space
                action = np.array([action], dtype=np.uint8)
            
            obs_flat = gym_utils.flatten(self.obs_space,obs)

            # Check if reward model is used
            if self.cfg.learn_reward == True:
                reward_hat = self.reward_model.r_hat(np.concatenate([obs_flat, action], axis=-1))

            # allow infinite bootstrap
            terminated = float(terminated)
            # Check if reward model is used
            if self.cfg.learn_reward == True:
                episode_reward += reward_hat
            else:
                episode_reward += reward
            true_episode_reward += reward
            
            if self.log_success:
                episode_success = max(episode_success, terminated)
            
            # Check if reward model is used
            if self.cfg.learn_reward == True:
                # adding data to the reward training data
                self.reward_model.add_data(obs, action, reward, terminated, truncated, env_snapshot)
                self.replay_buffer.add(obs, action, reward_hat, next_obs, terminated, truncated)
            else:
                self.replay_buffer.add(obs, action, reward, next_obs, terminated, truncated)

            # Save model checkpoint
            if self.cfg.checkpoint_frec > 0 and self.step % self.cfg.checkpoint_frec == 0:
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

    def save_snapshot(self):
        snapshot_dir = self.cfg.snapshot_dir
        snapshot = snapshot_dir / f'snapshot_{self.global_frame}.pt'
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        self.agent.save(snapshot_dir, self.global_frame)
        # Check if reward model is used
        if self.cfg.learn_reward == True:
            self.reward_model.save(snapshot_dir, self.global_frame)
        keys_to_save = ['replay_buffer', 'step', 'episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        torch.save(payload, snapshot, pickle_protocol=4)
        
@hydra.main(version_base=None, config_path="config", config_name='train_themis')
def main(cfg : DictConfig):
    work_dir = Path.cwd()
    cfg.snapshot_dir = work_dir / cfg.snapshot_dir
    snapshot = cfg.snapshot_dir / f'snapshot_{(cfg.num_seed_steps + cfg.num_unsup_steps)*cfg.action_repeat}.pt'
    
    if not snapshot.exists():
        print(f"Snapshot doesn't exist at {cfg.snapshot_dir}")
        print('Execute the pretraining phase first')
        exit()

    workspace = Workspace(cfg, work_dir)
    
    workspace.run()
    if snapshot.exists():
        print(f'Overwriting models at: {work_dir}')
    workspace.save_snapshot()

if __name__ == '__main__':
    main()