import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
import os
import random
import math

from collections import deque
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.wrappers import NormalizeReward, ResizeObservation, FrameStack
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper, FullyObsWrapper, PositionBonus
from rlkit.envs.wrappers import NormalizedBoxEnv, NormalizePixelObs

#from stable_baselines3.common.env_util import make_atari_env as sb3env
#from skimage.util.shape import view_as_windows
from torch import nn
from torch import distributions as pyd
    
def make_box2d_env(cfg, render_mode=None):
    env = eval_env = sim_env = None

    #Helper function to create Box2D environment
    id = cfg.env
    if 'BipedalWalker' in id:
        env = gym.make(id=id, render_mode=None)
        eval_env =  gym.make(id=id, render_mode=render_mode) 
    elif 'Moonlander' in id:   
        env = gym.make(id=id, continuous=True, render_mode=None)
        eval_env =  gym.make(id=id, continuous=True, render_mode=render_mode)
    elif 'CarRacing' in id:
        env = gym.make(id=id, continuous=True, render_mode=None)
        eval_env =  gym.make(id=id, continuous=True, render_mode=render_mode)
        
    return TimeLimit(NormalizedBoxEnv(env), env._max_episode_steps), TimeLimit(NormalizedBoxEnv(eval_env), eval_env._max_episode_steps)

def make_control_env(cfg, render_mode=None):
    env = eval_env = sim_env = None
    #Helper function to create MUJOCO environment
    id = cfg.env
    env = gym.make(id=id, render_mode=None)   
    env = TimeLimit(RewindWrapper(NormalizedBoxEnv(env), cfg.domain), env._max_episode_steps)

    eval_env =  gym.make(id=id, render_mode=render_mode)
    eval_env = TimeLimit(NormalizedBoxEnv(eval_env), eval_env._max_episode_steps)

    if cfg.human_teacher or cfg.debug:
        sim_env = gym.make(id=id, render_mode='rgb_array')
        sim_env = TimeLimit(RewindWrapper(NormalizedBoxEnv(sim_env), cfg.domain), sim_env._max_episode_steps)

    return env, eval_env, sim_env

def make_minigrid_env(cfg, render_mode=None):
    env = eval_env = sim_env = None
    #Helper function to create MiniGrid environment
    id=cfg.domain+'-'+cfg.env
    env = gym.make(id=id, render_mode=None)
    
    env = TimeLimit(
            RewindWrapper(
                NormalizePixelObs(
                    ImgObsWrapper(
                        RGBImgObsWrapper(
                            PositionBonus(FullyObsWrapper(env)),
                            tile_size = 10
                        )
                    ),
                    reward_scale = cfg.reward_scale,
                    reward_intercept = cfg.reward_intercept
                ), 
                cfg.domain), 
            max_episode_steps = 100)

    eval_env =  gym.make(id=id, render_mode=render_mode)   
    eval_env = TimeLimit(
                NormalizePixelObs(
                    ImgObsWrapper(
                        RGBImgObsWrapper(
                            PositionBonus(FullyObsWrapper(eval_env)),
                            tile_size = 10
                        )
                    ),
                    reward_scale = cfg.reward_scale,
                    reward_intercept = cfg.reward_intercept
                ), 
                max_episode_steps = 100)

    if cfg.human_teacher or cfg.debug:
        sim_env = gym.make(id=id, render_mode='rgb_array')
        sim_env = TimeLimit(
                    RewindWrapper(
                        NormalizePixelObs(
                            ImgObsWrapper(
                                RGBImgObsWrapper(
                                    PositionBonus(FullyObsWrapper(sim_env)),
                                    tile_size = 10
                                )
                            ),
                            reward_scale = cfg.reward_scale,
                            reward_intercept = cfg.reward_intercept
                        ), cfg.domain
                    ), 
                    max_episode_steps = 100)

    return env, eval_env, sim_env

def make_atari_env(cfg, render_mode=None):
    gym.logger.set_level(40)
    print('GYM LOCATION: ',gym.__file__)
    env = eval_env = sim_env = None
    #Helper function to create Atari environment
    id=cfg.domain+'/'+cfg.env
    max_episode_steps = 1000

    env = gym.make(id=id, 
                   mode=cfg.mode, 
                   difficulty=cfg.difficulty, 
                   obs_type=cfg.obs_type, 
                   frameskip = cfg.frameskip,
                   repeat_action_probability=cfg.repeat_action_probability,
                   full_action_space=cfg.full_action_space,
                   render_mode=None)
    env = NormalizeReward(TimeLimit(RewindWrapper(ResizeObservation(env,64), cfg.domain), max_episode_steps = max_episode_steps))

    eval_env =  gym.make(id=id, 
                        mode=cfg.mode, 
                        difficulty=cfg.difficulty, 
                        obs_type=cfg.obs_type, 
                        frameskip = cfg.frameskip,
                        repeat_action_probability=cfg.repeat_action_probability,
                        full_action_space=cfg.full_action_space,
                        render_mode = render_mode)
    eval_env = NormalizeReward(TimeLimit(ResizeObservation(eval_env,64), max_episode_steps = max_episode_steps))
    
    if cfg.human_teacher or cfg.debug:
        sim_env = gym.make(id=id, 
                   mode=cfg.mode, 
                   difficulty=cfg.difficulty, 
                   obs_type=cfg.obs_type, 
                   frameskip = cfg.frameskip,
                   repeat_action_probability=cfg.repeat_action_probability,
                   full_action_space=cfg.full_action_space,
                   render_mode='rgb_array')
        sim_env = NormalizeReward(TimeLimit(RewindWrapper(ResizeObservation(sim_env,64), cfg.domain), max_episode_steps = max_episode_steps))

    return env, eval_env, sim_env

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))
    
class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu
    
class TorchRunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=(), device=None):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon

    def update(self, x):
        with torch.no_grad():
            batch_mean = torch.mean(x, axis=0)
            batch_var = torch.var(x, axis=0)
            batch_count = x.shape[0]
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    @property
    def std(self):
        return torch.sqrt(self.var)

def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta + batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.pow(delta, 2) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

def cnn(obs_space, n_input_channels, mode=0):

    kernel_size = [[8,4,3],[3,3,3]] # Parameterisation
    stride = [[4,2,1],[1,1,1]]
    padding =[[0,0,0],[0,0,0]]

    kernel_size=kernel_size[mode]
    stride = stride[mode]
    padding = padding[mode]

    feature_extractor=nn.Sequential(
        nn.Conv2d(n_input_channels, 32, kernel_size=kernel_size[0], stride=stride[0], padding=padding[0]),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=kernel_size[1], stride=stride[1], padding=padding[1]),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=kernel_size[2], stride=stride[2], padding=padding[2]),
        nn.ReLU(),
        nn.Flatten(),
    )
    
    # Compute shape by doing one forward pass
    with torch.no_grad():
        obs_sample = torch.as_tensor(obs_space.sample()[None]).permute(0, 3, 1, 2)
        n_flatten = feature_extractor(obs_sample.float()).shape[1]
    
    #feature_extractor

    return feature_extractor, n_flatten

def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()

class RewindWrapper(gym.Wrapper):
    def __init__(self, env, domain):
        super().__init__(env)

        self.domain = domain

    def get_state(self):
        if self.domain == 'ALE':
            return self.env.ale.cloneState()
        elif self.domain == 'MiniGrid':
            return self.env.get_state()
        elif self.domain == 'BabyAI':
            return self.env.get_state()
        elif self.domain == 'Control':
            return self.env.get_data()
        elif self.domain == 'Box2D':
            return self.env.get_state()
        else:
            print('You need to provide the get snapshot functionality of your environment')
            raise NotImplementedError

    
    def set_state(self, snapshot):
        if self.domain == 'ALE':
            self.env.ale.restoreState(snapshot)
        elif self.domain == 'MiniGrid':
            self.env.set_state()
        elif self.domain == 'BabyAI':
            self.env.set_state()
        elif self.domain == 'Control':
            self.env.set_state()
        elif self.domain == 'Box2D':
            self.env.set_state()
        else:
            print('You need to provide the set snapshot functionality of your environment')
            raise NotImplementedError