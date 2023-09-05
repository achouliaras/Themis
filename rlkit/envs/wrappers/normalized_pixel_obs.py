import numpy as np

from rlkit.envs.proxy_env import ProxyEnv

class NormalizePixelObs(ProxyEnv):
    """
    Normalize pixel observations to [0,1].
    Optionally normalize observations and scale reward.
    """

    def __init__(
            self,
            env,
            reward_scale=1.,
            reward_intercept = 0.,
            obs_std=255,
    ):
        ProxyEnv.__init__(self, env)

        self._obs_std = np.array(obs_std, dtype = "float32")

        self._reward_scale = reward_scale
        self._reward_intercept = reward_intercept

    def _apply_normalize_obs(self, obs):
        return obs / self._obs_std

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self._wrapped_env.step(action)
        
        next_obs = self._apply_normalize_obs(next_obs)
        return next_obs, reward * self._reward_scale + self._reward_intercept, terminated, truncated, info

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env
