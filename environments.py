import pickle
import numpy as np
from rllab.envs.base import Env
from rllab.envs.gym_env import convert_gym_space

class ConstantStatistics(object):
    def __init__(self, running_mean):
        self.mean = running_mean.mean
        self.var = running_mean.var
        self.count = running_mean.count

    def update(self, x):
        pass

    def update_from_moments(self, _batch_mean, _batch_var, _batch_count):
        pass


def serialize_env_wrapper(env_wrapper):
    venv = env_wrapper.venv
    env_wrapper.venv = None
    serialized = pickle.dumps(env_wrapper)
    env_wrapper.venv = venv
    return serialized


def restore_serialized_env_wrapper(env_wrapper, venv):
    env_wrapper.venv = venv
    env_wrapper.num_envs = venv.num_envs
    if hasattr(env_wrapper, 'ret'):
        env_wrapper.ret = np.zeros(env_wrapper.num_envs)
    return env_wrapper


def make_const(norm):
    '''Monkey patch classes such as VecNormalize that use a
       RunningMeanStd (or compatible class) to keep track of statistics.'''
    for k, v in norm.__dict__.items():
        if hasattr(v, 'update_from_moments'):
            setattr(norm, k, ConstantStatistics(v))


# Copied from https://github.com/HumanCompatibleAI/population-irl/blob/master/pirl/irl/airl.py
# this hacks around airl being built on top of rllib, and not using gym
# environments
class VecGymEnv(Env):
    def __init__(self, venv):
        self.venv = venv
        self._observation_space = convert_gym_space(venv.observation_space)
        self._action_space = convert_gym_space(venv.action_space)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def terminate(self):
        # Normally we'd close environments, but pirl.experiments handles this.
        pass

    @property
    def vectorized(self):
        return True

    def vec_env_executor(self, n_envs, max_path_length):
        # SOMEDAY: make these parameters have an effect?
        # We're powerless as the environments have already been created.
        # But I'm not too bothered by this, as we can tweak them elsewhere.
        return self.venv