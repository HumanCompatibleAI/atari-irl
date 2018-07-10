import pickle
import numpy as np
from rllab.envs.base import Env
from rllab.envs.gym_env import convert_gym_space
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, wrap_deepmind
from gym.spaces.discrete import Discrete
from sandbox.rocky.tf.spaces import Box
import gym

def vec_normalize(env):
    return VecNormalize(env)

mujoco_modifiers = {
    'env_modifiers': [],
    'vec_env_modifiers': [vec_normalize]
}

# from baselines.common.cmd_util.make_atari_env
def wrap_env_with_args(Wrapper, **kwargs):
    return lambda env: Wrapper(env, **kwargs)

def noop_reset(noop_max):
    def _thunk(env):
        assert 'NoFrameskip' in env.spec.id
        return NoopResetEnv(env, noop_max=noop_max)
    return _thunk

def atari_setup(env):
    # from baselines.common.atari_wrappers
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env


class OneHotDecodingEnv(gym.Wrapper):
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, one_hot_actions):
        return self.env.step(np.argmax(one_hot_actions, axis=0))


class TimeLimitEnv(gym.Wrapper):
    def __init__(self, env, time_limit=500):
        gym.Wrapper.__init__(self, env)
        self.steps = 0
        self.time_limit = time_limit

    def reset(self, **kwargs):
        self.steps = 0
        return self.env.reset(**kwargs)

    def step(self, actions):
        f1, f2, done, f3 = self.env.step(actions)
        self.steps += 1
        if self.steps > self.time_limit:
            done = True

        return f1, f2, done, f3


atari_modifiers = {
    'env_modifiers': [
        wrap_env_with_args(NoopResetEnv, noop_max=30),
        wrap_env_with_args(MaxAndSkipEnv, skip=4),
        wrap_deepmind,
        wrap_env_with_args(TimeLimitEnv)
    ],
    'vec_env_modifiers': [
        wrap_env_with_args(VecFrameStack, nstack=4)
    ]
}


one_hot_atari_modifiers = {
    'env_modifiers': atari_modifiers['env_modifiers'] + [
        wrap_env_with_args(OneHotDecodingEnv)
    ],
    'vec_env_modifiers': atari_modifiers['vec_env_modifiers']
}


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


def wrap_action_space(action_space):
    return Box(0, 1, shape=action_space.n)


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
        if isinstance(self._action_space, Box):
            return self._action_space
        else:
            return wrap_action_space(self._action_space)

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

    def reset(self, **kwargs):
        print("Reset")
        self.venv.reset(**kwargs)
