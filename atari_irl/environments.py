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
        wrap_env_with_args(TimeLimitEnv, time_limit=1000)
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


class JustPress1Environment(gym.Env):
    def __init__(self):
        super().__init__()
        self.reward_range = (0, 1)
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(90, 90, 3))

        self.black = np.zeros(self.observation_space.shape)
        self.white = np.ones(self.observation_space.shape) * 255
        
        self.random_seed = 0
        self.np_random = np.random.RandomState(0)
        
    def seed(self, seed=None):
        if seed is None:
            seed = 0
        self.random_seed = seed
        self.np_random.seed(seed)
        
    def is_done(self):
        return self.np_random.random_sample() > .99

    def step(self, action):
        if action == 0:
            return self.black, 0.0, self.is_done(), {}
        else:
            return self.white, 1.0, self.is_done(), {}

    def reset(self):
        return self.black

    def render(self):
        raise NotImplementedError
        
    def get_action_meanings(self):
        return ['NOOP', 'OP', 'USELESS1', 'USELESS2', 'USELESS3', 'USELESS4']


class SimonSaysEnvironment(JustPress1Environment):
    def __init__(self):
        super().__init__()
        
        self.next_move = self.np_random.randint(2)
        self.obs_map = {
            0: self.black,
            1: self.white
        }
        self.turns = 0
        
    def is_done(self):
        self.turns += 1
        return self.turns > 100

    def step(self, action):
        reward = 0.0
        if isinstance(action, np.int64) and action == self.next_move or not isinstance(action, np.int64) and action[0] == self.next_move:
            reward = 1.0
        obs = self.reset()
        return obs, reward, self.is_done(), {'next_move': self.next_move}

    def reset(self):
        self.turns = 0
        self.next_move = self.np_random.randint(2)
        return self.obs_map[self.next_move]


class VisionSaysEnvironment(SimonSaysEnvironment):
    def __init__(self):
        super().__init__()
        self.zero = np.zeros(self.observation_space.shape)
        self.one = np.zeros(self.observation_space.shape)

        self.zero[3, 2:7, :] = 255
        self.zero[5, 2:7, :] = 255
        self.zero[4, 2, :] = 255
        self.zero[4, 6, :] = 255

        self.obs_map = {
            0: self.zero,
            1: self.one
        }