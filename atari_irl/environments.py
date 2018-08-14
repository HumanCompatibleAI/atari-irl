import pickle
import numpy as np
import tensorflow as tf
from rllab.envs.base import Env
from rllab.envs.gym_env import convert_gym_space
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, wrap_deepmind
from gym.spaces.discrete import Discrete
from sandbox.rocky.tf.spaces import Box
import gym
from atari_irl.utils import one_hot

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


class VecRewardZeroingEnv(VecEnvWrapper):
    def step(self, actions):
        _1, reward, _2, infos = self.venv.step(actions)
        for info in infos:
            if 'episode' in info:
                info['episode']['r'] = 0
        return _1, 0, _2, infos

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        self.venv.step_wait()


class VecIRLRewardEnv(VecEnvWrapper):
    def __init__(self, env, *, reward_network):
        VecEnvWrapper.__init__(self, env)
        self.reward_network = reward_network
        self.prev_obs = None

    def step(self, acts):
        obs, _, done, info = self.venv.step(acts)

        assert np.sum(_) == 0

        rewards = tf.get_default_session(
        ).run(self.reward_network.reward, feed_dict={
            self.reward_network.act_t: acts,
            self.reward_network.obs_t: obs
        })
        assert len(rewards) == len(obs)
        return obs, rewards.reshape(rewards.shape[0]), done, info

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        self.venv.step_wait()


class VecOneHotEncodingEnv(VecEnvWrapper):
    def __init__(self, venv, dim=6):
        VecEnvWrapper.__init__(self, venv)
        self.dim = dim

    def step(self, actions):
        return self.venv.step(one_hot(actions, self.dim))

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        self.venv.step_wait()

easy_env_modifiers = {
    'env_modifiers': [
        wrap_deepmind,
        wrap_env_with_args(TimeLimitEnv, time_limit=5000)
    ],
    'vec_env_modifiers': [
        wrap_env_with_args(VecFrameStack, nstack=4)
    ]
}

import functools
# Episode Life causes us to not actually reset the environments, which means
# that interleaving, and even running the normal sampler a bunch of times
# will give us super short trajectories. Right now we set it to false, but
# that's not an obviously correct way to handle the problem
atari_modifiers = {
    'env_modifiers': [
        wrap_env_with_args(NoopResetEnv, noop_max=30),
        wrap_env_with_args(MaxAndSkipEnv, skip=4),
        functools.partial(wrap_deepmind, episode_life=False),
        #wrap_env_with_args(TimeLimitEnv, time_limit=5000)
    ],
    'vec_env_modifiers': [
        wrap_env_with_args(VecFrameStack, nstack=4)
    ]
}


def one_hot_wrap_modifiers(modifiers):
    return {
        'env_modifiers': modifiers['env_modifiers'] + [
            wrap_env_with_args(OneHotDecodingEnv)
        ],
        'vec_env_modifiers': modifiers['vec_env_modifiers']
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
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(210, 160, 3))

        self.black = np.zeros(self.observation_space.shape).astype(np.uint8)
        self.white = np.ones(self.observation_space.shape).astype(np.uint8) * 255
        
        self.random_seed = 0
        self.np_random = np.random.RandomState(0)

        class Ale:
            def lives(self):
                return 1
        self.ale = Ale()

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

    @staticmethod
    def isint(n):
        return isinstance(n, np.int64) or isinstance(n, int)

    def set_next_move_get_obs(self):
        self.next_move = self.np_random.randint(2)
        return self.obs_map[self.next_move]

    def step(self, action):
        reward = 0.0
        self.turns += 1
        if (
            self.isint(action) and action == self.next_move or
            not self.isint(action) and action[0] == self.next_move
        ):
            reward = 1.0
        obs = self.set_next_move_get_obs()
        return obs, reward, self.turns >= 100, {'next_move': self.next_move}

    def reset(self):
        self.turns = 0
        return self.set_next_move_get_obs()


class VisionSaysEnvironment(SimonSaysEnvironment):
    def __init__(self):
        super().__init__()
        self.zero = np.zeros(self.observation_space.shape).astype(np.uint8)
        self.one = np.zeros(self.observation_space.shape).astype(np.uint8)

        self.one[50:150, 120:128, :] = 255

        self.zero[50:150, 100:108, :] = 255
        self.zero[50:150, 140:148, :] = 255
        self.zero[50:58, 100:148, :] = 255
        self.zero[142:150, 100:148, :] = 255

        self.obs_map = {
            0: self.one,
            1: self.zero
        }


gym.envs.register(
    id='VisionSays-v0',
    entry_point='atari_irl.environments:VisionSaysEnvironment'
)

gym.envs.register(
    id='SimonSays-v0',
    entry_point='atari_irl.environments:SimonSaysEnvironment'
)

env_mapping = {
    'PongNoFrameskip-v4': atari_modifiers,
    'CartPole-v1': mujoco_modifiers,
    'VisionSays-v0': easy_env_modifiers,
    'SimonSays-v0': easy_env_modifiers
}