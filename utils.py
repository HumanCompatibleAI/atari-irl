"""
This may all be thrown away soonish, but I could imagine keeping these design
patterns in some form or other.

I hope that most of our patches to the baselines + gym code can happen in this
library, and not need to move into other parts of the code.

Desiderata:
- Not introduce too many dependencies over Adam's patched baselines library
- Basically work and be easy to use
- Contain most of our other patches over other libraries
- Generate useful information about whether or not we want to keep this
  incarnation of things

This is heavily based on
- https://github.com/openai/baselines/blob/master/baselines/ppo2/run_mujoco.py
- https://github.com/AdamGleave/baselines/tree/master/baselines/ppo2
"""

import tensorflow as tf
import numpy as np

from baselines import bench, logger
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common import set_global_seeds

import gym
import csv
import matplotlib.pyplot as plt


def optional_teardown(context, teardown_on_context_exit=True):
    if teardown_on_context_exit:
        return context
    else:
        context.teardown = context.__exit__

        def no_args_safe_exit(*args):
            args = [None, None, None] if not args else args
            context.teardown(*args)
        context.__exit__ = no_args_safe_exit

    return context


class TfContext:
    def __init__(self, ncpu=1):
        config = tf.ConfigProto(
            allow_soft_placement=True,
            intra_op_parallelism_threads=ncpu,
            inter_op_parallelism_threads=ncpu,
            device_count={'GPU': 1},
        )
        config.gpu_options.allow_growth=True
        self.tf_session = tf.Session(config=config)
    def __enter__(self):
        self.tf_session.__enter__()
        return self

    def __exit__(self, *args):
        self.tf_session.__exit__(*args)
        tf.reset_default_graph()


class EnvironmentContext:
    def __init__(self, env_name, seed, n_envs=1, env_modifiers=list(), vec_env_modifiers=list()):
        self.env_name = env_name
        self.n_envs = n_envs
        self.env_modifiers = env_modifiers
        self.vec_env_modifiers = vec_env_modifiers
        self.seed = seed

    def __enter__(self):
        def make_env(i):
            def _thunk():
                env = gym.make(self.env_name)
                env.seed(i)
                for fn in self.env_modifiers:
                    env = fn(env)
                env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
                return env
            return _thunk

        set_global_seeds(self.seed)
        self.base_vec_env = SubprocVecEnv([make_env(i + self.seed) for i in range(self.n_envs)])
        self.environments = self.base_vec_env
        for fn in self.vec_env_modifiers:
            self.environments = fn(self.environments)

        return self

    def __exit__(self, *args):
        self.base_vec_env.close()


def read_cols_from_dict(dirname, *cols):
    ans = dict([(c, []) for c in cols])
    with open(dirname + '/progress.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for c in cols:
                ans[c].append(float(row[c]))
    return (ans[c] for c in cols)


def plot_from_dirname(dirname):
    plt.plot(*read_cols_from_dict(dirname,'total_timesteps', 'eprewmean'))


def one_hot(x, dim):
    assert isinstance(x, list) or len(x.shape) == 1
    ans = np.zeros((len(x), dim))
    for n, i in enumerate(x):
        ans[n, i] = 1
    return ans
