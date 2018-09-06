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
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common import set_global_seeds

from atari_irl import environments
from atari_irl.environments import one_hot

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
    def __init__(self, *, env_name=None, make_env=None, seed, n_envs=1, env_modifiers=list(), vec_env_modifiers=list()):
        self.env_name = env_name
        if make_env is None:
            make_env = lambda: gym.make(self.env_name)
        self.make_env = make_env
        self.n_envs = n_envs
        self.env_modifiers = env_modifiers
        self.vec_env_modifiers = vec_env_modifiers
        self.seed = seed

    def __enter__(self):
        def make_env(i):
            def _thunk():
                env = self.make_env()
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


def read_cols_from_dict(dirname, *cols, start=0, end=-1):
    ans = dict([(c, []) for c in cols])
    with open(dirname + '/progress.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for c in cols:
                ans[c].append(float(row[c]))
    return (ans[c][start:end] for c in cols)


def plot_from_dirname(dirname):
    plt.plot(*read_cols_from_dict(dirname,'total_timesteps', 'eprewmean'))


def batched_call(fn, batch_size, args, check_safety=True):
    N = args[0].shape[0]
    for arg in args:
        assert arg.shape[0] == N

    # Things get super slow if we don't do this
    if N == batch_size:
        return fn(*args)

    arg0_batches = []
    fn_results = []

    start = 0

    def slice_result(result, subslice):
        if isinstance(result, dict):
            return dict(
                (key, value[subslice])
                for key, value in result.items()
            )
        else:
            return result[subslice]

    def add_batch(*args_batch, subslice=None):
        results_batch = fn(*args_batch)
        if subslice:
            results_batch = [slice_result(r, subslice) for r in results_batch]
            args_batch = [slice_result(r, subslice) for r in args_batch]
        fn_results.append(results_batch)
        if check_safety:
            arg0_batches.append(args_batch[0])

    # add data for all of the batches that cleanly fit inside the batch size
    for start in range(0, N - batch_size, batch_size):
        end = start + batch_size
        add_batch(*[arg[start:end] for arg in args])

    # add data for the last batch that would run past the end of the data if it
    # were full
    start += batch_size
    if start != N:
        remainder_slice = slice(start - N, batch_size)
        add_batch(
            *(arg[N - batch_size:N] for arg in args),
            subslice=remainder_slice
        )

    # integrity check
    if check_safety:
        final_arg0 = np.vstack(arg0_batches)

    # reshape everything
    final_results = []
    for i, res in enumerate(fn_results[0]):
        if isinstance(res, np.ndarray) or isinstance(res, list):
            final_results.append(
                np.vstack([results_batch[i] for results_batch in fn_results])
            )
        elif isinstance(res, dict):
            for key, item in res.items():
                assert isinstance(item, np.ndarray) or isinstance(item, list)
            final_results.append(dict(
                (
                    key,
                    np.vstack([
                        results_batch[i][key] for results_batch in fn_results
                    ])
                )
                for key in res.keys()
            ))
        else:
            raise NotImplementedError

    # Integrity checks in case I wrecked this
    if check_safety:
        assert len(final_arg0) == N
        assert np.isclose(final_arg0, args[0]).all()
    return final_results


class TfEnvContext:
    def __init__(self, tf_cfg, env_config):
        self.tf_cfg = tf_cfg
        self.env_config = env_config
        self.seed = env_config['seed']

        env_modifiers = environments.env_mapping[env_config['env_name']]
        one_hot_code = env_config.pop('one_hot_code')
        if one_hot_code:
            env_modifiers = environments.one_hot_wrap_modifiers(env_modifiers)
        self.env_config.update(env_modifiers)

    def __enter__(self):
        self.env_context = EnvironmentContext(**self.env_config)
        self.env_context.__enter__()
        self.train_graph = tf.Graph()
        self.tg_context = self.train_graph.as_default()
        self.tg_context.__enter__()
        self.sess = tf.Session(config=self.tf_cfg)
        # from tensorflow.python import debug as tf_debug
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess , ui_type='readline')
        self.sess_context = self.sess.as_default()
        self.sess_context.__enter__()
        tf.set_random_seed(self.seed)

        return self

    def __exit__(self, *args):
        self.sess_context.__exit__(*args)
        self.tg_context.__exit__(*args)
        self.env_context.__exit__(*args)
