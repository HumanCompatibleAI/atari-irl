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
from baselines import bench, logger
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common import set_global_seeds
import gym


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
            log_device_placement=True,
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
                env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
                for fn in self.env_modifiers:
                    env = fn(env)
                return env
            return _thunk

        def do_nothing():
            pass

        set_global_seeds(self.seed)
        self.base_vec_env = SubprocVecEnv([make_env(i + self.seed) for i in range(self.n_envs)])
        self.environments = self.base_vec_env
        for fn in self.vec_env_modifiers:
            self.environments = fn(self.environments)

        # Hacky monkey-patch so that we can work with rllab's environments
        # we'll actually close the environment for real when we exit it later
        #self.environments.terminate = do_nothing

        return self

    def __exit__(self, *args):
        self.base_vec_env.close()
        #[monitor.unwrapped.close() for monitor in self.base_vec_env.unwrapped.envs]
        pass
