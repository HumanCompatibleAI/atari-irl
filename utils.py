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
import gym


class PPOInterfaceContext():
    """
    A context for running PPO steps on a model

    This context is intended to help with cleanup and teardown of our
    tensorflow session and environments. A lot of unexpected-by-me behavior
    stems from having to do setup and teardown steps, and it seems like the
    context abstraction is good for this -- we can just be inside a context,
    and assume that it sets up + gets rid of itself correctly.
    """
    def __init__(
            self,
            env_name='CartPole-v1', ncpu=1, n_envs=1,
            teardown_on_context_exit=True
    ):
        """
        Create a context for running PPO steps on a model

        Args:
            env_name: name of the environment we want to use
            normalize_env: whether or not to normalize the environment
            ncpu: number of operation parallelism threads
            n_envs: number of environments
            teardown_on_context_exit: whether or not to teardown on exit
        """
        self.env_name = env_name
        self.teardown_on_context_exit = teardown_on_context_exit

        self.config = tf.ConfigProto(
            allow_soft_placement=True,
            intra_op_parallelism_threads=ncpu,
            inter_op_parallelism_threads=ncpu,
            device_count={'GPU': 2},
            log_device_placement=True
        )
        self.tf_session_context = tf.Session(config=self.config)

        def make_env():
            env = gym.make(self.env_name)
            env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
            return env

        self.environments = DummyVecEnv([make_env for _ in range(n_envs)])

    def __enter__(self):
        self.tf_session_context.__enter__()
        return self

    def teardown_environments(self):
        [monitor.unwrapped.close() for monitor in self.environments.unwrapped.envs]

    def teardown_tf_session(self, *args):
        # the tf session context wants exception parameters. If there weren't
        # any, then our args should have been None
        args = [None, None, None] if not args else args
        self.tf_session_context.__exit__(*args)
        tf.reset_default_graph()

    def teardown(self, *args):
        # the tf session context wants exception parameters. If there weren't
        # any, then our args should have been None
        args = [None, None, None] if not args else args

        self.teardown_environments()
        self.teardown_tf_session(*args)

    def __exit__(self, *args):
        if self.teardown_on_context_exit:
            self.teardown(*args)