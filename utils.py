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

import numpy as np
from baselines.common.cmd_util import mujoco_arg_parser
from baselines import bench, logger
import tensorflow as tf

from baselines.common import set_global_seeds
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import MlpPolicy, CnnPolicy
import gym
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

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
            env_name='CartPole-v1', normalize_env=True, ncpu=1, n_envs=1,
            close_on_context_exit=True
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
        self.teardown_on_context_exit = close_on_context_exit

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

        dummy_vec = DummyVecEnv([make_env for _ in range(n_envs)])

        if normalize_env:
            self.environments = VecNormalize(dummy_vec)
        else:
            self.environments = dummy_vec

    def __enter__(self):
        self.tf_session_context.__enter__()
        return self

    def teardown(self, *args):
        # TODO(Aaron): iterate through environments and close them for real
        self.environments.close()

        self.tf_session_context.__exit__(*args)
        tf.reset_default_graph()

    def __exit__(self, *args):
        if self.teardown_on_context_exit:
            self.teardown(*args)

def run_policy(*, model, environments):
    logger.configure()
    logger.log("Running trained model")

    # Initialize the stuff we want to keep track of
    rewards = []

    # Initialize our environment
    done = [False]
    obs = np.zeros((environments.num_envs,) + environments.observation_space.shape)
    obs[:] = environments.reset()

    # run the policy until done
    while not any(done):
        actions, _, _, _ = model.step(obs)
        obs[:], reward, done, info = environments.step(actions)
        rewards.append(reward)
        environments.render()

    logger.log("Survived {} time steps".format(len(rewards)))
    logger.log("Got total reward {}\n".format(sum(rewards[:])))


def train(*, env, policy, num_timesteps, seed):
    set_global_seeds(seed)

    # this uses patched behavior, since baselines 1.5.0 doesn't return from
    # learn
    return ppo2.learn(
        policy=policy, env=env, nsteps=2048, nminibatches=32,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=0.0, lr=3e-4, cliprange=0.2, total_timesteps=num_timesteps
    )