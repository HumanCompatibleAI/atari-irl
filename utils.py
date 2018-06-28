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

        dummy_vec = DummyVecEnv([make_env for _ in range(n_envs)])

        if normalize_env:
            self.environments = VecNormalize(dummy_vec)
        else:
            self.environments = dummy_vec

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

from baselines.ppo2.ppo2 import Model, Runner, constfn, safemean
from baselines.common import explained_variance
from collections import deque, namedtuple
import time
import os
import os.path as osp

RunInfo = namedtuple('RunInfo', [
    'obs', 'returns', 'masks', 'actions', 'values', 'neglogpacs', 'states',
    'epinfos'
])
RunInfo.train_args = lambda self: (
    self.obs, self.returns, self.masks, self.actions, self.values, self.neglogpacs
)
BatchingConfig = namedtuple('BatchingInfo', [
    'nbatch', 'noptepochs', 'nenvs', 'nsteps', 'nminibatches'
])


def train_steps(
        *, model, run_info, batching_config, lrnow, cliprangenow, nbatch_train
):
    states = run_info.states
    nbatch, noptepochs = batching_config.nbatch, batching_config.noptepochs
    nenvs, nminibatches = batching_config.nenvs, batching_config.nminibatches
    nsteps = batching_config.nsteps

    mblossvals = []
    if states is None:  # nonrecurrent version
        inds = np.arange(nbatch)
        for _ in range(noptepochs):
            np.random.shuffle(inds)
            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                mbinds = inds[start:end]
                slices = (arr[mbinds] for arr in run_info.train_args())
                mblossvals.append(model.train(lrnow, cliprangenow, *slices))

    else:  # recurrent version
        assert nenvs % nminibatches == 0
        envsperbatch = nenvs // nminibatches
        envinds = np.arange(nenvs)
        flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
        envsperbatch = nbatch_train // nsteps
        for _ in range(noptepochs):
            np.random.shuffle(envinds)
            for start in range(0, nenvs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mbflatinds = flatinds[mbenvinds].ravel()
                slices = (arr[mbflatinds] for arr in run_info.train_args())
                mbstates = states[mbenvinds]
                mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

    return mblossvals


def print_log(
        *, model, run_info, batching_config,
        lossvals, update, fps, epinfobuf, tnow, tfirststart
):
    ev = explained_variance(run_info.values, run_info.returns)
    logger.logkv("serial_timesteps", update * batching_config.nsteps)
    logger.logkv("nupdates", update)
    logger.logkv("total_timesteps", update * batching_config.nbatch)
    logger.logkv("fps", fps)
    logger.logkv("explained_variance", float(ev))
    logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
    logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
    logger.logkv('time_elapsed', tnow - tfirststart)
    for (lossval, lossname) in zip(lossvals, model.loss_names):
        logger.logkv(lossname, lossval)
    logger.dumpkvs()


def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            yield_interval=0, save_interval=0, load_path=None):

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    batching_config = BatchingConfig(
        nbatch=nbatch, noptepochs=noptepochs, nenvs=nenvs, nsteps=nsteps,
        nminibatches=nminibatches
    )

    make_model = lambda : Model(
        policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs,
        nbatch_train=nbatch_train, nsteps=nsteps, ent_coef=ent_coef,
        vf_coef=vf_coef, max_grad_norm=max_grad_norm
    )
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    if load_path is not None:
        model.load(load_path)

    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)

        # Run the model on the environments
        run_info = RunInfo(*runner.run())
        # Run the training steps for PPO
        mblossvals = train_steps(
            model=model, run_info=run_info, batching_config=batching_config,
            lrnow=lrnow, cliprangenow=cliprangenow, nbatch_train=nbatch_train
        )

        epinfobuf.extend(run_info.epinfos)

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))

        if log_interval and update % log_interval == 0 or update == 1:
            print_log(
                model=model, run_info=run_info, batching_config=batching_config,
                lossvals=lossvals, update=update, fps=fps, epinfobuf=epinfobuf,
                tnow=tnow, tfirststart=tfirststart
            )

        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)

        if yield_interval and update % yield_interval == 0 or update == 1:
            yield model, update, safemean([epinfo['r'] for epinfo in epinfobuf])

    env.close()
    return model


def train(*, environments, policy, num_timesteps, seed, yield_interval=0, log_interval=1):
    set_global_seeds(seed)

    # this uses patched behavior, since baselines 1.5.0 doesn't return from
    # learn
    return learn(
        policy=policy, env=environments, nsteps=2048, nminibatches=32,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=log_interval,
        yield_interval=yield_interval, ent_coef=0.0, lr=3e-4, cliprange=0.2,
        total_timesteps=num_timesteps
    )