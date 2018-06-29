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



from baselines.ppo2.ppo2 import Model, Runner, constfn, safemean
from baselines.common import explained_variance
from collections import deque, namedtuple
import time


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


class Learner:
    def __init__(self, policy_class, env, total_timesteps, seed, nsteps=2048,
                 ent_coef=0.0, lr=3e-4, vf_coef=0.5,  max_grad_norm=0.5,
                 gamma=0.99, lam=0.95, nminibatches=4, noptepochs=4,
                 cliprange=0.2, normalize_env=True):
        # Set the random seed
        set_global_seeds(seed)

        # Deal with constant arguments
        if isinstance(lr, float): lr = constfn(lr)
        else: assert callable(lr)
        if isinstance(cliprange, float): cliprange = constfn(cliprange)
        else: assert callable(cliprange)

        self.lr = lr
        self.cliprange = cliprange

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

        model_args = dict(
            policy=policy_class, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs,
            nbatch_train=nbatch_train, nsteps=nsteps, ent_coef=ent_coef,
            vf_coef=vf_coef, max_grad_norm=max_grad_norm
        )
        if normalize_env:
            env = VecNormalize(env)
            policy = EnvPolicy(model_args=model_args, envs=env)
        else:
            policy = Policy(model_args)

        model = policy.model
        runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

        # Set our major learner objects
        self.batching_config = batching_config
        self.policy = policy
        self.model  = model
        self.runner = runner

        # Set our last few run configurations
        self.callbacks = []
        self.nbatch = nbatch
        self.nupdates = total_timesteps // nbatch

        # Initialize the objects that will change as we learn
        self._update = 1
        self._epinfobuf = deque(maxlen=100)
        self._tfirststart = None
        self._run_info = None
        self._tnow = None
        self._fps = None
        self._lossvals = None

    @property
    def update(self):
        return self._update

    @property
    def mean_reward(self):
        return safemean([epinfo['r'] for epinfo in self._epinfobuf])

    def step(self):
        # initialize our start time if we haven't already
        if not self._tfirststart:
           self._tfirststart = time.time()

        # compute our learning rate and clip ranges
        assert self.nbatch % self.batching_config.nminibatches == 0
        nbatch_train = self.nbatch // self.batching_config.nminibatches
        tstart = time.time()
        frac = 1.0 - (self._update - 1.0) / self.nupdates
        lrnow = self.lr(frac)
        cliprangenow = self.cliprange(frac)

        # Run the model on the environments
        self._run_info = RunInfo(*self.runner.run())
        self._epinfobuf.extend(self._run_info.epinfos)

        # Run the training steps for PPO
        mblossvals = train_steps(
            model=self.policy.model,
            run_info=self._run_info,
            batching_config=self.batching_config,
            lrnow=lrnow,
            cliprangenow=cliprangenow,
            nbatch_train=nbatch_train
        )

        self._lossvals = np.mean(mblossvals, axis=0)
        self._tnow = time.time()
        self._fps = int(self.nbatch / (self._tnow - tstart))

        for check, fn in self.callbacks:
            if check(self.update):
                fn(**locals())

        self._update += 1
        if self._update > self.nupdates:
            logger.log("Warning, exceeded planned number of updates")

    def register_callback(self, check, fn):
        self.callbacks.append((check, fn))

    @staticmethod
    def check_update_interval(freq, include_first=True):
        return lambda i: i % freq == 0 or include_first and i == 1

    @staticmethod
    def print_log(self, **kwargs):
        print_log(
            model=self.model, batching_config=self.batching_config,
            update=self.update, epinfobuf=self._epinfobuf,
            tfirststart=self._tfirststart, run_info=self._run_info,
            lossvals=self._lossvals, fps=self._fps, tnow=self._tnow
        )

    def learn_and_yield(self, yield_fn, yield_freq, log_freq=None):
        if log_freq:
            self.register_callback(
                self.check_update_interval(log_freq),
                self.print_log
            )
        should_yield = self.check_update_interval(yield_freq)

        while self.update < self.nupdates:
            self.step()
            if should_yield(self.update):
                yield yield_fn(self)

        yield yield_fn(self)