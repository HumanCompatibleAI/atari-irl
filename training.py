import numpy as np
import tensorflow as tf

from baselines import logger
from baselines.common import explained_variance, set_global_seeds
from baselines.ppo2.ppo2 import Runner, constfn, safemean

import policies

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


def setup_policy(*, model_args, nenvs, ob_space, ac_space, env, save_env, checkpoint):
    if checkpoint:
        policy = policies.restore_policy_from_checkpoint_dir(
            checkpoint_dir=checkpoint, envs=env
        )
        assert policy.model_args == policy.model_args
        if isinstance(policy, policies.EnvPolicy):
            assert nenvs == policy.envs.num_envs
            assert ob_space == policy.envs.observation_space
            assert ac_space == policy.envs.action_space
            env = policy.envs
    else:
        if save_env:
            policy = policies.EnvPolicy(model_args=model_args, envs=env)
        else:
            policy = policies.Policy(model_args)
    return policy


class Learner:
    def __init__(self, policy_class, env, total_timesteps, seed, nsteps=2048,
                 ent_coef=0.0, lr=3e-4, vf_coef=0.5,  max_grad_norm=0.5,
                 gamma=0.99, lam=0.95, nminibatches=4, noptepochs=4,
                 cliprange=0.2, checkpoint=None, save_env=True):
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

        policy = setup_policy(
            model_args=model_args, env=env, nenvs=nenvs, checkpoint=checkpoint,
            ob_space=ob_space, ac_space=ac_space, save_env=save_env
        )

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