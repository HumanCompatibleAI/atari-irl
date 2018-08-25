from baselines.ppo2.ppo2 import Model, constfn
from .sampling import PPOBatch

import numpy as np

from collections import namedtuple


"""
Heavily based on the ppo2 implementation found in the OpenAI baselines library,
particularly the ppo_trainsteps function.
"""

BatchingConfig = namedtuple('BatchingInfo', [
    'nbatch', 'nbatch_train', 'noptepochs', 'nenvs', 'nsteps', 'nminibatches'
])

def make_batching_config(*, nenvs, nsteps, noptepochs, nminibatches):
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    return BatchingConfig(
        nbatch=nbatch, nbatch_train=nbatch_train, noptepochs=noptepochs,
        nenvs=nenvs, nsteps=nsteps, nminibatches=nminibatches
    )


def ppo_train_steps(
        *,
        model: Model,
        run_info: PPOBatch,
        batching_config: BatchingConfig,
        lrnow: float,
        cliprangenow: float,
        nbatch_train: int
): # I'm not quite sure what type mblossvals is
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


class PPOOptimizer:
    def __init__(
            self,
            *,
            policy,
            batching_config: BatchingConfig,

            lr=3e-4,
            cliprange=0.2,

            total_timesteps=10e6
    ):
        # Deal with the policy
        self.policy = policy
        assert hasattr(policy, 'model')
        assert isinstance(policy.model, Model)

        # Set things based on the batching config
        self.batching_config = batching_config
        self.nbatch = self.batching_config.nbatch
        self.nupdates = total_timesteps // self.batching_config.nbatch

        # Deal with constant arguments
        if isinstance(lr, float):
            lr = constfn(lr)
        else:
            assert callable(lr)
        if isinstance(cliprange, float):
            cliprange = constfn(cliprange)
        else:
            assert callable(cliprange)

        self.lr = lr
        self.cliprange = cliprange

    def optimize_policy(self, itr: int, ppo_batch: PPOBatch):
        # compute our learning rate and clip ranges
        assert self.nbatch % self.batching_config.nminibatches == 0
        nbatch_train = self.nbatch // self.batching_config.nminibatches
        frac = 1.0 - (itr - 1.0) / self.nupdates
        assert frac > 0.0
        lrnow = self.lr(frac)
        cliprangenow = self.cliprange(frac)

        # Run the training steps for PPO
        mblossvals = ppo_train_steps(
            model=self.policy.model,
            run_info=ppo_batch,
            batching_config=self.batching_config,
            lrnow=lrnow,
            cliprangenow=cliprangenow,
            nbatch_train=nbatch_train
        )

        return np.mean(mblossvals, axis=0)
