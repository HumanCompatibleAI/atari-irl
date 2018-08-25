import numpy as np

from baselines import logger
from baselines.common import explained_variance
from baselines.ppo2.ppo2 import constfn, safemean
from baselines.ppo2 import ppo2

from . import policies, utils
from .sampling import PPOBatchSampler
from .optimizers import PPOOptimizer, make_batching_config

from collections import deque, namedtuple
import time


PPOBatch = namedtuple('PPOBatch', [
    'obs', 'returns', 'masks', 'actions', 'values', 'neglogpacs', 'states',
    'epinfos'
])
PPOBatch.train_args = lambda self: (
    self.obs, self.returns, self.masks, self.actions, self.values, self.neglogpacs
)

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




def ppo_samples_to_trajectory_format(ppo_samples, num_envs=8):
    unravel_index = lambda *args: None
    # This is what IRLTRPO/IRLNPO actually uses
    # this will totally explode if we don't have them
    names_to_indices = {
        # The data that IRLTRPO/IRLNPO uses
        # we should expect the rllab-formatted code to explode if it
        # needs something else
        'observations': 0,
        'actions': 3,
        # The data that PPO uses
        'returns': 1,
        'dones': 2,
        'values': 4,
        'neglogpacs': 5
    }

    unraveled = dict(
        (key, unravel_index(index, ppo_samples, num_envs))
        for key, index in names_to_indices.items()
    )

    # This is a special case because TRPO wants advantages, but PPO
    # doesn't compute it
    returns = unravel_index(1, ppo_samples, num_envs)
    values = unravel_index(4, ppo_samples, num_envs)
    unraveled['advantages'] = returns - values

    T = unraveled['observations'].shape[0]
    for key, value in unraveled.items():
        assert len(value) == T

    trajectories = [dict((key, []) for key in unraveled.keys())]
    for t in range(T):
        for key in unraveled.keys():
            for i in range(num_envs):
                trajectories[i][key].append(unraveled[t][i])

    for i in range(num_envs):
        for key in unraveled.keys():
            if key != 'actions':
                trajectories[i]


class Runner(ppo2.AbstractEnvRunner):
    """
    This is the PPO2 runner, but splitting the sampling and processing stage up
    more explicitly
    """
    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.lam = lam
        self.gamma = gamma
        self.sampler = PPOBatchSampler(env=env, model=model, nsteps=nsteps)

    def sample(self):
        return self.sampler.run()

    def run(self, gamma=.99, lam=.95):
        return self.sample().to_ppo_batch(gamma=self.gamma, lam=self.lam)


class Learner:
    def __init__(self, policy_class, env, *, total_timesteps, nsteps=2048,
                 ent_coef=0.0, lr=3e-4, vf_coef=0.5,  max_grad_norm=0.5,
                 gamma=0.99, lam=0.95, nminibatches=4, noptepochs=4,
                 cliprange=0.2, checkpoint=None, save_env=True):
        # The random seed should already be set before running this

        print(locals())

        total_timesteps = int(total_timesteps)
        batching_config = make_batching_config(
            nenvs=env.num_envs,
            nsteps=nsteps,
            noptepochs=noptepochs,
            nminibatches=nminibatches
        )

        ob_space = env.observation_space
        ac_space = env.action_space

        model_args = dict(
            policy=policy_class, ob_space=ob_space, ac_space=ac_space,
            nbatch_act=batching_config.nenvs,
            nbatch_train=batching_config.nbatch_train,
            nsteps=batching_config.nsteps,
            ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm
        )

        policy = setup_policy(
            model_args=model_args, env=env, nenvs=env.num_envs, checkpoint=checkpoint,
            ob_space=ob_space, ac_space=ac_space, save_env=save_env
        )

        model = policy.model
        runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
        optimizer = PPOOptimizer(policy=policy, batching_config=batching_config)

        # Set our major learner objects
        self.policy = policy
        self.model  = model
        self.runner = runner
        self.optimizer = optimizer

        # Set our last few run configurations
        self.callbacks = []

        # Initialize the objects that will change as we learn
        self._update = 1
        self._epinfobuf = deque(maxlen=100)
        self._tfirststart = None
        self._run_info = None
        self._tnow = None
        self._fps = None
        self._lossvals = None
        self._itr = None

    @property
    def update(self):
        return self._update

    @property
    def mean_reward(self):
        return safemean([epinfo['r'] for epinfo in self._epinfobuf])

    @property
    def mean_length(self):
        return safemean([epinfo['l'] for epinfo in self._epinfobuf])

    def obtain_samples(self, itr):
        # Run the model on the environments
        self._run_info = self.runner.run()
        self._epinfobuf.extend(self._run_info.epinfos)
        self._itr = itr

    def optimize_policy(self, itr):
        assert self._itr == itr
        if not self._tfirststart:
           self._tfirststart = time.time()
        tstart = time.time()

        # Actually do the optimization
        self._lossvals = self.optimizer.optimize_policy(itr, self._run_info)

        self._tnow = time.time()
        self._fps = int(self.optimizer.nbatch / (self._tnow - tstart))

        for check, fn in self.callbacks:
            if check(self.update):
                fn(**locals())

        self._update += 1
        if self._update > self.optimizer.nupdates:
            logger.log("Warning, exceeded planned number of updates")

    def step(self):
        self.obtain_samples(self._update)
        self.optimize_policy(self._update)

    def register_callback(self, check, fn):
        self.callbacks.append((check, fn))

    @staticmethod
    def check_update_interval(freq, include_first=True):
        return lambda i: i % freq == 0 or include_first and i == 1

    @staticmethod
    def print_log(self, **kwargs):
        print_log(
            model=self.model, batching_config=self.optimizer.batching_config,
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

        while self.update < self.optimizer.nupdates:
            self.step()
            if should_yield(self.update):
                yield yield_fn(self)

        yield yield_fn(self)