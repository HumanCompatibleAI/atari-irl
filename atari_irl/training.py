import numpy as np

from baselines import logger
from baselines.common import explained_variance
from baselines.ppo2.ppo2 import constfn, safemean
from baselines.ppo2 import ppo2

from rllab.sampler.base import BaseSampler

from . import policies

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
    'nbatch', 'nbatch_train', 'noptepochs', 'nenvs', 'nsteps', 'nminibatches'
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
    print(safemean([epinfo['r'] for epinfo in epinfobuf]))
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


def make_batching_config(*,env, nsteps, noptepochs, nminibatches):
    nenvs = env.num_envs
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    return BatchingConfig(
        nbatch=nbatch, nbatch_train=nbatch_train, noptepochs=noptepochs,
        nenvs=nenvs, nsteps=nsteps, nminibatches=nminibatches
    )


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


class Runner(ppo2.AbstractEnvRunner, BaseSampler):
    """
    This is the PPO2 runner, but splitting the sampling and processing stage up
    more explicitly
    """
    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.lam = lam
        self.gamma = gamma

    def set_algo(self, algo):
        self.algo = algo

    def sample(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)

        return mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, mb_states, epinfos

    def process_ppo_samples(
            self, mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs,
            mb_states, epinfos
    ):
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(ppo2.sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)

    def process_trajectory(self, traj):
        def batch_reshape(single_traj_data):
            single_traj_data = np.asarray(single_traj_data)
            s = single_traj_data.shape
            return single_traj_data.reshape(s[0], 1, *s[1:])

        agent_info = traj['agent_infos']
        self.state = None
        self.obs[:] = traj['observations'][-1]
        self.dones = np.ones(self.env.num_envs)
        dones = np.zeros(agent_info['values'].shape)
        dones[-1] = 1
        # This is actually kind of weird w/r/t to the PPO code, because the
        # batch length is so much longer. Maybe this will work? But if PPO
        # ablations don't crash, but fail to learn this is probably why.
        return self.process_ppo_samples(
            # The easy stuff that we just observe
            batch_reshape(traj['observations']),
            np.hstack([batch_reshape(traj['rewards']) for _ in range(8)]),
            batch_reshape(traj['actions']).argmax(axis=1),
            # now the things from the agent info
            agent_info['values'], dones, agent_info['neglogpacs'],
            # and the annotations
            None, traj['env_infos']
        )

    def run(self):
        return self.process_ppo_samples(*self.sample())


class Learner:
    def __init__(self, policy_class, env, *, total_timesteps, nsteps=2048,
                 ent_coef=0.0, lr=3e-4, vf_coef=0.5,  max_grad_norm=0.5,
                 gamma=0.99, lam=0.95, nminibatches=4, noptepochs=4,
                 cliprange=0.2, checkpoint=None, save_env=True):
        # The random seed should already be set before running this

        print(locals())

        # Deal with constant arguments
        if isinstance(lr, float): lr = constfn(lr)
        else: assert callable(lr)
        if isinstance(cliprange, float): cliprange = constfn(cliprange)
        else: assert callable(cliprange)

        self.lr = lr
        self.cliprange = cliprange

        total_timesteps = int(total_timesteps)

        ob_space = env.observation_space
        ac_space = env.action_space

        batching_config = make_batching_config(
            env=env, nsteps=nsteps, noptepochs=noptepochs,
            nminibatches=nminibatches
        )

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

        # Set our major learner objects
        self.batching_config = batching_config
        self.policy = policy
        self.model  = model
        self.runner = runner

        # Set our last few run configurations
        self.callbacks = []
        self.nbatch = batching_config.nbatch
        self.nupdates = total_timesteps // batching_config.nbatch

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

    def obtain_samples(self, itr):
        # Run the model on the environments
        self._run_info = RunInfo(*self.runner.run())
        self._epinfobuf.extend(self._run_info.epinfos)
        self._itr = itr
        #import pdb; pdb.set_trace()

    def optimize_policy(self, itr):
        assert self._itr == itr

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