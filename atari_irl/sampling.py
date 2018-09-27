import numpy as np
import tensorflow as tf
from . import utils
from collections import namedtuple, deque

from rllab.misc.overrides import overrides
from rllab.sampler.base import BaseSampler
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler

from baselines.ppo2 import ppo2

"""
Heavily based on the ppo2 implementation found in the OpenAI baselines library,
particularly in the PPOSampler class.
"""

# This is a PPO Batch that the OpenAI Baselines PPO code uses as its underlying
# representation
PPOBatch = namedtuple('PPOBatch', [
    'obs', 'returns', 'masks', 'actions', 'values', 'neglogpacs', 'states',
    'epinfos'
])
PPOBatch.train_args = lambda self: (
    self.obs, self.returns, self.masks, self.actions, self.values, self.neglogpacs
)


# This is a full trajectory, using the interface defines by RLLab and the
# AIRL library
class Trajectory:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.env_infos = {
            'dones': []
        }
        self.agent_infos = {
            'values': [],
            'neglogpacs': [],
            'prob': []
        }
        self.is_finalized = False

        self.added_data = {}

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return self.added_data[key]

    def __setitem__(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.added_data[key] = value

    def __contains__(self, key):
        return hasattr(self, key) or key in self.added_data

    def add_ppo_batch_data(self, obs, act, rew, done, value, neglogpac, prob):
        self.observations.append(obs)
        self.actions.append(act)
        self.rewards.append(rew)
        self.env_infos['dones'].append(done)
        self.agent_infos['values'].append(value)
        self.agent_infos['neglogpacs'].append(neglogpac)
        self.agent_infos['prob'].append(prob)

        assert np.isclose(1.0, prob.sum())

    def finalize(self):
        assert not self.is_finalized
        self.observations = np.asarray(self.observations)
        self.actions = utils.one_hot(self.actions, 6)
        self.rewards = np.asarray(self.rewards)
        self.is_finalized = True


class Trajectories:
    def __init__(self, trajectories, ppo_sample=None):
        self.trajectories = trajectories
        self.ppo_sample = ppo_sample

    def __getitem__(self, idx):
        return self.trajectories[idx]

    def __len__(self):
        return len(self.trajectories)

    def to_ppo_sample(self) -> 'PPOSample':
        # This is kind of hacky, and ideally we would roundtrip it, but doing
        # a proper roundtrip is going to be slower than doing this
        if self.ppo_sample is not None:
            return self.ppo_sample


class PPOSample:
    """
    A trajectory slice generated according to the PPO batch logic.

    This can be transformed into both a PPOBatch (for PPO training), and a
    Trajectories object which doesn't actually correspond to full trajectories,
    (for the discriminator training)
    """
    def __init__(
            self, obs, rewards, actions, values, dones, neglogpacs, states,
            epinfos, sampler
    ):
        self.obs = np.asarray(obs)
        self.rewards = np.asarray(rewards)
        self.returns = rewards # match PPOBatch
        self.actions = np.asarray(actions)
        self.values = np.asarray(values)
        self.dones = np.asarray(dones)
        self.neglogpacs = np.asarray(neglogpacs)
        self.states = states
        self.epinfos = epinfos

        self.sampler = sampler

        self.sample_batch_timesteps = self.obs.shape[0]
        self.sample_batch_num_envs = self.obs.shape[1]
        self.sample_batch_size = (
            self.sample_batch_timesteps * self.sample_batch_num_envs
        )
        self.train_batch_size = self.sampler.model.train_model.X.shape[0].value
        assert self.sample_batch_size % self.train_batch_size == 0

        self.obs_next = None
        self.actions_next = None

        self.probabilities = self._get_sample_probabilities()

    def to_ppo_batches(self, batch_size):
        all_data = self.sampler.process_to_ppo_batch(
            self, gamma=self.sampler.gamma, lam=self.sampler.lam
        )
        if all_data.states is not None:
            raise NotImplemented

        N = all_data.obs.shape[0]
        assert N % batch_size == 0
        for start in range(0, N, batch_size):
            end = start + batch_size
            yield PPOBatch(
                all_data.obs[start:end],
                all_data.returns[start:end],
                all_data.masks[start:end],
                all_data.actions[start:end],
                all_data.values[start:end],
                all_data.neglogpacs[start:end],
                None,
                None
            )

    def to_ppo_batch(self):
        return self.sampler.process_to_ppo_batch(
            self, gamma=self.sampler.gamma, lam=self.sampler.lam
        )

    def _ravel_time_env_batch_to_train_batch(self, inpt):
        assert inpt.shape[0] == self.sample_batch_timesteps
        assert inpt.shape[1] == self.sample_batch_num_envs

        num_train_batches = self.sample_batch_size // self.train_batch_size

        # change the first index into environments, not timesteps
        ans = inpt.swapaxes(0, 1
        # reshape first indices into # of batches x train batch size
        ).reshape(
            num_train_batches, self.train_batch_size, *inpt.shape[2:]
        )
        return ans

    def _ravel_train_batch_to_time_env_batch(self, inpt):
        # reshape things into number of envs x number of timesteps
        ans = inpt.reshape(
            self.sample_batch_num_envs,
            self.sample_batch_timesteps,
            *inpt.shape[2:]
        # swap the timesteps back into the first index
        ).swapaxes(0, 1)
        assert ans.shape[0] == self.sample_batch_timesteps
        assert ans.shape[1] == self.sample_batch_num_envs
        return ans

    def _get_sample_probabilities(self):
        train_batched_obs = self._ravel_time_env_batch_to_train_batch(self.obs)
        ps = np.asarray([
            # we weirdly don't have direct access to the probabilities anywhere
            # so we need to construct this node from teh logits
            self.sampler.get_probabilities_for_obs(train_batch_obs)
            for train_batch_obs in train_batched_obs
        ])
        return self._ravel_train_batch_to_time_env_batch(ps)

    def to_trajectories(self) -> 'Trajectories':
        T = len(self.obs)
        num_envs = self.obs[0].shape[0]
        buffer = [Trajectory() for _ in range(num_envs)]

        for t in range(T):
            for e in range(num_envs):
                buffer[e].add_ppo_batch_data(
                    self.obs[t][e],
                    self.actions[t][e],
                    self.rewards[t][e],
                    self.dones[t][e],
                    self.values[t][e],
                    self.neglogpacs[t][e],
                    self.probabilities[t][e]
                )

        for traj in buffer:
            traj.finalize()

        return Trajectories(buffer, self)

    def get_path_key(self, key, pad_val=0.0):
        if key == 'observations':
            return self.obs
        elif key == 'actions':
            return self.actions
        elif key == 'observations_next':
            if self.obs_next is None:
                obs = self.obs
                self.obs_next = np.r_[
                    obs[1:],
                    pad_val*np.expand_dims(np.ones_like(obs[0]), axis=0)
                ]
            return self.obs_next
        elif key == 'actions_next':
            if self.actions_next is None:
                self.actions_next = np.r_[
                    self.actions[1:],
                    pad_val*np.expand_dims(np.ones_like(self.actions[0]), axis=0)
                ]
            return self.actions_next
        elif key == 'a_logprobs':
            """
            alogprobs = self.sampler.get_a_logprobs(
                ppo2.sf01(self.obs),
                utils.one_hot(ppo2.sf01(self.actions).astype(np.int32), 6)
            )
            assert np.isclose(
                ppo2.sf01(-1 * self.neglogpacs),
                alogprobs
            ).all()
            """
            return -1 * self.neglogpacs
        else:
            raise NotImplementedError

    def extract_paths(self, keys, obs_modifier=lambda obs, *args: obs):
        data = [
            ppo2.sf01(self.get_path_key(key))
            for key in keys
        ]

        def process_data(inpt):
            key, value = inpt
            if 'actions' in key:
                return utils.one_hot(value.astype(np.int32), 6)
            elif 'observations' in key:
                return obs_modifier(value, key=key, sample=self)
            else:
                return value
        return map(process_data, zip(keys, data))


class DummyAlgo:
    def __init__(self, policy):
        self.policy = policy


class PPOBatchSampler(BaseSampler, ppo2.AbstractEnvRunner):
    # If you want to use the baselines PPO sampler as a sampler for the
    # airl interfaced code, use this.
    def __init__(self, algo, *, nsteps, baselines_venv, gamma=0.99, lam=0.95):
        model = algo.policy.model
        env = baselines_venv
        # The biggest weird thing about this piece of code is that it does a
        # a bunch of work to handle the context of what happens if the model
        # that we're training is actually recurrent.
        # This means that we store the observations, states, and dones so that
        # we can continue a run.
        # We have not actually tested that functionality
        ppo2.AbstractEnvRunner.__init__(self, env=env, model=model, nsteps=nsteps)
        self.algo = algo
        self.env = env
        self.model = model
        self.nsteps = nsteps
        self.gamma = gamma
        self.lam = lam
        self.cur_sample = None
        self._epinfobuf = deque(maxlen=100)

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def run(self):
        return self._sample()

    def get_probabilities_for_obs(self, obs):
        tm = self.model.train_model
        if obs.shape[1:] != self.env.observation_space.shape and self.env.venv.encoder:
            obs = self.env.venv.encoder.base_vector(obs)
        return tf.get_default_session().run(
            tf.nn.softmax(tm.pd.logits),
            {tm.X: obs}
        )

    def get_a_logprobs(self, obs, acts):
        probs = utils.batched_call(
            # needs to be a tuple for the batched call to work
            lambda obs: (self.get_probabilities_for_obs(obs),),
            self.model.train_model.X.shape[0].value,
            (obs, ),
            check_safety=False
        )[0]
        return np.log((probs * acts).sum(axis=1))

    def _sample(self) -> PPOSample:
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

            should_show = False and np.random.random() > .95

            if should_show:
                print()
                obs_summaries = '\t'.join([f"{o[0,0,0]}{o[0,-1,0]}" for o in self.obs])
                act_summaries = '\t'.join([str(a) for a in actions])
                print(f"State:\t{obs_summaries}")
                print(f"Action:\t{act_summaries}")

            self.obs[:], rewards, self.dones, infos = self.env.step(actions)

            if should_show:
                rew = '\t'.join(['{:.3f}'.format(r) for r in rewards])
                print(f"Reward:\t{rew}")
                print()


            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)

            mb_rewards.append(rewards)

        self._epinfobuf.extend(epinfos)
        return PPOSample(
            mb_obs, mb_rewards, mb_actions, mb_values, mb_dones,
            mb_neglogpacs, mb_states, epinfos, self
        )

    def _process_ppo_samples(
            self, *,
            obs, rewards, actions, values, dones, neglogpacs,
            states, epinfos,
            gamma, lam
    ) -> PPOBatch:
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(rewards, dtype=np.float32)
        mb_actions = np.asarray(actions)
        mb_values = np.asarray(values, dtype=np.float32)
        mb_dones = np.asarray(dones, dtype=np.bool)
        mb_neglogpacs = np.asarray(neglogpacs, dtype=np.float32)
        last_values = self.model.value(self.obs, self.states, self.dones)
        #discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return PPOBatch(
            *map(
                ppo2.sf01,
                (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)
            ),
            states,
            epinfos
        )

    def process_to_ppo_batch(
            self, ppo_sample: PPOSample, *, gamma: float, lam: float
    ) -> PPOBatch:
        return self._process_ppo_samples(
            obs=ppo_sample.obs,
            rewards=ppo_sample.rewards,
            actions=ppo_sample.actions,
            values=ppo_sample.values,
            dones=ppo_sample.dones,
            neglogpacs=ppo_sample.neglogpacs,
            states=ppo_sample.states,
            epinfos=ppo_sample.epinfos,
            gamma=gamma, lam=lam
        )

    def process_trajectory(self, traj, *, gamma, lam):
        def batch_reshape(single_traj_data):
            single_traj_data = np.asarray(single_traj_data)
            s = single_traj_data.shape
            return single_traj_data.reshape(s[0], 1, *s[1:])

        agent_info = traj['agent_infos']

        # These are trying to deal with the fact that the PPOSampler maintains
        # an internal state that it uses to remember information between
        # trajectories, which is important if you have a recurrent policy
        self.state = None
        self.obs[:] = traj['observations'][-1]
        self.dones = np.ones(self.env.num_envs)
        dones = np.zeros(agent_info['values'].shape)
        # This is a full trajectory, so it's a bunch of not-done, followed by
        # a single done
        dones[-1] = 1

        # This is actually kind of weird w/r/t to the PPO code, because the
        # batch length is so much longer. Maybe this will work? But if PPO
        # ablations don't crash, but fail to learn this is probably why.
        return self._process_ppo_samples(
            # The easy stuff that we just observe
            obs=batch_reshape(traj['observations']),
            rewards=np.hstack([batch_reshape(traj['rewards']) for _ in range(8)]),
            actions=batch_reshape(traj['actions']).argmax(axis=1),
            # now the things from the agent info
            values=agent_info['values'],
            dones=dones,
            neglogpacs=agent_info['neglogpacs'],
            states=None, # recurrent trajectories should include states
            # and the annotations
            epinfos=traj['env_infos'],
            gamma=gamma, lam=lam
        )

    def obtain_samples(self, itr):
        self.cur_sample = self._sample()
        return self.cur_sample

    def process_samples(self, itr, paths):
        ppo_batch = self.cur_sample.to_ppo_batch()
        self._epinfobuf.extend(ppo_batch.epinfos)
        return ppo_batch

    @property
    def mean_reward(self):
        return ppo2.safemean([epinfo['r'] for epinfo in self._epinfobuf])

    @property
    def mean_length(self):
        return ppo2.safemean([epinfo['l'] for epinfo in self._epinfobuf])


class FullTrajectorySampler(VectorizedSampler):
    # If you want to use the RLLab sampling code with a baselines-interfaced
    # policy, use this.
    @overrides
    def process_samples(self, itr, paths):
        """
        We need to go from paths to PPOBatch shaped samples. This does it in a
        way that's pretty hacky and doesn't crash, but isn't overall promising,
        because when you tune the PPO hyperparameters to look at single full
        trajectories that doesn't work well either
        """
        print("Processing samples, albeit hackily!")
        samples_data = self.algo.policy.learner.runner.process_trajectory(
            paths[0]
        )
        T = samples_data[0].shape[0]
        return PPOBatch(
            *([data[:512] for data in samples_data[:-2]] + [None, None])
        )


class PPOBatchBuffer(PPOSample):
    def __init__(self, ppo_sample, n_batches):
        self.n_batches = n_batches
        self.cur_idx = 0

        T = ppo_sample.obs.shape[0]
        assert ppo_sample.rewards.shape[0] == T
        assert ppo_sample.actions.shape[0] == T
        assert ppo_sample.values.shape[0] == T
        assert ppo_sample.dones.shape[0] == T
        assert ppo_sample.neglogpacs.shape[0] == T

        self.batch_T = T

        def fix_shape(shape):
            return (self.batch_T * self.n_batches, ) + shape[1:]

        super().__init__(
            np.zeros(fix_shape(ppo_sample.obs.shape)),
            np.zeros(fix_shape(ppo_sample.rewards.shape)),
            np.zeros(fix_shape(ppo_sample.actions.shape)),
            np.zeros(fix_shape(ppo_sample.values.shape)),
            np.zeros(fix_shape(ppo_sample.dones.shape)),
            np.zeros(fix_shape(ppo_sample.neglogpacs.shape)),
            None,
            None,
            ppo_sample.sampler
        )

    def add(self, sample):
        if self.cur_idx >= self.n_batches * self.batch_T:
            self.cur_idx = 0

        for key in ['obs', 'rewards', 'actions', 'values', 'dones', 'neglogpacs']:
            s = slice(self.cur_idx, self.cur_idx + self.batch_T)
            getattr(self, key)[s] = getattr(sample, key)

        self.cur_idx += self.batch_T
        
    def to_ppo_batches(self, batch_size):
        for start in range(0, self.batch_T * self.n_batches, batch_size):
            end = start + batch_size
            s = slice(start, end)
            
            yield self.sampler._process_ppo_samples(
                obs=self.obs[s],
                rewards=self.rewards[s],
                actions=self.actions[s],
                values=self.values[s],
                dones=self.dones[s],
                neglogpacs=self.neglogpacs[s],
                states=None, epinfos=None,
                gamma=self.sampler.gamma,
                lam=self.sampler.lam
            )