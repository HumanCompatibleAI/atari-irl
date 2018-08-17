import numpy as np
import tensorflow as tf
from . import utils, training
from collections import namedtuple

from rllab.misc.overrides import overrides
from rllab.sampler.base import BaseSampler
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler

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
            epinfos, runner
    ):
        self.obs = np.asarray(obs)
        self.rewards = np.asarray(rewards)
        self.actions = np.asarray(actions)
        self.values = np.asarray(values)
        self.dones = np.asarray(dones)
        self.neglogpacs = np.asarray(neglogpacs)
        self.states = states
        self.epinfos = epinfos
        self.runner = runner

        self.sample_batch_timesteps = self.obs.shape[0]
        self.sample_batch_num_envs = self.obs.shape[1]
        self.sample_batch_size = (
            self.sample_batch_timesteps * self.sample_batch_num_envs
        )
        self.train_batch_size = self.runner.model.train_model.X.shape[0].value
        assert self.sample_batch_size % self.train_batch_size == 0

        self.probabilities = self._get_sample_probabilities()

    def to_ppo_batch(self) -> PPOBatch:
        return PPOBatch(*self.runner.process_ppo_samples(
            self.obs, self.rewards, self.actions, self.values, self.dones,
            self.neglogpacs, self.states, self.epinfos
        ))

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
        sess = tf.get_default_session()
        tm = self.runner.model.train_model
        ps = np.asarray([
            # we weirdly don't have direct access to the probabilities anywhere
            # so we need to construct this node from teh logits
            sess.run(tf.nn.softmax(tm.pd.logits), {tm.X: train_batch_obs})
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


class PPOBatchSampler(BaseSampler):
    # If you want to use the baselines PPO sampler as a sampler for the
    # airl interfaced code, use this.
    def __init__(self, algo):
        super(PPOBatchSampler, self).__init__(algo)
        assert isinstance(algo.policy.learner, training.Learner)
        self.cur_sample = None

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def obtain_samples(self, itr):
        self.cur_sample = self.algo.policy.learner.runner.sample()
        samples = self.cur_sample.to_trajectories()
        self.algo.irl_model._insert_next_state(samples)
        return samples

    def process_samples(self, itr, paths):
        ppo_batch = self.cur_sample.to_ppo_batch()
        self.algo.policy.learner._run_info = ppo_batch
        self.algo.policy.learner._epinfobuf.extend(ppo_batch.epinfos)
        return ppo_batch


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