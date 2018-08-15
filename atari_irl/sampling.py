import numpy as np
from . import utils
from collections import namedtuple


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
        }
        self.is_finalized = False

    def __getitem__(self, key):
        return {
            'observations': self.observations,
            'actions': self.actions,
            'rewards': self.rewards
        }[key]

    def __contains__(self, key):
        return key in {'observations', 'actions', 'rewards'}

    def add_ppo_batch_data(self, obs, act, rew, done, value, neglogpac):
        self.observations.append(obs)
        self.actions.append(act)
        self.rewards.append(rew)
        self.env_infos['dones'].append(done)
        self.agent_infos['values'].append(value)
        self.agent_infos['neglogpacs'].append(neglogpac)

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

    def __get__(self, idx):
        return self.trajectories[idx]

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
        self.obs = obs
        self.rewards = rewards
        self.actions = actions
        self.values = values
        self.dones = dones
        self.neglogpacs = neglogpacs
        self.states = states
        self.epinfos = epinfos
        self.runner = runner

    def to_ppo_batch(self) -> PPOBatch:
        return PPOBatch(*self.runner.process_ppo_samples(
            self.obs, self.rewards, self.actions, self.values, self.dones,
            self.neglogpacs, self.states, self.epinfos
        ))

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
                )

        for traj in buffer:
            traj.finalize()

        return Trajectories(buffer, self)


# This code isn't actually used right now, but it is tested so I'm keeping it
# around
def invert_ppo_sample_raveling(ppo_samples, num_envs=8):
    return ppo_samples.reshape(
        num_envs, ppo_samples.shape[0] // num_envs,
        *ppo_samples.shape[1:]
    ).swapaxes(1, 0)


def ppo_samples_to_trajectory_format(ppo_samples, num_envs=8):
    OBS_IDX = 0
    ACTS_IDX = 3

    obs = invert_ppo_sample_raveling(ppo_samples[OBS_IDX], num_envs=num_envs)
    acts = invert_ppo_sample_raveling(ppo_samples[ACTS_IDX], num_envs=num_envs)

    T = obs.shape[0]
    observations = [[] for _ in range(num_envs)]
    actions = [[] for _ in range(num_envs)]

    assert acts.shape[0] == T
    for t in range(T):
        for i, (o, a) in enumerate(zip(obs[t], acts[t])):
            observations[i].append(o)
            actions[i].append(a)

    trajectories = [
        {
            'observations': np.array(observations[i]),
            'actions': utils.one_hot(actions[i], 6)
        }
        for i in range(num_envs)
    ]
    np.random.shuffle(trajectories)

    return trajectories