import tensorflow as tf

from baselines.ppo2.policies import CnnPolicy

from . import policies, training
import pickle


def update_policy(policy, trajectories):
    pass


def relabel_trajectory_observations(policy, trajectory):
    return trajectory


class CloningContext:
    def __init__(
            self, policy, envs, policy_class, *,
            start_trajectories=None,
            nsteps=2048, noptepochs=4, nminibatches=32, n_trajectories=10
    ):
        self.base_policy = policy
        self.envs = envs

        self.n_trajectories = n_trajectories
        self.base_trajectories = start_trajectories
        self.batching_config = training.make_batching_config(
            env=envs,
            nsteps=nsteps, noptepochs=noptepochs, nminibatches=nminibatches
        )

        self.policy_clone = policy_class(
            sess=tf.get_default_session(),
            ob_space=envs.observation_space,
            ac_space=envs.action_space,
            nbatch=self.batching_config.nbatch,
            nsteps=self.batching_config.nsteps
        )

    def __enter__(self):
        if self.base_trajectories is None:
            self.base_trajectories = policies.sample_trajectories(
                model=self.base_policy, environments=self.envs,
                n_trajectories=self.n_trajectories
            )
        else:
            self.base_trajectories = pickle.load(open(self.base_trajectories, 'rb'))
        return self

    def __exit__(self, *args):
        pass


def dagger(
        policy, envs, policy_class, *, total_timesteps,
        nsteps=2048, noptepochs=4, nminibatches=32, n_trajectories=10,
):
    with CloningContext(
            policy, envs, policy_class,
            nsteps=nsteps, noptepochs=noptepochs, nminibatches=nminibatches,
            n_trajectories=n_trajectories
    ) as context:
        trajectories = context.base_trajectories.copy()
        for t in range(total_timesteps // context.batching_config.nbatch):
            update_policy(context.policy_clone, trajectories)
            cloned_policy_trajectories_t = policies.sample_trajectories(
                model=context.policy_clone,
                environments=envs,
                n_trajectories=n_trajectories
            )
            trajectories += relabel_trajectory_observations(
                cloned_policy_trajectories_t
            )

