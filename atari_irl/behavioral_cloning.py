import tensorflow as tf

from baselines.ppo2.policies import CnnPolicy

from . import policies, training


def update_policy(policy, trajectories):
    pass


def relabel_trajectory_observations(policy, trajectory):
    return trajectory


def dagger(
        policy, envs, policy_class,
        *,
        total_timesteps, nsteps=2048, noptepochs=4, nminibatches=32
):
    batching_config = training.make_batching_config(
        env=envs,
        nsteps=nsteps, noptepochs=noptepochs, nminibatches=nminibatches
    )
    policy_clone = policy_class(
        sess=tf.get_default_session(),
        ob_space=envs.observation_space,
        ac_space=envs.action_space,
        nbatch=batching_config.nbatch,
        nsteps=batching_config.nsteps
    )
    trajectories = policies.sample_trajectories(
        model=policy.model, environments=envs, n_trajectories=10
    )

    for t in range(total_timesteps // batching_config.nbatch):
        update_policy(policy_clone, trajectories)
        cloned_policy_trajectories_t = policies.sample_trajectories(
            model=policy_clone, environments=envs, n_trajectories=10
        )
        trajectories += relabel_trajectory_observations(
            cloned_policy_trajectories_t
        )

