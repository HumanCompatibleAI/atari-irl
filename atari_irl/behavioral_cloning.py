import tensorflow as tf
import numpy as np

from baselines.ppo2.policies import CnnPolicy
from baselines import logger

from . import policies, training
import pickle
from collections import namedtuple

"""
Training Code

This should all get moved/factored into training.py of we actually like how
it works, but for now it's specific to behavioral cloning
"""
# This controls all of the training information which don't change the input
# shapes
TrainingConfig = namedtuple('TrainingConfig', [
    'learning_rate', 'max_grad_norm', 'optimizer'
])

def make_train_node(*, params, loss, training_config):
    """
    Creates the training node

    Args:
        params: tensorflow nodes for the policy that we can train
        loss: tensorflow node for the loss to optimize
        training_config: config for the training

    Returns:
        _train, a tensorflow operation which trains the policy when we run it
    """
    grads = tf.gradients(loss, params)

    if training_config.max_grad_norm is not None:
        grads, _grad_norm = tf.clip_by_global_norm(
            grads, training_config.max_grad_norm
        )
    trainer = tf.train.AdamOptimizer(
        learning_rate=training_config.learning_rate, epsilon=1e-5
    )

    grads = list(zip(grads, params))
    _train = trainer.apply_gradients(grads)
    return _train

def make_dagger_imitation_cost(
        *,
        policy,
        params,
        training_config,
        sess
):
    """
    Set up the cost function for DAGGER

    This is intended to make the code more modular by breaking up the creation
    of a trainable model into a few different components:
    - The Policy (a baselines ppo2 policy), which maps inputs to actions and
      probabilities
    - The TrainablePolicy (see below), which wraps a policy and sets up what
      you need in order to train it
    - A cost function, which defines what's actually minimized when you train
        # construct the loss and start the training function
        self.loss, self.train = make_cost(
            policy=self.policy,
            params=params,
            training_config=training_config,
            sess=self.sess
        )
    Thankfully, tensorflow makes it easy to set up a policy + train it, and
    baselines gives us a bunch of policies, so the first 2 are pretty easy.

    The policy interface exposes a few things, and we should restrict

 ourselves
    to only using things which are genuinely exposed
    - X: The input
    - vf: The Value Function
    - pd: The probability distribution
    - pdtype: The type of the probability distribution

    Args:
        policy: Policy that we're trying to optimize
        params: Parameters for the above policy
            it seems like we could figure them out in this code, but it felt
            safer to just grab these after initializing the policy and passing
            them in, rather than needing to worry about whatever context we
            may or may not be in in the future
        training_config: Configuration spec for training
        sess: tensorflow session to use for everything

    Returns:
        loss: The tensorflow node for the actual loss
        train: A function accepting the appropriate inputs that trains
    """
    # Placeholder for actions
    actions_placeholder = policy.pdtype.sample_placeholder([None])
    # Compute the loss based on the policy's probability distribution
    loss = tf.reduce_mean(policy.pd.neglogp(actions_placeholder))

    _train = make_train_node(
        params=params, loss=loss, training_config=training_config
    )

    def train(observations, actions, log=False):
        """
        Train our policy based on observed MDP observations and actions

        Args:
            observations: observations for the trajectory
            actions: actions chosen in the trajectory
            log: whether or not to log information for this training

        Returns:
            mean negative log likelihood
        """
        td_map = {
            policy.X: observations,
            actions_placeholder: actions
        }
        mean_nll, _ = sess.run([loss, _train], td_map)
        if log:
            logger.logkv("mean nll of actions")
        return mean_nll

    return loss, train


class TrainablePolicy:
    def __init__(
            self, *,
            policy_class, make_cost, name='clone',
            sess, envs, batching_config, training_config
    ):
        self.batching_config = batching_config
        self.envs = envs
        self.sess = sess
        self.make_cost = make_cost

        # Set up the actual tensorflow graph for our policy
        with tf.variable_scope(name):
            kwargs = dict(
                sess=tf.get_default_session(),
                ob_space=envs.observation_space,
                ac_space=envs.action_space,
                nsteps=batching_config.nsteps,
            )
            self.act_policy = policy_class(
                **kwargs, nbatch=envs.num_envs, reuse=False
            )
            self.train_policy = policy_class(
                **kwargs, nbatch=batching_config.nbatch_train, reuse=True
            )
            # It's super important that this is scoped, otherwise we'll include
            # tensorflow nodes that aren't actually part of the trainable
            # policy
            params = tf.trainable_variables(scope=name)

            # construct the loss and start the training function
            self.loss, self.train = make_cost(
                policy=self.train_policy,
                params=params,
                training_config=training_config,
                sess=self.sess
            )

            # If we do the globals initializer we'll overwrite whatever we
            # defined already. That would be bad, so we just only initialize
            # the variables actually associated with the policy
            for p in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name):
                self.sess.run(p.initializer)

        self.step = self.act_policy.step



def sample_trajectories(policy, environments, batch_size):
    """
    Sample batch_size time steps worth of trajectories from the policy.

    This assumes that we're running in an atari environment, getting densely
    coded actions, otherwise vstack and hstack are the wrong operations.

    The policy's observation placeholder has to have length environment number,
    otherwise this will all crash

    Args:
        policy: policy to sample from
        environments: environments to sample in
        batch_size: number of timesteps to get

    Returns:
        observations: batch_size x 84 x 84 x 4 array
        actions: batch_size length vector containing actions
    """
    observations = []
    actions = []
    obs = environments.reset()
    t = 0
    while t < batch_size:
        acts, _, _, _ = policy.step(obs)
        t += environments.num_envs

        observations.append(obs)
        actions.append(acts)

        obs, _, _, _ = environments.step(acts)
    return np.vstack(observations), np.hstack(a for a in actions)


def update_policy(policy_clone, obs, acts, n_iter=10):
    """
    Update the cloned policy based on the given trajectories and
    batching config by fitting to the data n_iter times

    Args:
        policy_clone: Policy clone that we're trying to train
        obs: observations from our trajectory buffer
        acts: actions from our trajectory buffer
        batching_config: batching config to follow

    Returns:
        nothing
    """
    assert acts.shape[0] == obs.shape[0]
    indices = np.arange(obs.shape[0])

    batching_config = policy_clone.batching_config
    for i in range(n_iter):
        loss_vals = []
        for _ in range(batching_config.noptepochs):
            np.random.shuffle(indices)
            for start in range(0, batching_config.nbatch, batching_config.nbatch_train):
                end = start + batching_config.nbatch_train
                epoch_indices = indices[start:end]
                loss_vals.append(
                    policy_clone.train(obs[epoch_indices], acts[epoch_indices])
                )
        logger.log("{}: {}".format(i, np.mean(loss_vals)))


def actions_for_trajectory_observations(expert_policy, obs):
    """
    Get the actions for a trajectory of observations

    Note that the observations can come in any order, so we're excluding
    RNN-based policies from considerations.

    Args:
        expert_policy: policy to generate action labels from
        obs: observations to label

    Returns:
        actions: a length len(obs) vector of densely coded actions
    """
    actions = []
    nbatch = expert_policy.envs.num_envs
    for start in range(0, len(obs), nbatch):
        end = start + nbatch
        acts, _, _, _ = expert_policy.step(obs[start:end])
        actions.append(acts)
    return np.hstack(a for a in actions)


class CloningContext:
    def __init__(
            self, policy, envs, policy_class, *,
            start_trajectories=None, policy_name='clone',
            nsteps=2048, noptepochs=4, nminibatches=32, n_trajectories=10
    ):
        self.base_policy = policy
        self.envs = envs

        self.n_trajectories = n_trajectories
        self.base_trajectories = start_trajectories
        self.batching_config = training.make_batching_config(
            env=envs,
            nsteps=nsteps,
            noptepochs=noptepochs,
            nminibatches=nminibatches
        )
        self.training_config = TrainingConfig(
            learning_rate=3e-4,
            max_grad_norm=0.5,
            optimizer=tf.train.AdamOptimizer
        )

        self.policy_clone = TrainablePolicy(
            policy_class=policy_class,
            envs=envs,
            name=policy_name,
            make_cost=make_dagger_imitation_cost,
            sess=tf.get_default_session(),
            batching_config=self.batching_config,
            training_config=self.training_config
        )

    def __enter__(self):
        if self.base_trajectories is None:
            self.base_trajectories = sample_trajectories(
                self.base_policy, self.envs, self.batching_config.nbatch
            )
        else:
            self.base_trajectories = pickle.load(
                open(self.base_trajectories, 'rb')
            )
        return self

    def __exit__(self, *args):
        pass


def dagger(
        policy, envs, policy_class, *, total_timesteps,
        nsteps=2048, noptepochs=4, nminibatches=32, n_trajectories=10
):
    with CloningContext(
            policy, envs, policy_class,
            nsteps=nsteps, noptepochs=noptepochs, nminibatches=nminibatches,
            n_trajectories=n_trajectories
    ) as context:
        observations, actions = context.base_trajectories

        for t in range(total_timesteps // context.batching_config.nbatch):
            # Update the policy clone
            update_policy(context.policy_clone, observations, actions, 10)

            # Sample the trajectories from our cloned policy
            observations_t, _ = sample_trajectories(
                context.policy_clone, envs, context.batching_config.nbatch
            )

            # Label the sampled trajectories with the expert actions
            actions_t = actions_for_trajectory_observations(
                context.base_policy, observations
            )

            observations = np.vstack([observations, observations_t])
            actions      = np.vstack([actions, actions_t])
            logger.logkv(t)


def imitate(
        policy, envs, policy_class, *, n_iter=500,
        nsteps=2048, noptepochs=4, nminibatches=32, n_trajectories=10
):
    with CloningContext(
            policy, envs, policy_class,
            nsteps=nsteps, noptepochs=noptepochs, nminibatches=nminibatches,
            n_trajectories=n_trajectories
    ) as context:
        observations, actions = context.base_trajectories
        # Update the policy clone
        update_policy(
            context.policy_clone, observations, actions, n_iter=n_iter
        )