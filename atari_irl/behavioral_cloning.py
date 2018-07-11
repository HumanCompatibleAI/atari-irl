import tensorflow as tf

from baselines.ppo2.policies import CnnPolicy

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

    Thankfully, tensorflow makes it easy to set up a policy + train it, and
    baselines gives us a bunch of policies, so the first 2 are pretty easy.

    The policy interface exposes a few things, and we should restrict ourselves
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
    pass

class TrainablePolicy:
    def __init__(
            self, *,
            policy_class, make_cost, name='clone',
            sess, envs, batching_config, training_config
    ):
        self.batching_config = batching_config
        self.sess = sess
        self.make_cost = make_cost

        # Set up the actual tensorflow graph for our policy
        with tf.variable_scope(name):
            self.policy = policy_class(
                sess=tf.get_default_session(),
                ob_space=envs.observation_space,
                ac_space=envs.action_space,
                nbatch=batching_config.nbatch,
                nsteps=batching_config.nsteps
            )
            params = tf.trainable_variables()

            # construct the loss and start the training function
            self.loss, self.train = make_cost(
                policy=self.policy,
                params=params,
                training_config=training_config,
                sess=self.sess
            )

def update_policy(policy, trajectories):
    pass

def relabel_trajectory_observations(policy, trajectory):
    return trajectory


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
            policy_class=CnnPolicy,
            envs=envs,
            make_cost=make_dagger_imitation_cost,
            sess=tf.get_default_session(),
            batching_config=self.batching_config,
            training_config=self.training_config
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

