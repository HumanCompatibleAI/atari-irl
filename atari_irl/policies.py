import numpy as np
from baselines.ppo2.ppo2 import Model
from . import environments
from .utils import one_hot
import os
import os.path as osp
import joblib


class Policy:
    """
    Lets us save, restore, and step a policy forward

    Plausibly we'll want to start passing a SerializationContext argument
    instead of save_dir, so that we can abstract out the handling of the
    load/save logic, and ensuring that the relevant directories exist.

    If we do that we'll have to intercept the Model's use of joblib,
    maybe using a temporary file.
    """
    model_args_fname = 'model_args.pkl'
    model_fname = 'model' # extension assigned automatically
    annotations_fname = 'annotations.pkl'

    def __init__(self, model_args):
        self.model_args = model_args
        self.model = Model(**model_args)
        self.annotations = {}

    def step(self, obs):
        return self.model.step(obs)

    def save(self, save_dir, **kwargs):
        """
        Saves a policy, along with other annotations about it

        Args:
            save_dir: directory to put the policy in
            **kwargs: annotations that you want to store along with the model
                kind of janky interface, but seems maybe not-that-bad
                canonical thing that we'll want to store is the training reward
        """
        self.annotations.update(kwargs)
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.model_args, osp.join(save_dir, self.model_args_fname))
        self.model.save(osp.join(osp.join(save_dir, self.model_fname)))
        joblib.dump(self.annotations, osp.join(save_dir, self.annotations_fname))

    @classmethod
    def load(cls, save_dir):
        """
        Restore a model completely from storage.

        This needs to be a class method, so that we don't have to initialize
        a policy in order to get the parameters back.
        """
        model_args = joblib.load(osp.join(save_dir, cls.model_args_fname))
        policy = cls(model_args)
        policy.model.load(osp.join(save_dir, cls.model_fname))
        policy.annotations = joblib.load(osp.join(save_dir, cls.annotations_fname))
        return policy


class EnvPolicy(Policy):
    """
    Lets us save and restore a policy where the policy depends on some sort of
    modification to the environment, expressed as an environment wrapper.

    Unfortunately, we still need to initialize the same type of environment
    before we can open it here, so that it has a chance
    """
    env_params_fname = 'env_params.pkl'

    def __init__(self, model_args, envs=None):
        super().__init__(model_args)
        self.envs = envs

    def save(self, save_dir, **kwargs):
        assert self.envs
        super().save(save_dir, **kwargs)
        env_params = environments.serialize_env_wrapper(self.envs)
        joblib.dump(env_params, osp.join(save_dir, self.env_params_fname))

    @classmethod
    def load(cls, save_dir, envs):
        import pickle
        policy = super().load(save_dir)
        # we save the pickle-serialized env_params, so we need pickle to deserialize them
        env_params = pickle.loads(joblib.load(osp.join(save_dir, cls.env_params_fname)))
        policy.envs = environments.restore_serialized_env_wrapper(env_params, envs)
        environments.make_const(policy.envs)
        return policy


def restore_policy_from_checkpoint_dir(checkpoint_dir, envs=None):
    if EnvPolicy.env_params_fname in os.listdir(checkpoint_dir):
        assert envs is not None
        return EnvPolicy.load(checkpoint_dir, envs)
    else:
        return Policy.load(checkpoint_dir)


def run_policy(*, model, environments, render=True):
    # Initialize the stuff we want to keep track of
    rewards = []

    # Initialize our environment
    done = [False]
    obs = np.zeros((environments.num_envs,) + environments.observation_space.shape)
    obs[:] = environments.reset()

    # run the policy until done
    while not any(done):
        actions, _, _, _ = model.step(obs)
        obs[:], reward, done, info = environments.step(actions)
        rewards.append(reward)
        if render:
            environments.render()

    return sum(rewards[:])


def sample_trajectories(*, model, environments, one_hot_code=False, n_trajectories=10, render=False):
    # vectorized environments reset after done
    # pirl format: [
    #    (observations, actions), <- single trajectory same length for both
    #    (observations, actions),
    #    ...
    # ]
    # airl format: [ <- if airl accepts this list, then we're happy
    #    {observations: numpy array, actions: numpy array}, <- single trajectory
    #    {observations: numpy array, actions: numpy array},
    #    ...
    # ]
    # simple simulated robotics can work with 1 trajectory, 5-10 for harder, scales
    # with complexity
    completed_trajectories = []
    observations = [[] for _ in range(environments.num_envs)]
    actions = [[] for _ in range(environments.num_envs)]

    obs = environments.reset()
    while len(completed_trajectories) < n_trajectories:
        acts, _, _, _ = model.step(obs)

        # We append observation, actions tuples here, since they're defined now
        for i, (o, a) in enumerate(zip(obs, acts)):
            observations[i].append(o)
            actions[i].append(a)

        # Figure out our consequences
        obs, _, dones, _ = environments.step(acts)
        if render:
            environments.render()

        # If we're done, then append that trajectory and restart
        for i, done in enumerate(dones):
            if done:
                completed_trajectories.append({
                    'observations': np.array(observations[i]),
                    # TODO(Aaron): get the real dim
                    'actions': one_hot(actions[i], environments.action_space.n) if one_hot_code else np.vstack(actions[i])
                })
                observations[i] = []
                actions[i] = []

    np.random.shuffle(completed_trajectories)
    return completed_trajectories[:n_trajectories]