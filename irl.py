import tensorflow as tf
import pickle

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy

from airl.algos.irl_trpo import IRLTRPO
from airl.models.airl_state import AIRL as AIRLStateOnly
from airl.utils.log_utils import rllab_logdir

from environments import VecGymEnv


# Heavily based on implementation in https://github.com/HumanCompatibleAI/population-irl/blob/master/pirl/irl/airl.py
def airl(venv, trajectories, discount, seed, log_dir, *,
         tf_cfg, model_cfg=None, policy_cfg=None, training_cfg={}):
    envs = VecGymEnv(venv)
    envs = TfEnv(envs)
    experts = trajectories
    train_graph = tf.Graph()
    with train_graph.as_default():
        tf.set_random_seed(seed)

        if model_cfg is None:
            model_cfg = {'model': AIRLStateOnly, 'state_only': True,
                         'max_itrs': 10}
        if policy_cfg is None:
            policy_cfg = {'policy': GaussianMLPPolicy,
                          'hidden_sizes': (32, 32)}

        model_kwargs = dict(model_cfg)
        model_cls = model_kwargs.pop('model')
        irl_model = model_cls(env_spec=envs.spec, expert_trajs=experts,
                              **model_kwargs)

        policy_fn = policy_cfg.pop('policy')
        policy = policy_fn(name='policy', env_spec=envs.spec, **policy_cfg)

        training_kwargs = {
            'n_itr': 1000,
            'batch_size': 10000,
            'max_path_length': 500,
            'irl_model_wt': 1.0,
            'entropy_weight': 0.1,
            # paths substantially increase storage requirements
            'store_paths': False,
        }
        training_kwargs.update(training_cfg)
        algo = IRLTRPO(
            env=envs,
            policy=policy,
            irl_model=irl_model,
            discount=discount,
            sampler_args=dict(n_envs=venv.num_envs),
            zero_environment_reward=True,
            baseline=LinearFeatureBaseline(env_spec=envs.spec),
            **training_kwargs
        )

        with rllab_logdir(algo=algo, dirname=log_dir):
            with tf.Session(config=tf_cfg):
                algo.train()

                reward_params = irl_model.get_params()

                # Side-effect: forces policy to cache all parameters.
                # This ensures they are saved/restored during pickling.
                policy.get_params()
                # Must pickle policy rather than returning it directly,
                # since parameters in policy will not survive across tf sessions.
                policy_pkl = pickle.dumps(policy)

    reward = model_cfg, reward_params
    return reward, policy_pkl
