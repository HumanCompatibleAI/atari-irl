import tensorflow as tf
import numpy as np
import pickle

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.policies.base import StochasticPolicy
from sandbox.rocky.tf.core.layers_powered import LayersPowered
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.distributions.categorical import Categorical
from sandbox.rocky.tf.spaces.box import Box
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
from rllab.misc.overrides import overrides

from airl.algos.irl_trpo import IRLTRPO
from airl.models.airl_state import AIRL as AIRLStateOnly
from airl.utils.log_utils import rllab_logdir
from airl.models.imitation_learning import DIST_CATEGORICAL

from baselines.ppo2.policies import CnnPolicy
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_input
from baselines.a2c.utils import fc
from baselines.ppo2.policies import nature_cnn

from environments import VecGymEnv
from utils import one_hot

from policies import EnvPolicy
from sandbox.rocky.tf.misc import tensor_utils


from tensorflow.python import debug as tf_debug


class DiscreteIRLPolicy(StochasticPolicy, Serializable):
    def __init__(
            self,
            name,
            *,
            policy_model,
            envs,
            sess
    ):
        Serializable.quick_init(self, locals())
        assert isinstance(envs.action_space, Box)
        self._dist = Categorical(envs.action_space.shape[0])

        baselines_venv = envs._wrapped_env.venv
        with tf.variable_scope(name) as scope:
            self.act_model = policy_model(
                sess,
                baselines_venv.observation_space,
                baselines_venv.action_space,
                None,
                1,
                reuse=False
            )

            self.scope = scope

        self.probs = tf.nn.softmax(self.act_model.pd.logits)
        obs_var = self.act_model.X
        #dist_info_sym = self.dist_info_sym(None, None)
        self._f_dist = tensor_utils.compile_function(
            inputs=[obs_var],
            outputs=self.probs
        )

        StochasticPolicy.__init__(self, envs.spec)
        self.name = name

    @property
    def vectorized(self):
        return True

    def dist_info_sym(self, obs_var, state_info_vars=None):
        return dict(prob=self.probs)

    @overrides
    def get_action(self, observation):
        obs = np.array([observation])
        action, _, _, _ = self.act_model.step(obs)
        # TODO(Aaron) get the real dim
        return one_hot(action, 6), dict(prob=self._f_dist(obs))

    def get_actions(self, observations):
        actions, _, _, _ = self.act_model.step(observations)
        return one_hot(actions, 6), dict(prob=self._f_dist(observations))

    def get_params_internal(self, **tags):
        return self.scope.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    def get_reparam_action_sym(self, obs_var, action_var, old_dist_info_vars):
        """
        Given observations, old actions, and distribution of old actions, return a symbolically reparameterized
        representation of the actions in terms of the policy parameters
        :param obs_var:
        :param action_var:
        :param old_dist_info_vars:
        :return:
        """
        raise NotImplemented

    def log_diagnostics(self, paths):
        pass
        #log_stds = np.vstack([path["agent_infos"]["log_std"] for path in paths])
        #logger.record_tabular('AveragePolicyStd', np.mean(np.exp(log_stds)))

    @property
    def distribution(self):
        return self._dist

from tensorflow.python import debug as tf_debug


# Heavily based on implementation in https://github.com/HumanCompatibleAI/population-irl/blob/master/pirl/irl/airl.py
def airl(venv, trajectories, discount, seed, log_dir, *,
         tf_cfg, model_cfg=None, policy_cfg=None, training_cfg={}):
    envs = VecGymEnv(venv)
    envs = TfEnv(envs)
    experts = trajectories
    train_graph = tf.Graph()
    with train_graph.as_default():
        sess = tf.Session(config=tf_cfg)
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess , ui_type='readline')
        with sess.as_default():
            tf.set_random_seed(seed)
            if model_cfg is None:
                model_cfg = {'model': AIRLStateOnly, 'state_only': True,
                             'max_itrs': 10}
            if policy_cfg is None:
                policy_cfg = {'policy': DiscreteIRLPolicy,
                              'policy_model': CnnPolicy,
                              'envs': envs,
                              'sess': sess}

            model_kwargs = dict(model_cfg)
            model_cls = model_kwargs.pop('model')
            irl_model = model_cls(env_spec=envs.spec, expert_trajs=experts,
                                  **model_kwargs)

            policy_fn = policy_cfg.pop('policy')
            policy = policy_fn(name='policy', **policy_cfg)

            training_kwargs = {
                'n_itr': 1000,
                'batch_size': 1000,
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
                baseline=ZeroBaseline(env_spec=envs.spec),
                **training_kwargs
            )

            with rllab_logdir(algo=algo, dirname=log_dir):
                algo.train()

                reward_params = irl_model.get_params()

                # Must pickle policy rather than returning it directly,
                # since parameters in policy will not survive across tf sessions.
                policy_params = sess.run(policy.get_params())

    reward = model_cfg, reward_params
    return reward, policy_params
