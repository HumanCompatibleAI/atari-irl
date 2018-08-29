import tensorflow as tf
import numpy as np
import pickle

from rllab.misc import logger
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.misc.overrides import overrides

from sandbox.rocky.tf.envs.base import TfEnv
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.policies.base import StochasticPolicy
from sandbox.rocky.tf.distributions.categorical import Categorical
from sandbox.rocky.tf.spaces.box import Box

from airl.algos.irl_trpo import IRLTRPO
from airl.models.airl_state import AIRL
from airl.utils.log_utils import rllab_logdir
from airl.models.fusion_manager import RamFusionDistr
from airl.utils import TrainingIterator

from baselines.ppo2.policies import CnnPolicy, nature_cnn, fc
from baselines.a2c.utils import conv, fc, conv_to_fc

from .environments import VecGymEnv, wrap_env_with_args, VecRewardZeroingEnv, VecIRLRewardEnv, VecOneHotEncodingEnv
from .utils import one_hot
from . import sampling, training, utils, environments, optimizers, policies

from sandbox.rocky.tf.misc import tensor_utils

import os
import joblib
import time
from collections import defaultdict, namedtuple


class DiscreteIRLPolicy(StochasticPolicy, Serializable):
    """
    Wraps our ppo2-based Policy to fit the interface that AIRL uses.
    """
    def __init__(
            self,
            *,
            name,
            policy_model,
            num_envs,
            env_spec,
            wrapped_env_action_space,
            action_space,
            observation_space,
            batching_config,
            init_location=None
    ):
        Serializable.quick_init(self, locals())
        assert isinstance(wrapped_env_action_space, Box)
        self._dist = Categorical(wrapped_env_action_space.shape[0])

        # this is going to be serialized, so we can't add in the envs or
        # wrappers
        self.init_args = dict(
            name=name,
            policy_model=policy_model,
            init_location=init_location
        )

        ent_coef = 0.0
        vf_coef = 0.5
        max_grad_norm = 0.5
        model_args = dict(
            policy=policy_model,
            ob_space=observation_space,
            ac_space=action_space,
            nbatch_act=batching_config.nenvs,
            nbatch_train=batching_config.nbatch_train,
            nsteps=batching_config.nsteps,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm
        )

        self.num_envs=num_envs

        with tf.variable_scope(name) as scope:
            policy = policies.Policy(model_args)
            self.model = policy.model
            self.act_model = self.model.act_model
            self.scope = scope

        StochasticPolicy.__init__(self, env_spec)
        self.name = name

        self.probs = tf.nn.softmax(self.act_model.pd.logits)
        obs_var = self.act_model.X

        self.tensor_values = lambda **kwargs: tf.get_default_session().run(self.get_params())

        self._f_dist = tensor_utils.compile_function(
            inputs=[obs_var],
            outputs=self.probs
        )

        if init_location:
            data = joblib.load(open(init_location, 'rb'))
            self.restore_from_snapshot(data['policy_params'])

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

    def _get_actions_right_shape(self, observations):
        actions, values, _, neglogpacs = self.act_model.step(observations)
        return (
            one_hot(actions, 6),
            dict(
                prob=self._f_dist(observations),
                values=values.reshape(self.num_envs, 1),
                neglogpacs=neglogpacs.reshape(self.num_envs, 1)
            )
        )

    def get_actions(self, observations):
        N = observations.shape[0]
        batch_size = self.act_model.X.shape[0].value

        final_actions, agent_info = utils.batched_call(
            self._get_actions_right_shape,
            batch_size,
            (observations,)
        )

        for key, value in agent_info.items():
            if len(value.shape) == 2 and value.shape[1] == 1:
                agent_info[key] = value.reshape((value.shape[0],))

        # Integrity checks in case I wrecked this
        assert len(final_actions) == N
        for key in agent_info.keys():
            assert len(agent_info[key]) == N

        return final_actions, agent_info

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

    @property
    def distribution(self):
        return self._dist

    def restore_param_values(self, params):
        param_tensors = self.get_params()
        restores = []
        for tf_tensor, np_array in zip(param_tensors, params):
            restores.append(tf_tensor.assign(np_array))
        tf.get_default_session().run(restores)

    def show_run_in_gym_env(self, venv):
        dones = [False]
        obs = venv.reset()

        while not any(dones):
            actions, _ = self.get_actions(obs)
            obs, reward, dones, _ = venv.step(actions)
            print(reward)
            #venv.render()

    def get_itr_snapshot(self):
        return {
            'config': self.init_args,
            # unfortunately get_params() is already taken...
            'tf_params': self.tensor_values()
        }

    def restore_from_snapshot(self, data):
        """
        Restore a policy from snapshot data.

        Note that this only restores the model part of the policy -- the
        learner doesn't actually get its state repaired, and so for instances
        the step size will be different than it was when you saved.
        """
        for key, value in data['config'].items():
            if self.init_args[key] != value:
                print(f"Warning: different values for {key}")
        self.restore_param_values(data['tf_params'])


def dcgan_cnn(unscaled_images, **conv_kwargs):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.

    def batch_norm(x, name):
        shape = (1, *x.shape[1:])
        with tf.variable_scope(name):
            mean = tf.get_variable('mean', shape, initializer=tf.constant_initializer(0.0))
            variance = tf.get_variable('variance', shape, initializer=tf.constant_initializer(1.0))
            offset = tf.get_variable('offset', shape, initializer=tf.constant_initializer(0.0))
            scale = tf.get_variable('scale', shape, initializer=tf.constant_initializer(1.0))
        return tf.nn.batch_normalization(
            x, mean, variance, offset, scale, 0.001, name
        )

    activ = lambda name, inpt: tf.nn.leaky_relu(batch_norm(inpt, name))

    h = activ('l1', conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
                   **conv_kwargs))
    h2 = activ('l2', conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ('l3', conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(h3)
    return h3


def cnn_net(x, actions=None, dout=1, **conv_kwargs):
    #h = nature_cnn(x, **conv_kwargs)
    h = dcgan_cnn(x, **conv_kwargs)
    if actions is not None:
        assert dout == 1
        action_size = actions.get_shape()[1].value
        print(actions.get_shape())
        selection = fc(h, 'action_selection', nh=action_size, init_scale=np.sqrt(2))

        h = tf.concat([
            tf.multiply(actions, selection),
            actions,
            h
        ], axis=1)
    output = fc(h, 'output', nh=dout, init_scale=np.sqrt(2))
    return output


class AtariAIRL(AIRL):
    """
    This actually fits the AIRL interface! Yay!

    Args:
        fusion (bool): Use trajectories from old iterations to train.
        state_only (bool): Fix the learned reward to only depend on state.
        score_discrim (bool): Use log D - log 1-D as reward (if true you should not need to use an entropy bonus)
        max_itrs (int): Number of training iterations to run per fit step.
    """
    #TODO(Aaron): Figure out what all of these args mean
    def __init__(self, *,
                 env_spec, # No good default, but we do need to have it
                 expert_trajs=None,
                 reward_arch=cnn_net,
                 reward_arch_args={},
                 value_fn_arch=cnn_net,
                 score_discrim=False,
                 discount=1.0,
                 state_only=False,
                 max_itrs=100,
                 fusion=False,
                 name='airl',
                 drop_framestack=False,
                 only_show_scores=False,
                 rescore_expert_trajs=True
                 ):
        super(AIRL, self).__init__()

        # Write down everything that we're going to need in order to restore
        # this. All of these arguments are serializable, so it's pretty easy
        self.init_args = dict(
            model=AtariAIRL,
            env_spec=env_spec,
            expert_trajs=expert_trajs,
            reward_arch=reward_arch,
            reward_arch_args=reward_arch_args,
            value_fn_arch=value_fn_arch,
            score_discrim=score_discrim,
            discount=discount,
            state_only=state_only,
            max_itrs=max_itrs,
            fusion=fusion,
            name=name,
            rescore_expert_trajs=rescore_expert_trajs,
            drop_framestack=drop_framestack,
            only_show_scores=only_show_scores
        )

        if fusion:
            self.fusion = RamFusionDistr(100, subsample_ratio=0.5)
        else:
            self.fusion = None
        self.dO = env_spec.observation_space.flat_dim
        self.dOshape = env_spec.observation_space.shape
        if drop_framestack:
            assert len(self.dOshape) == 3
            self.dOshape = (*self.dOshape[:-1], 1)
        self.dU = env_spec.action_space.flat_dim
        assert isinstance(env_spec.action_space, Box)
        self.score_discrim = score_discrim
        self.gamma = discount
        assert value_fn_arch is not None
        #self.set_demos(expert_trajs)
        self.expert_trajs = expert_trajs
        self.state_only = state_only
        self.max_itrs = max_itrs
        self.drop_framestack = drop_framestack
        self.only_show_scores = only_show_scores
        self.expert_cache = None
        self.rescore_expert_trajs = rescore_expert_trajs
        # build energy model
        with tf.variable_scope(name) as _vs:
            # Should be batch_size x T x dO/dU
            self.obs_t = tf.placeholder(tf.int32, list((None,) + self.dOshape), name='obs')
            self.nobs_t = tf.placeholder(tf.int32, list((None,) + self.dOshape), name='nobs')
            self.act_t = tf.placeholder(tf.float32, [None, self.dU], name='act')
            self.nact_t = tf.placeholder(tf.float32, [None, self.dU], name='nact')
            self.labels = tf.placeholder(tf.float32, [None, 1], name='labels')
            self.lprobs = tf.placeholder(tf.float32, [None, 1], name='log_probs')
            self.lr = tf.placeholder(tf.float32, (), name='lr')

            with tf.variable_scope('discrim') as dvs:
                rew_input = self.obs_t
                with tf.variable_scope('reward'):
                    if self.state_only:
                        self.reward = reward_arch(
                            rew_input, dout=1, **reward_arch_args
                        )
                    else:
                        print("Not state only", self.act_t)
                        self.reward = reward_arch(
                            rew_input, actions=self.act_t,
                            dout=1, **reward_arch_args
                        )
                # value function shaping
                with tf.variable_scope('vfn'):
                    fitted_value_fn_n = value_fn_arch(self.nobs_t, dout=1)
                with tf.variable_scope('vfn', reuse=True):
                    self.value_fn = fitted_value_fn = value_fn_arch(self.obs_t, dout=1)

                # Define log p_tau(a|s) = r + gamma * V(s') - V(s)
                self.qfn = self.reward + self.gamma*fitted_value_fn_n
                log_p_tau = self.reward  + self.gamma*fitted_value_fn_n - fitted_value_fn

            log_q_tau = self.lprobs

            log_pq = tf.reduce_logsumexp([log_p_tau, log_q_tau], axis=0)
            self.discrim_output = tf.exp(log_p_tau-log_pq)
            self.accuracy, self.update_accuracy = tf.metrics.accuracy(
                labels=self.labels,
                predictions=self.discrim_output > 0.5
            )
            self.loss = -tf.reduce_mean(self.labels*(log_p_tau-log_pq) + (1-self.labels)*(log_q_tau-log_pq))
            self.step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            self._make_param_ops(_vs)

            self.grad_reward = tf.gradients(self.reward, [self.obs_t, self.act_t])

            self.modify_obs = self.get_ablation_modifiers()

    def change_kwargs(self, **kwargs):
        for key, value in kwargs.items():
            # All of these are used in graph construction, and so we can't
            # just changing the parameter value won't do anything here
            assert key not in {
                'model',
                'env_spec',
                'dO',
                'dOshape',
                'dU',
                'discount',
                'gamma',
                'drop_framestack',
                'reward_arch',
                'reward_arch_args',
                'value_fn_arch',
                'state_only',
                'name'
            }
            # We have to serialize it
            assert key in self.init_args
            # And here's a whitelist just to be safe
            assert key in {
                'rescore_expert_trajs'
            }
            self.__setattr__(key, value)

    def get_itr_snapshot(self):
        return {
            'config': self.init_args,
            'tf_params': self.get_params()
        }

    def restore_from_snapshot(self, data):
        for key, value in data['config'].items():
            if self.init_args[key] != value:
                print(f"Warning: different values for {key}")
        self.set_params(data['tf_params'])

    def get_ablation_modifiers(self):
        def process_obs(obs):
            if self.drop_framestack:
                obs = obs[:, :, :, -1:]
            if self.only_show_scores:
                obs = obs.copy()
                obs[:, 10:, :, :] *= 0
            return obs
        return process_obs

    @overrides
    def fit(self, paths, policy=None, batch_size=256, logger=None, lr=1e-3, itr=0, **kwargs):
        if isinstance(self.expert_trajs[0], dict):
            self._insert_next_state(self.expert_trajs)
            expert_obs_base, expert_obs_next_base, expert_acts, expert_acts_next = \
                self.extract_paths(self.expert_trajs, keys=(
                    'observations', 'observations_next',
                    'actions', 'actions_next'
                ))
        else:
            expert_obs_base, expert_obs_next_base, expert_acts, expert_acts_next, _ = \
                self.expert_trajs

        #expert_probs = paths.sampler.get_a_logprobs(
        obs, obs_next, acts, acts_next, path_probs = paths.extract_paths((
            'observations', 'observations_next', 'actions', 'actions_next', 'a_logprobs'
        ), obs_modifier=self.modify_obs)

        expert_obs = expert_obs_base
        expert_obs_next = expert_obs_next_base

        # Train discriminator
        for it in TrainingIterator(self.max_itrs, heartbeat=5):
            nobs_batch, obs_batch, nact_batch, act_batch, lprobs_batch = \
                self.sample_batch(obs_next, obs, acts_next, acts, path_probs, batch_size=batch_size)

            nexpert_obs_batch, expert_obs_batch, nexpert_act_batch, expert_act_batch = \
                self.sample_batch(
                    expert_obs_next,
                    expert_obs,
                    expert_acts_next,
                    expert_acts,
                    # expert_probs,
                    batch_size=batch_size
                )
            expert_lprobs_batch = paths.sampler.get_a_logprobs(
                expert_obs_batch,
                expert_act_batch
            )
            expert_obs_batch = self.modify_obs(expert_obs_batch)
            nexpert_obs_batch = self.modify_obs(nexpert_obs_batch)

            # Build feed dict
            labels = np.zeros((batch_size*2, 1))
            labels[batch_size:] = 1.0
            obs_batch = np.concatenate([obs_batch, expert_obs_batch], axis=0)
            nobs_batch = np.concatenate([nobs_batch, nexpert_obs_batch], axis=0)
            act_batch = np.concatenate([act_batch, expert_act_batch], axis=0)
            nact_batch = np.concatenate([nact_batch, nexpert_act_batch], axis=0)
            lprobs_batch = np.expand_dims(np.concatenate([lprobs_batch, expert_lprobs_batch], axis=0), axis=1).astype(np.float32)

            feed_dict = {
                self.act_t: act_batch,
                self.obs_t: obs_batch,
                self.nobs_t: nobs_batch,
                self.nact_t: nact_batch,
                self.labels: labels,
                self.lprobs: lprobs_batch,
                self.lr: lr
            }

            loss, _, acc = tf.get_default_session().run(
                [self.loss, self.step, self.update_accuracy],
                feed_dict=feed_dict
            )
            it.record('loss', loss)
            it.record('accuracy', acc)
            if it.heartbeat:
                print(it.itr_message())
                mean_loss = it.pop_mean('loss')
                print('\tLoss:%f' % mean_loss)
                mean_acc = it.pop_mean('accuracy')
                print('\tAccuracy:%f' % mean_acc)

        if logger:
            logger.record_tabular('GCLDiscrimLoss', mean_loss)
            logger.record_tabular('GCLDiscrimAccuracy', mean_acc)

        return mean_loss

    @overrides
    def eval(self, samples, show_grad=False, **kwargs):
        """
        Return bonus
        """
        if self.score_discrim:
            obs, obs_next, acts, path_probs = samples.extract_paths(
                ('observations', 'observations_next', 'actions', 'a_logprobs'),
                obs_modifier=self.modify_obs
            )
            path_probs = np.expand_dims(path_probs, axis=1)
            scores = tf.get_default_session().run(
                self.discrim_output,
                feed_dict={
                    self.act_t: acts,
                    self.obs_t: obs,
                    self.nobs_t: obs_next,
                    self.lprobs: path_probs
                }
            )
            score = np.log(scores) - np.log(1-scores)
            score = score[:,0]
        else:
            obs, acts = samples.extract_paths(
                ('observations', 'actions'), obs_modifier=self.modify_obs
            )
            reward = tf.get_default_session().run(
                self.reward, feed_dict={self.act_t: acts, self.obs_t: obs}
            )
            score = reward[:,0]
        # TODO(Aaron, maybe): do something with show_grad
        return samples._ravel_train_batch_to_time_env_batch(score)


def policy_config(
        name='policy',
        policy=DiscreteIRLPolicy,
        policy_model=CnnPolicy,
        init_location=None
):
    return dict(
        name=name,
        policy=policy,
        policy_model=policy_model,
        init_location=init_location
    )


def reward_model_config(
        *,
        # These are serializable, but also there's no reasonably default value
        # so we have to provide it
        env_spec,
        expert_trajs,
        ablation='none',
        model=AtariAIRL,
        state_only=False,
        reward_arch=cnn_net,
        value_fn_arch=cnn_net,
        score_discrim=True,
        max_itrs=10,
        drop_framestack=False,
        only_show_scores=False
):
    return dict(
        model=model,
        state_only=state_only,
        reward_arch=reward_arch,
        value_fn_arch=value_fn_arch,
        env_spec=env_spec,
        expert_trajs=expert_trajs,
        score_discrim=score_discrim,
        max_itrs=max_itrs,
        drop_framestack=drop_framestack,
        only_show_scores=only_show_scores
    )


def training_config(
        n_itr=1000,
        discount=0.99,
        batch_size=5000,
        max_path_length=100,
        entropy_weight=0.01,
        step_size=0.01,
        irl_model_wt=1.0,
        zero_environment_reward=True,
):
    return dict(
        n_itr=n_itr,
        discount=discount,
        batch_size=batch_size,
        max_path_length=max_path_length,
        entropy_weight=entropy_weight,
        step_size=step_size,
        store_paths=False,
        irl_model_wt=irl_model_wt,
        zero_environment_reward=zero_environment_reward
    )


def make_irl_policy(policy_cfg, *, wrapped_venv, baselines_venv):
    policy_fn = policy_cfg.pop('policy')
    policy = policy_fn(
        num_envs=baselines_venv.num_envs,
        env_spec=wrapped_venv.spec,
        wrapped_env_action_space=wrapped_venv.action_space,
        action_space=baselines_venv.action_space,
        observation_space=baselines_venv.observation_space,
        **policy_cfg
    )

    return policy


def make_irl_model(model_cfg):
    model_kwargs = dict(model_cfg)
    model_cls = model_kwargs.pop('model')
    return model_cls(**model_kwargs)


Ablation = namedtuple('Ablation', [
    'policy_modifiers', 'discriminator_modifiers', 'training_modifiers'
])


def get_ablation_modifiers(*, irl_model, ablation):
    irl_reward_wrappers = [
        wrap_env_with_args(VecRewardZeroingEnv),
        #wrap_env_with_args(VecIRLRewardEnv, reward_network=irl_model)
    ]

    # Default to wrapping the environment with the irl rewards
    ablations = defaultdict(lambda: Ablation(
        policy_modifiers={'baseline_wrappers': irl_reward_wrappers},
        discriminator_modifiers={},
        training_modifiers={}
    ))
    ablations['train_rl'] = Ablation(
        policy_modifiers={'baseline_wrappers': []},
        discriminator_modifiers={},
        training_modifiers={
            'irl_model_wt': 0.0,
            # TODO(Aaron): Figure out if this should be false...
            'zero_environment_reward': True,
            'skip_discriminator': True
        }
    )
    ablations['train_discriminator'] = Ablation(
        policy_modifiers={'baseline_wrappers': []},
        discriminator_modifiers={
            'rescore_expert_trajs': False
        },
        training_modifiers={
            'skip_policy_update': True
        }
    )
    ablations['run_expert'] = Ablation(
        policy_modifiers={'baseline_wrappers': []},
        discriminator_modifiers={},
        training_modifiers={
            'irl_model_wt': 0.0,
            'zero_envvironment_reward': True,
            'skip_discriminator': True,
            'skip_policy_update': True
        }
    )
    return ablations[ablation]


def add_ablation(cfg, ablation_modifiers):
    for key in ablation_modifiers.keys():
        if key in cfg and cfg[key] != ablation_modifiers[key]:
            print(
                f"Warning: Overriding provided value {cfg[key]} "
                f"for {key} with {ablation_modifiers[key]} for ablation"
            )
    cfg.update(ablation_modifiers)
    return cfg


def rllab_wrap_venv(envs):
    return TfEnv(VecGymEnv(envs))


def get_training_kwargs(
        *,
        venv,
        ablation='normal', nsteps_sampler=2048, nsteps_model=128,
        reward_model_cfg={}, policy_cfg={}, training_cfg={}
):
    envs = rllab_wrap_venv(venv)

    policy_cfg = policy_config(**policy_cfg)
    reward_model_cfg = reward_model_config(
        env_spec=envs.spec,
        ablation=ablation,
        **reward_model_cfg
    )
    training_cfg = training_config(**training_cfg)

    # Unfortunately we need to construct a reward model in order to handle the
    # ablations, since in the normal case we need to pass it as an argument to
    # the policy in order to wrap its environment and look at the irl rewards
    irl_model = make_irl_model(reward_model_cfg)

    # Handle the ablations and default value overrides
    ablation_modifiers = get_ablation_modifiers(
        irl_model=irl_model, ablation=ablation
    )

    irl_model.change_kwargs(**ablation_modifiers.discriminator_modifiers)

    # Construct our fixed training keyword arguments. Other values for these
    # are incorrect
    baseline_wrappers = ablation_modifiers.policy_modifiers.pop(
        'baseline_wrappers'
    )

    baselines_venv = venv
    for fn in baseline_wrappers + [wrap_env_with_args(VecOneHotEncodingEnv, dim=6)]:
        print("Wrapping baseline with function")
        baselines_venv = fn(baselines_venv)

    baselines_venv = baselines_venv

    assert nsteps_sampler % nsteps_model == 0

    batching_config = training.make_batching_config(
        nenvs=baselines_venv.num_envs,
        nsteps=nsteps_model,
        noptepochs=4,
        nminibatches=4
    )
    policy_cfg['batching_config'] = batching_config

    training_kwargs = dict(
        env=envs,
        policy=make_irl_policy(
            add_ablation(policy_cfg, ablation_modifiers.policy_modifiers),
            wrapped_venv=envs,
            baselines_venv=baselines_venv
        ),
        irl_model=irl_model,
        baseline=ZeroBaseline(env_spec=envs.spec),
        ablation=ablation,
        sampler_args=dict(
            baselines_venv=baselines_venv,
            nsteps=nsteps_sampler,
            gamma=0.99,
            lam=0.95
        ),
        optimizer_args=dict(
            batching_config=batching_config,
            lr=3e-4,
            cliprange=0.2,
            total_timesteps=10e6
        )
    )
    training_kwargs.update(
        add_ablation(training_cfg, ablation_modifiers.training_modifiers)
    )

    if policy_cfg['init_location']:
        snapshot = joblib.load(open(policy_cfg['init_location'], 'rb'))
        training_kwargs['init_pol_params'] = snapshot['policy_params']['tf_params']

    return training_kwargs, policy_cfg, reward_model_cfg, training_cfg


class IRLRunner(IRLTRPO):
    """
    This takes over the IRLTRPO code, to actually run IRL. Right now it has a
    few issues...
    [ ] It doesn't share the sample buffer between the discriminator and policy
    [ ] It doesn't work on MuJoCo
    """
    def __init__(
            self,
            *args,
            ablation='none',
            skip_policy_update=False,
            skip_discriminator=False,
            optimizer=None,
            optimizer_args={},
            **kwargs
    ):
        if optimizer is None:
            optimizer = optimizers.PPOOptimizer(**optimizer_args)
        IRLTRPO.__init__(self, *args, optimizer=optimizer, **kwargs)
        self.ablation = ablation
        self.skip_policy_update = skip_policy_update
        self.skip_discriminator = skip_discriminator

    @overrides
    def init_opt(self):
        self.optimizer.update_opt(self.policy)

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            ablation=self.ablation,
            reward_params=self.irl_model.get_itr_snapshot(),
            policy_params=self.policy.get_itr_snapshot()
        )

    def restore_from_snapshot(self, snapshot):
        data = joblib.load(open(snapshot, 'rb'))
        self.irl_model.restore_from_snapshot(data['reward_params'])
        self.policy.restore_from_snapshot(data['policy_params'])

    @overrides
    def obtain_samples(self, itr):
        return super(IRLRunner, self).obtain_samples(itr)

    @overrides
    def optimize_policy(self, itr, samples):
        self.optimizer.optimize_policy(itr, samples)

    @overrides
    def compute_irl(self, samples, itr=0):
        if self.no_reward:
            logger.record_tabular(
                'EnvironmentBatchAverageReturn', np.mean(samples.rewards)
            )
            logger.record_tabular(
                'OriginalTaskAverageReturn', samples.sampler.mean_reward
            )
            samples.rewards *= 0

        if self.irl_model_wt <=0:
            return samples

        if self.train_irl:
            max_itrs = self.discrim_train_itrs
            lr=1e-3
            mean_loss = self.irl_model.fit(
                samples,
                policy=self.policy, itr=itr, max_itrs=max_itrs, lr=lr,
                logger=logger
            )

            logger.record_tabular('IRLLoss', mean_loss)
            self.__irl_params = self.irl_model.get_params()

        probs = self.irl_model.eval(samples, gamma=self.discount, itr=itr)
        probs_flat = np.concatenate(probs)  # trajectory length varies

        logger.record_tabular('IRLRewardMean', np.mean(probs_flat))
        logger.record_tabular('IRLRewardMax', np.max(probs_flat))
        logger.record_tabular('IRLRewardMin', np.min(probs_flat))

        if self.irl_model_wt > 0.0:
            samples.rewards += self.irl_model_wt * probs

        return samples

    def train(self):
        sess = tf.get_default_session()
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        if self.init_pol_params is not None:
            self.policy.restore_param_values(self.init_pol_params)
        if self.init_irl_params is not None:
            self.irl_model.set_params(self.init_irl_params)
        self.start_worker()
        start_time = time.time()

        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            with logger.prefix('itr #%d | ' % itr):
                logger.record_tabular('Itr', itr)

                logger.log("Obtaining samples...")
                samples = self.obtain_samples(itr)

                if not self.skip_discriminator:
                    logger.log("Optimizing discriminator...")
                    # The fact that we're not using the reward labels from here
                    # means that the policy optimization is effectively getting
                    # an off-by-one issue. I'm not sure that this would fix the
                    # issues that we're seeing, but it's definitely different
                    # from the original algorithm and so we should try fixing
                    # it anyway.
                    samples = self.compute_irl(samples, itr=itr)

                if not self.skip_policy_update:
                    logger.log("Optimizing policy...")
                    # Another issue is that the expert trajectories start from
                    # a particular set of random seeds, and that controls how
                    # the resets happen. This means that the difference between
                    # environment seeds might be enough to make the
                    # discriminator's job easy.
                    self.optimize_policy(itr, samples)

                logger.record_tabular(
                    "PolicyBufferOriginalTaskRewardMean",
                    self.sampler.mean_reward
                )
                logger.record_tabular(
                    "PolicyBufferEpisodeLengthMean",
                    self.sampler.mean_length
                )

                if itr % 10 == 0:
                    logger.log("Saving snapshot...")
                    params = self.get_itr_snapshot(itr, samples)  # , **kwargs)
                    if self.store_paths:
                        raise NotImplementedError
                    logger.save_itr_params(itr, params)
                    logger.log(f"Saved in directory {logger.get_snapshot_dir()}")

                logger.record_tabular('Time', time.time() - start_time)
                logger.record_tabular('ItrTime', time.time() - itr_start_time)
                logger.dump_tabular(with_prefix=False)
                if self.plot:
                    self.update_plot()
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                              "continue...")
        return


class IRLContext:
    def __init__(self, tf_cfg, env_config):
        self.tf_cfg = tf_cfg
        self.env_config = env_config
        self.seed = env_config['seed']

        env_modifiers = environments.env_mapping[env_config['env_name']]
        one_hot_code = env_config.pop('one_hot_code')
        if one_hot_code:
            env_modifiers = environments.one_hot_wrap_modifiers(env_modifiers)
        self.env_config.update(env_modifiers)

    def __enter__(self):
        self.env_context = utils.EnvironmentContext(**self.env_config)
        self.env_context.__enter__()
        self.train_graph = tf.Graph()
        self.tg_context = self.train_graph.as_default()
        self.tg_context.__enter__()
        self.sess = tf.Session(config=self.tf_cfg)
        # from tensorflow.python import debug as tf_debug
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess , ui_type='readline')
        self.sess_context = self.sess.as_default()
        self.sess_context.__enter__()
        tf.set_random_seed(self.seed)

        return self

    def __exit__(self, *args):
        self.sess_context.__exit__(*args)
        self.tg_context.__exit__(*args)
        self.env_context.__exit__(*args)


# Heavily based on implementation in https://github.com/HumanCompatibleAI/population-irl/blob/master/pirl/irl/airl.py
def airl(
        log_dir,
        *,
        tf_cfg, env_config, reward_model_cfg={}, policy_cfg={}, training_cfg={},
        ablation='normal'
):
    with IRLContext(tf_cfg, env_config) as context:
        training_kwargs, policy_cfg, reward_model_cfg, training_cfg = get_training_kwargs(
            venv=context.env_context.environments,
            reward_model_cfg=reward_model_cfg,
            policy_cfg=policy_cfg,
            training_cfg=training_cfg,
            ablation=ablation,
        )
        print("Training arguments: ", training_kwargs)
        algo = IRLRunner(
            **training_kwargs,
            sampler_cls=sampling.PPOBatchSampler,
        )
        irl_model = algo.irl_model
        policy = algo.policy

        with rllab_logdir(algo=algo, dirname=log_dir):
            print("Training!")
            algo.train()
            reward_params = irl_model.get_params()
            # Must pickle policy rather than returning it directly,
            # since parameters in policy will not survive across tf sessions.
            policy_params = policy.tensor_values()

    policy = policy_cfg, policy_params
    reward = reward_model_cfg, reward_params
    return reward, policy
