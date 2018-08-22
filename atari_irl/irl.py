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
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler

from airl.algos.irl_trpo import IRLTRPO
from airl.models.airl_state import AIRL
from airl.utils.log_utils import rllab_logdir
from airl.models.fusion_manager import RamFusionDistr
from airl.models import architectures
from airl.utils.math_utils import gauss_log_pdf, categorical_log_pdf

from baselines.ppo2.policies import CnnPolicy, nature_cnn, fc

from .environments import VecGymEnv, wrap_env_with_args, VecRewardZeroingEnv, VecIRLRewardEnv, VecOneHotEncodingEnv
from .utils import one_hot
from . import sampling, training, utils, environments

from sandbox.rocky.tf.misc import tensor_utils

import os
import joblib
import time
from collections import defaultdict, namedtuple

DIST_GAUSSIAN = 'gaussian'
DIST_CATEGORICAL = 'categorical'


class DiscreteIRLPolicy(StochasticPolicy, Serializable):
    """
    This lets us wrap our PPO + old training code to almost fit into the
    the IRLBatchPolOpt framework. Unfortunately, it currently conflates a few
    things because of a bunch of shape issues between MuJoCo environments and
    Atari environments.
    - It has some of the responsibilities of the Policy
        that is correct and should stay
    - It has some of the responsibilities of the RLAlgorithm (TRPO for instance)
        we should figure out how to split that out of here
        in particular, training.Learner already implements optimize_policy
        which depends on a private _run_info buffer, that seems like we just
        need to reshape and it should be good
        -> this means that we implement PPOOptimizer

    The good news is that the IRL code doesn't actually look at any of the
    shapes, so we should be able to move back to the actual IRLBatchPolOpt
    interface by breaking some things out.
    """
    def __init__(
            self,
            *,
            name,
            policy_model,
            envs,
            baseline_wrappers
    ):
        Serializable.quick_init(self, locals())
        assert isinstance(envs.action_space, Box)
        self._dist = Categorical(envs.action_space.shape[0])

        # this is going to be serialized, so we can't add in the envs or
        # wrappers
        self.init_args = dict(
            name=name,
            policy_model=policy_model
        )

        baselines_venv = envs._wrapped_env.venv
        for fn in baseline_wrappers:
            print("Wrapping baseline with function")
            baselines_venv = fn(baselines_venv)

        self.baselines_venv = baselines_venv
        print("Environment: ", self.baselines_venv)

        with tf.variable_scope(name) as scope:
            self.learner = training.Learner(
                policy_class=policy_model,
                env=baselines_venv,
                total_timesteps=10e6,
                vf_coef=0.5, ent_coef=0.01,
                nsteps=128, noptepochs=4, nminibatches=4,
                gamma=0.99, lam=0.95,
                lr=lambda alpha: alpha * 2.5e-4,
                cliprange=lambda alpha: alpha * 0.1
            )
            self.act_model = self.learner.model.act_model
            self.scope = scope

        StochasticPolicy.__init__(self, envs.spec)
        self.name = name

        self.probs = tf.nn.softmax(self.act_model.pd.logits)
        obs_var = self.act_model.X

        self.tensor_values = lambda **kwargs: tf.get_default_session().run(self.get_params())

        self._f_dist = tensor_utils.compile_function(
            inputs=[obs_var],
            outputs=self.probs
        )

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
        import pdb; pdb.set_trace()
        actions, values, _, neglogpacs = self.act_model.step(observations)
        return (
            actions,
            dict(
                prob=self._f_dist(observations),
                values=values.reshape(self.baselines_venv.num_envs, 1),
                neglogpacs=neglogpacs.reshape(self.baselines_venv.num_envs, 1)
            )
        )

    def get_actions(self, observations):
        N = observations.shape[0]
        batch_size = self.act_model.X.shape[0].value

        # Things get super slow if we don't do this
        if N == batch_size:
            return self._get_actions_right_shape(observations)

        actions = []
        obs = []
        infos = []
        start = 0

        def add_observation_batch(obs_batch, subslice=None):
            batch_actions, batch_info = self._get_actions_right_shape(obs_batch)

            if subslice:
                batch_actions = batch_actions[subslice]
                batch_info = dict(
                    (key, batch_info[key][subslice])
                    for key in batch_info.keys()
                )
                obs_batch = obs_batch[subslice]

            actions.append(batch_actions)
            obs.append(obs_batch)
            infos.append(batch_info)

        for start in range(0, N-batch_size, batch_size):
            end = start + batch_size
            add_observation_batch(observations[start:end])

        start += batch_size
        if start != N:
            remainder_slice = slice(start - N, batch_size)
            add_observation_batch(
                observations[N-batch_size:N],
                subslice=remainder_slice
            )

        # Note: If we change the shape a bunch this will make us sad
        final_actions = np.vstack(actions)
        final_obs = np.vstack(obs)

        agent_info_keys = infos[0].keys()
        for info in infos:
            assert agent_info_keys == info.keys()

        agent_info = dict(
            (key, np.vstack([info[key] for info in infos]))
            for key in agent_info_keys
        )
        for key, value in agent_info.items():
            if len(value.shape) == 2 and value.shape[1] == 1:
                agent_info[key] = value.reshape((value.shape[0],))

        # Integrity checks in case I wrecked this
        assert len(final_actions) == N
        for key in agent_info_keys:
            assert len(agent_info[key]) == N
        # This checks that our observations survived the roundtrip of being
        # sliced + rearranged with everything else
        assert np.isclose(final_obs, observations).all()

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


def cnn_net(x, actions=None, dout=1, **conv_kwargs):
    h = nature_cnn(x, **conv_kwargs)
    activ = tf.nn.relu
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


def relu_net(x, actions=None, **kwargs):
    rew_input = x
    if actions is not None:
        rew_input = tf.concat([x, actions], axis=1)
    return architectures.relu_net(rew_input, **kwargs)

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
                 name='airl'):
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
            name=name
        )

        if fusion:
            self.fusion = RamFusionDistr(100, subsample_ratio=0.5)
        else:
            self.fusion = None
        self.dO = env_spec.observation_space.flat_dim
        self.dOshape = env_spec.observation_space.shape
        self.dU = env_spec.action_space.flat_dim
        assert isinstance(env_spec.action_space, Box)
        self.score_discrim = score_discrim
        self.gamma = discount
        assert value_fn_arch is not None
        self.set_demos(expert_trajs)
        self.state_only=state_only
        self.max_itrs=max_itrs

        # build energy model
        with tf.variable_scope(name) as _vs:
            # Should be batch_size x T x dO/dU
            self.obs_t = tf.placeholder(tf.float32, list((None,) + self.dOshape), name='obs')
            self.nobs_t = tf.placeholder(tf.float32, list((None,) + self.dOshape), name='nobs')
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
            cent_loss = -tf.reduce_mean(self.labels*(log_p_tau-log_pq) + (1-self.labels)*(log_q_tau-log_pq))

            self.loss = cent_loss
            tot_loss = self.loss
            self.step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(tot_loss)
            self._make_param_ops(_vs)

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


def policy_config(name='policy', policy=DiscreteIRLPolicy, policy_model=CnnPolicy):
    return dict(name=name, policy=policy, policy_model=policy_model)


def reward_model_config(
        *,
        # These are serializable, but also there's no reasonably default value
        # so we have to provide it
        env_spec,
        expert_trajs,
        model=AtariAIRL,
        state_only=True,
        reward_arch=cnn_net,
        value_fn_arch=cnn_net,
        score_discrim=False
):
    return dict(
        model=model,
        state_only=state_only,
        reward_arch=reward_arch,
        value_fn_arch=value_fn_arch,
        env_spec=env_spec,
        expert_trajs=expert_trajs,
        score_discrim=score_discrim
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


def make_irl_policy(policy_cfg, *, envs, baseline_wrappers=None):
    policy_fn = policy_cfg.pop('policy')
    blah = policy_cfg.pop('policy_model')
    return policy_fn(
        #envs=envs,
        #baseline_wrappers=baseline_wrappers,
        env_spec=envs.spec,
        **policy_cfg
    )


def make_irl_model(model_cfg):
    model_kwargs = dict(model_cfg)
    model_cls = model_kwargs.pop('model')
    return model_cls(**model_kwargs)


Ablation = namedtuple('Ablation', [
    'policy_modifiers', 'training_modifiers'
])


def get_ablation_modifiers(*, irl_model, ablation):
    irl_reward_wrappers = [
        wrap_env_with_args(VecRewardZeroingEnv),
        wrap_env_with_args(VecIRLRewardEnv, reward_network=irl_model)
    ]

    # Default to wrapping the environment with the irl rewards
    ablations = defaultdict(lambda: Ablation(
        policy_modifiers={'baseline_wrappers': irl_reward_wrappers},
        training_modifiers={}
    ))
    ablations['train_rl'] = Ablation(
        policy_modifiers={'baseline_wrappers': []},
        training_modifiers={'irl_model_wt': 0.0, 'zero_environment_reward': True}
    )

    return ablations[ablation]


def add_ablation(cfg, ablation_modifiers):
    for key in ablation_modifiers.keys():
        if key in cfg:
            print(
                f"Warning: Overriding provided value {cfg[key]}"
                f"for {key} with {ablation_modifiers[key]} for ablation"
            )
    cfg.update(ablation_modifiers)
    return cfg


def get_training_kwargs(
        *,
        venv,
        ablation='normal', one_hot_code=True,
        reward_model_cfg={}, policy_cfg={}, training_cfg={}
):
    envs = venv
    envs = VecGymEnv(envs)
    envs = TfEnv(envs)

    policy_cfg = policy_config(**policy_cfg)
    reward_model_cfg = reward_model_config(
        env_spec=envs.spec,
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

    # Construct our fixed training keyword arguments. Other values for these
    # are incorrect
    baseline_wrappers = ablation_modifiers.policy_modifiers.pop(
        'baseline_wrappers'
    )
    if one_hot_code:
        baseline_wrappers += [wrap_env_with_args(VecOneHotEncodingEnv, dim=6)]
    training_kwargs = dict(
        env=envs,
        policy=make_irl_policy(
            add_ablation(policy_cfg, ablation_modifiers.policy_modifiers),
            envs=envs, baseline_wrappers=baseline_wrappers
        ),
        irl_model=irl_model,
        baseline=ZeroBaseline(env_spec=envs.spec),
        ablation=ablation,
    )
    training_kwargs.update(
        add_ablation(training_cfg, ablation_modifiers.training_modifiers)
    )

    return training_kwargs, policy_cfg, reward_model_cfg, training_cfg


class IRLRunner(IRLTRPO):
    """
    This takes over the IRLTRPO code, to actually run IRL. Right now it has a
    few issues...
    [ ] It doesn't share the sample buffer between the discriminator and policy
    [ ] It doesn't work on MuJoCo
    """
    def __init__(self, *args, ablation='none', **kwargs):
        IRLTRPO.__init__(self, *args, **kwargs)
        self.ablation = ablation
        self.skip_policy_update = self.ablation == 'train_discriminator'
        self.skip_discriminator = self.ablation == 'train_rl'

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

    def obtain_samples(self, itr):
        paths = super(IRLRunner, self).obtain_samples(itr)
        negs = 0
        poss = 0
        for i in range(len(paths)):
            negs += (paths[i]['rewards'] == -1).sum()
            poss += (paths[i]['rewards'] == 1).sum()

        logger.record_tabular("PointsGained", poss)
        logger.record_tabular("PointsLost", negs)
        logger.record_tabular("NumTimesteps", sum([len(p['rewards']) for p in paths]))
        return paths

    @overrides
    def optimize_policy(self, itr, samples_data):
        self.policy.learner._itr = itr
        self.policy.learner._run_info = samples_data
        self.policy.learner.optimize_policy(itr)
        logger.record_tabular(
            "PolicyBufferOriginalTaskRewardMean",
            self.policy.learner.mean_reward
        )
        logger.record_tabular(
            "PolicyBufferEpisodeLengthMean",
            self.policy.learner.mean_length
        )

    def train(self):
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        if self.init_pol_params is not None:
            self.policy.set_param_values(self.init_pol_params)
        if self.init_irl_params is not None:
            self.irl_model.set_params(self.init_irl_params)
        self.start_worker()
        start_time = time.time()

        returns = []
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            with logger.prefix('itr #%d | ' % itr):
                logger.record_tabular('Itr', itr)

                logger.log("Obtaining samples...")
                paths = self.obtain_samples(itr)

                logger.log("Processing samples...")
                if not self.skip_discriminator:
                    paths = self.compute_irl(paths, itr=itr)
                returns.append(self.log_avg_returns(paths))
                samples_data = self.process_samples(itr, paths)

                logger.log("Optimizing policy...")
                if not self.skip_policy_update:
                    self.optimize_policy(itr, samples_data)

                if itr % 10 == 0:
                    logger.log("Saving snapshot...")
                    params = self.get_itr_snapshot(itr, samples_data)  # , **kwargs)
                    if self.store_paths:
                        params["paths"] = samples_data["paths"]
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
    one_hot_code = env_config['one_hot_code']
    with IRLContext(tf_cfg, env_config) as context:
        training_kwargs, policy_cfg, reward_model_cfg, training_cfg = get_training_kwargs(
            venv=context.env_context.environments,
            one_hot_code=one_hot_code,
            reward_model_cfg=reward_model_cfg,
            policy_cfg=policy_cfg,
            training_cfg=training_cfg,
            ablation=ablation,
        )
        print("Training arguments: ", training_kwargs)
        algo = IRLRunner(
            **training_kwargs,
            sampler_cls=VectorizedSampler#sampling.PPOBatchSampler,
            #sampler_args={
            #    'one_hot_code': one_hot_code
            #}
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
