from atari_irl import irl, utils, environments, policies, training, sampling
import tensorflow as tf
import numpy as np
import pickle

from baselines.ppo2 import ppo2


def assert_trajectory_formatted(samples):
    print(f"Found {len(samples)} trajectories")
    for sample in samples:
        assert 'observations' in sample
        assert 'actions' in sample
        T = len(sample['observations'])
        print(f"\tFound trajectory of length {T}")
        if not hasattr(sample['observations'], 'shape'):
            print("\tTime index is list, not numpy dimension")
        assert np.array(sample['observations']).shape == (T, 84, 84, 4)
        assert np.array(sample['actions']).shape == (T, 6)


class TestAtariIRL:
    def setup_method(self, method):
        self.env = 'PongNoFrameskip-v4'
        env_modifiers = environments.env_mapping[self.env]
        self.env_modifiers = environments.one_hot_wrap_modifiers(env_modifiers)

        self.config = tf.ConfigProto(
            allow_soft_placement=True,
            intra_op_parallelism_threads=8,
            inter_op_parallelism_threads=8,
            device_count={'GPU': 1},
        )
        self.config.gpu_options.allow_growth=True

    def test_sample_shape(self):
        def check_base_policy_sampler(algo, env_context):
            print("Checking straightforward policy trajectory sampler")
            policy_samples = policies.sample_trajectories(
                model=algo.policy.learner.model,
                environments=env_context.environments,
                one_hot_code=True,
                n_trajectories=10,
                render=False
            )
            assert len(policy_samples) == 10
            assert_trajectory_formatted(policy_samples)

        def check_irl_discriminator_sampler(algo, env_context):
            print("Checking discriminator sampler")
            #env_context.environments.reset()
            algo.start_worker()
            irl_discriminator_samples = algo.obtain_samples(0)
            assert_trajectory_formatted(irl_discriminator_samples)

        with utils.EnvironmentContext(
            env_name=self.env, n_envs=8, seed=0, **self.env_modifiers
        ) as env_context:
            with irl.IRLContext(self.config, env_config={
                'seed': 0,
                'env_name': 'PongNoFrameskip-v4',
                'one_hot_code': True
            }):
                training_kwargs, _, _, _ = irl.get_training_kwargs(
                    venv=env_context.environments,
                    reward_model_cfg={
                        'expert_trajs': pickle.load(open('scripts/short_trajectories.pkl', 'rb')),
                    }
                )
                print("Training arguments: ", training_kwargs)
                algo = irl.IRLRunner(**training_kwargs)
                check_base_policy_sampler(algo, env_context)
                check_irl_discriminator_sampler(algo, env_context)

    def test_vectorized_sampler_processing_to_ppo_results(self):
        with utils.EnvironmentContext(
            env_name=self.env, n_envs=1, seed=0, **self.env_modifiers
        ) as env_context:
            with irl.IRLContext(self.config, env_config={
                'seed': 0,
                'env_name': 'PongNoFrameskip-v4',
                'one_hot_code': True
            }):
                training_kwargs, _, _, _ = irl.get_training_kwargs(
                    venv=env_context.environments,
                    reward_model_cfg={
                        'expert_trajs': pickle.load(open('scripts/short_trajectories.pkl', 'rb')),
                    }
                )
                training_kwargs['batch_size'] = 50
                print("Training arguments: ", training_kwargs)

                env_context.environments.reset()
                algo = irl.IRLRunner(**training_kwargs)

                algo.start_worker()
                vectorized_samples = algo.obtain_samples(0)

                # check some basic things about the vectorized samples
                # We should only have one path
                assert len(vectorized_samples) == 1
                assert_trajectory_formatted(vectorized_samples)
                # It shouldn't be super short
                assert len(vectorized_samples[0]['actions']) > 100

                sampler = sampling.PPOBatchSampler(
                    model=algo.policy.learner.model,
                    env=env_context.environments,
                    nsteps=128*env_context.environments.num_envs
                )

                # These are very different because the policy is
                # non-deterministic. This test is only checking that the
                # shapes are right, and we need something more deterministic to
                # determine that the return calculation is also correct
                ppo_processed = sampler.process_trajectory(
                    vectorized_samples[0], gamma=0.99, lam=0.95
                ).train_args()
                ppo_generated = sampler.process_to_ppo_batch(
                    sampler.run(), gamma=0.99, lam=0.95
                ).train_args()

                assert len(ppo_processed) == len(ppo_generated)
                # the indices before the states and episode infos
                for i in range(len(ppo_processed)):
                    assert ppo_processed[i][:128].shape == ppo_generated[i].shape

    def test_ppo_sampling_roundtrips(self):
        with utils.EnvironmentContext(
            env_name=self.env, n_envs=8, seed=0, **self.env_modifiers
        ) as env_context:
            with irl.IRLContext(self.config, env_config={
                'seed': 0,
                'env_name': 'PongNoFrameskip-v4',
                'one_hot_code': True
            }):
                training_kwargs, _, _, _ = irl.get_training_kwargs(
                    venv=env_context.environments,
                    reward_model_cfg={
                        'expert_trajs': pickle.load(open('scripts/short_trajectories.pkl', 'rb')),
                    }
                )
                training_kwargs['batch_size'] = 50
                print("Training arguments: ", training_kwargs)

                env_context.environments.reset()
                algo = irl.IRLRunner(**training_kwargs)

                ppo_sample = algo.policy.learner.runner.sample()
                trajectories = ppo_sample.to_trajectories()
                assert_trajectory_formatted(trajectories.trajectories)
                roundtrip_sample = trajectories.to_ppo_sample()

                assert (ppo_sample.obs == roundtrip_sample.obs).all()
                assert (ppo_sample.rewards == roundtrip_sample.rewards).all()
                assert (ppo_sample.actions == roundtrip_sample.actions).all()
                assert (ppo_sample.values == roundtrip_sample.values).all()
                assert (ppo_sample.dones == roundtrip_sample.dones).all()
                assert (ppo_sample.neglogpacs == roundtrip_sample.neglogpacs).all()
                assert ppo_sample.states == roundtrip_sample.states
                assert ppo_sample.epinfos == roundtrip_sample.epinfos
                assert ppo_sample.sampler == roundtrip_sample.sampler

    def test_ppo_sampling_raveling(self):
        with utils.EnvironmentContext(
            env_name=self.env, n_envs=8, seed=0, **self.env_modifiers
        ) as env_context:
            with irl.IRLContext(self.config, env_config={
                'seed': 0,
                'env_name': 'PongNoFrameskip-v4',
                'one_hot_code': True
            }):
                training_kwargs, _, _, _ = irl.get_training_kwargs(
                    venv=env_context.environments,
                    reward_model_cfg={
                        'expert_trajs': pickle.load(open('scripts/short_trajectories.pkl', 'rb')),
                    }
                )
                training_kwargs['batch_size'] = 50
                print("Training arguments: ", training_kwargs)

                env_context.environments.reset()
                algo = irl.IRLRunner(**training_kwargs)

                ppo_sample = algo.policy.learner.runner.sample()

                train_batch_raveled_obs = ppo_sample._ravel_time_env_batch_to_train_batch(
                    ppo_sample.obs
                )
                # check that the second chunk of the first batch is the same as
                # the second environment in the ppo sample. This shows that we
                # stacked the environments correctly
                assert np.isclose(
                    train_batch_raveled_obs[0][ppo_sample.obs.shape[0]:],
                    ppo_sample.obs[:, 1]
                ).all()

                # check that the roundtrip works, as a sanity check
                assert np.isclose(
                    ppo_sample.obs, ppo_sample._ravel_train_batch_to_time_env_batch(
                        train_batch_raveled_obs
                    )
                ).all()

    def test_ppo_sampling_probs_calculation(self):
        with utils.EnvironmentContext(
            env_name=self.env, n_envs=8, seed=0, **self.env_modifiers
        ) as env_context:
            with irl.IRLContext(self.config, env_config={
                'seed': 0,
                'env_name': 'PongNoFrameskip-v4',
                'one_hot_code': True
            }):
                training_kwargs, _, _, _ = irl.get_training_kwargs(
                    venv=env_context.environments,
                    reward_model_cfg={
                        'expert_trajs': pickle.load(open('scripts/short_trajectories.pkl', 'rb')),
                    }
                )
                training_kwargs['batch_size'] = 50
                print("Training arguments: ", training_kwargs)

                env_context.environments.reset()
                algo = irl.IRLRunner(**training_kwargs)

                ppo_sample = algo.policy.learner.runner.sample()

                # check that the probabilities are probabilities and sum to one
                sums = ppo_sample.probabilities.sum(axis=2)
                assert np.isclose(sums, np.ones(sums.shape)).all()

                # the probabilities are consistent with the neglogpacs
                one_hot_actions = utils.one_hot(
                    ppo_sample.actions.reshape(128 * 8), 6
                ).reshape(128, 8, 6)
                neglogpacs = -1 * np.log(
                    (ppo_sample.probabilities * one_hot_actions).sum(axis=2)
                )
                assert np.isclose(neglogpacs, ppo_sample.neglogpacs).all()