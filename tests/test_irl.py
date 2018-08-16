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

        def check_irl_ppo_policy_sampler(algo, env_context):
            print("Checking IRL PPO policy sampler")
            irl_policy_samples = algo.policy.learner.runner.run()
            obs = irl_policy_samples[0]
            # Test that our sample raveling inversion actually inverts the
            # sf01 function applied by the ppo2 code
            assert np.isclose(
                ppo2.sf01(sampling.invert_ppo_sample_raveling(
                    obs, env_context.environments.num_envs
                )), obs
            ).all()
            assert_trajectory_formatted(sampling.ppo_samples_to_trajectory_format(
                irl_policy_samples, num_envs=env_context.environments.num_envs
            ))

        with utils.EnvironmentContext(
            env_name=self.env, n_envs=8, seed=0, **self.env_modifiers
        ) as env_context:
            with irl.IRLContext(self.config, seed=0) as irl_context:
                training_kwargs, _, _ = irl.get_training_kwargs(
                    venv=env_context.environments,
                    irl_context=irl_context,
                    expert_trajectories=pickle.load(open('scripts/short_trajectories.pkl', 'rb')),
                )
                print("Training arguments: ", training_kwargs)
                algo = irl.IRLRunner(**training_kwargs)
                check_base_policy_sampler(algo, env_context)
                check_irl_discriminator_sampler(algo, env_context)
                check_irl_ppo_policy_sampler(algo, env_context)

    def test_vectorized_sampler_processing_to_ppo_results(self):
        with utils.EnvironmentContext(
            env_name=self.env, n_envs=1, seed=0, **self.env_modifiers
        ) as env_context:
            with irl.IRLContext(self.config, seed=0) as irl_context:
                training_kwargs, _, _ = irl.get_training_kwargs(
                    venv=env_context.environments,
                    irl_context=irl_context,
                    expert_trajectories=pickle.load(open('scripts/short_trajectories.pkl', 'rb')),
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

                # These are very different because the policy is
                # non-deterministic. This test is only checking that the
                # shapes are right, and we need something more deterministic to
                # determine that the return calculation is also correct
                ppo_processed = algo.policy.learner.runner.process_trajectory(
                    vectorized_samples[0]
                )
                ppo_generated = algo.policy.learner.runner.run()

                # the indices before the states and episode infos
                for i in range(6):
                    print(i)
                    assert ppo_processed[i][:128].shape == ppo_generated[i].shape

    def test_ppo_sampling_roundtrips(self):
        with utils.EnvironmentContext(
            env_name=self.env, n_envs=8, seed=0, **self.env_modifiers
        ) as env_context:
            with irl.IRLContext(self.config, seed=0) as irl_context:
                training_kwargs, _, _ = irl.get_training_kwargs(
                    venv=env_context.environments,
                    irl_context=irl_context,
                    expert_trajectories=pickle.load(open('scripts/short_trajectories.pkl', 'rb')),
                )
                training_kwargs['batch_size'] = 50
                print("Training arguments: ", training_kwargs)

                env_context.environments.reset()
                algo = irl.IRLRunner(**training_kwargs)

                ppo_sample = algo.policy.learner.runner.sample()
                trajectories = ppo_sample.to_trajectories()
                assert_trajectory_formatted(trajectories.trajectories)
                roundtrip_sample = trajectories.to_ppo_sample()

                assert ppo_sample.obs == roundtrip_sample.obs
                assert ppo_sample.rewards == roundtrip_sample.rewards
                assert ppo_sample.actions == roundtrip_sample.actions
                assert ppo_sample.values == roundtrip_sample.values
                assert ppo_sample.dones == roundtrip_sample.dones
                assert ppo_sample.neglogpacs == roundtrip_sample.neglogpacs
                assert ppo_sample.states == roundtrip_sample.states
                assert ppo_sample.epinfos == roundtrip_sample.epinfos
                assert ppo_sample.runner == roundtrip_sample.runner

    def test_ppo_sampling_raveling(self):
        with utils.EnvironmentContext(
            env_name=self.env, n_envs=8, seed=0, **self.env_modifiers
        ) as env_context:
            with irl.IRLContext(self.config, seed=0) as irl_context:
                training_kwargs, _, _ = irl.get_training_kwargs(
                    venv=env_context.environments,
                    irl_context=irl_context,
                    expert_trajectories=pickle.load(open('scripts/short_trajectories.pkl', 'rb')),
                )
                training_kwargs['batch_size'] = 50
                print("Training arguments: ", training_kwargs)

                env_context.environments.reset()
                algo = irl.IRLRunner(**training_kwargs)

                ppo_sample = algo.policy.learner.runner.sample()

                train_batch_raveled_obs = ppo_sample._ravel_time_env_batch_to_train_batch(
                    ppo_sample.obs
                )
                # check that the second chunk of the first batch is the same as the second
                # environment in the set. This shows that we stacked the environments
                assert np.isclose(
                    train_batch_raveled_obs[0][1], ppo_sample.obs[:, 1]
                ).all()
                # check that the roundtrip worked
                assert np.isclose(
                    ppo_sample.obs, ppo_sample._ravel_train_batch_to_time_env_batch(
                        train_batch_raveled_obs
                    )
                ).all()

                ppo_batch = ppo_sample.to_ppo_batch()

                import pdb; pdb.set_trace()