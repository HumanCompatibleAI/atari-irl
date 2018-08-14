from atari_irl import irl, utils, environments, policies
import tensorflow as tf
import numpy as np
import pickle


class TestAtariIRL:
    def test_sample_shape(self):
        env = 'PongNoFrameskip-v4'
        env_modifiers = environments.env_mapping[env]
        env_modifiers = environments.one_hot_wrap_modifiers(env_modifiers)

        config = tf.ConfigProto(
            allow_soft_placement=True,
            intra_op_parallelism_threads=8,
            inter_op_parallelism_threads=8,
            device_count={'GPU': 1},
        )
        config.gpu_options.allow_growth=True

        def assert_trajectory_formatted(samples):
            print(f"Found {len(samples)} trajectories")
            for sample in samples:
                assert 'observations' in sample
                assert 'actions' in sample
                T = len(sample['observations'])
                print(f"\tFound trajectory of length {T}")
                if not hasattr(sample['observations'], 'shape'):
                    print("Time index is list, not numpy dimension")
                assert np.array(sample['observations']).shape == (T, 84, 84, 4)
                assert np.array(sample['actions']).shape == (T, 6)


        with utils.EnvironmentContext(
            env_name=env, n_envs=8, seed=0, **env_modifiers
        ) as env_context:
            with irl.IRLContext(config, seed=0) as irl_context:
                algo = irl.IRLRunner(**irl.get_training_kwargs(
                    venv=env_context.environments,
                    irl_context=irl_context,
                    expert_trajectories=pickle.load(
                        open('scripts/short_trajectories.pkl', 'rb')
                    )
                )[0])

                policy_samples = policies.sample_trajectories(
                    model=algo.policy.learner.model,
                    environments=env_context.environments,
                    one_hot_code=True,
                    n_trajectories=10,
                    render=False
                )
                assert len(policy_samples) == 10
                assert_trajectory_formatted(policy_samples)

                env_context.environments.reset()
                algo.start_worker()
                irl_discriminator_samples = algo.obtain_samples(0)

                assert_trajectory_formatted(irl_discriminator_samples)
