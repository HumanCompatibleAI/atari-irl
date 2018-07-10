from atari_irl import utils, environments, irl
import pickle
from arguments import add_atari_args, add_trajectory_args, add_irl_args
import argparse
from baselines import logger
import tensorflow as tf


def train_airl(args):
    tf_cfg = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=args.n_cpu,
        inter_op_parallelism_threads=args.n_cpu,
        device_count={'GPU': 1},
        log_device_placement=True
    )
    import pickle
    ts = pickle.load(open(args.trajectories_file, 'rb'))

    env_config = environments.one_hot_atari_modifiers
    if not args.one_hot_code:
        env_config = environments.atari_modifiers
    with utils.EnvironmentContext(
            env_name=args.env,
            n_envs=args.num_envs,
            seed=args.seed,
            **env_config
    ) as context:
        logger.configure()
        reward, policy_params = irl.airl(
            context.environments, ts, args.discount, args.irl_seed, logger.get_dir(),
            tf_cfg=tf_cfg,
            training_cfg={'n_itr': args.n_iter},
            policy_cfg=irl.policy_config(args)
        )

        import pickle
        pickle.dump(policy_params, open(args.irl_policy_file, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_atari_args(parser)
    add_trajectory_args(parser)
    add_irl_args(parser)

    args = parser.parse_args()
    train_airl(args)