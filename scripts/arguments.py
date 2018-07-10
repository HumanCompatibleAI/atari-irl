#from atari_irl import utils, environments, training, policies, irl

import os.path as osp
import os
import argparse
#import tensorflow as tf
#from baselines import logger
#from sandbox.rocky.tf.envs.base import TfEnv
#import pickle

EXPERT_POLICY_FILENAME_ARG = '--expert_path'


def add_atari_args(parser):
    # see baselines.common.cmd_util
    parser.add_argument('--env', help='environment ID', default='PongNoFrameskip-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num_timesteps', type=int, default=int(10e6))
    parser.add_argument('--num_envs', type=int, default=8)


def add_trajectory_args(parser):
    parser.add_argument('--n_cpu', help='Number of CPUs', default=8)
    parser.add_argument('--num_trajectories', help='number of trajectories', type=int, default=8)
    parser.add_argument(
        '--one_hot_code',
        help='Whether or not to one hot code the actions',
        type=bool, default=True
    )
    parser.add_argument(
        '--trajectories_file',
        help='file to write the trajectories to',
        default='trajectories.pkl'
    )


def add_irl_args(parser):
    parser.add_argument('--irl_seed', help='seed for the IRL tensorflow session', default=0)
    parser.add_argument(
        '--irl_policy_file',
        help='filename for the IRL policy',
        default='irl_policy_params.pkl'
    )


def train_airl(args):
    tf_cfg = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=args.ncpu,
        inter_op_parallelism_threads=args.ncpu,
        device_count={'GPU': 1},
        log_device_placement=True
    )

    ts = pickle.load(open(args.trajectories_file, 'rb'))

    env_config = environments.one_hot_atari_modifiers
    if not args.one_hot_code:
        env_config = environments.atari_modifiers
    with utils.EnvironmentContext(
            env_name=args.env,
            n_envs=args.num_envs,
            seed=args.env_seed,
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


def run_irl(args):
    with utils.TfContext():
        with utils.EnvironmentContext(
                env_name=args.env,
                n_envs=args.num_env,
                seed=args.seed,
                **environments.atari_modifiers
        ) as context:
            envs = environments.VecGymEnv(context.environments)
            envs = TfEnv(envs)

            policy = irl.make_irl_policy(
                irl.policy_config(args),
                envs=envs,
                sess=tf.get_default_session()
            )
            policy.restore_param_values(args.irl_policy_file)
            policy.show_run_in_gym_env(context.environments)



