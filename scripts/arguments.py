#from atari_irl import utils, environments, training, policies, irl

import os.path as osp
import os
import argparse
#
#
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
    parser.add_argument('--irl_seed', help='seed for the IRL tensorflow session', type=int, default=0)
    parser.add_argument(
        '--irl_policy_file',
        help='filename for the IRL policy',
        default='irl_policy_params.pkl'
    )
    parser.add_argument('--discount', help='discount rate for the IRL policy', default=.99)
    parser.add_argument(
        '--n_iter',
        help='number of iterations for irl training',
        type=int, default=500
    )


def run_irl_policy(args):
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



