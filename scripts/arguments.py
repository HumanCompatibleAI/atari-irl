#from atari_irl import utils, environments, training, policies, irl
from atari_irl import utils, environments


def add_bool_feature(parser, name, default=True):
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--' + name, dest=name, action='store_true')
    feature_parser.add_argument('--no-' + name, dest=name, action='store_false')
    parser.set_defaults(**{name: default})


def add_atari_args(parser):
    # see baselines.common.cmd_util
    parser.add_argument('--env', help='environment ID', default='PongNoFrameskip-v4')
    parser.add_argument('--n_cpu', help='Number of CPUs', default=8)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num_timesteps', type=int, default=int(10e6))
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--policy_type', default='cnn')
    parser.add_argument('--ent_coef', type=float, default=0.01)
    add_bool_feature(parser, 'one_hot_code')


def add_trajectory_args(parser):
    parser.add_argument('--num_trajectories', help='number of trajectories', type=int, default=8)
    parser.add_argument(
        '--trajectories_file',
        help='file to write the trajectories to',
        default='trajectories.pkl'
    )
    add_bool_feature(parser, 'render')


def add_expert_args(parser):
    parser.add_argument(
        '--expert_path',
        help='file for the expert policy',
        default='experts/new_expert'
    )
    parser.add_argument(
        '--nsteps',
        help='length of time in a minibatch step',
        type=int, default=128
    )
    parser.add_argument(
        '--expert_type',
        help='type of the expert',
        choices=['baselines_ppo', 'irl'],
        default='baselines_ppo'
    )


def add_irl_args(parser):
    parser.add_argument('--irl_seed', help='seed for the IRL tensorflow session', type=int, default=0)
    parser.add_argument(
        '--irl_policy_file',
        help='filename for the IRL policy',
        default='irl_policy_params.pkl'
    )
    parser.add_argument(
        '--irl_reward_file',
        help='filename for the IRL reward',
        default='irl_reward.pkl'
    )
    parser.add_argument('--discount', help='discount rate for the IRL policy', default=.99)
    parser.add_argument(
        '--n_iter',
        help='number of iterations for irl training',
        type=int, default=500
    )
    parser.add_argument(
        '--batch_size',
        help='batch size for each iteration',
        type=int, default=5000
    )
    parser.add_argument(
        '--ablation',
        help='what ablation to run',
        choices=['none', 'train_rl', 'train_discriminator'],
        type=str, default='none'
    )
    parser.add_argument(
        '--entropy_wt',
        help='entropy_weight',
        type=float, default=0.01
    )
    parser.add_argument(
        '--init_location',
        help='location to initialize training from',
        type=str, default='none'
    )
    add_bool_feature(parser, 'state_only', default=False)


def env_context_for_args(args):
    env_modifiers = environments.env_mapping[args.env]
    if args.one_hot_code:
        env_modifiers = environments.one_hot_wrap_modifiers(env_modifiers)

    return utils.EnvironmentContext(
        env_name=args.env,
        n_envs=args.num_envs,
        seed=args.seed,
        **env_modifiers
    )


def tf_context_for_args(args):
    return utils.TfContext(
        ncpu=args.n_cpu
    )