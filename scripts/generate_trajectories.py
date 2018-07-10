from atari_irl import utils, environments, policies
import pickle
from arguments import add_atari_args, add_trajectory_args, EXPERT_POLICY_FILENAME_ARG
import argparse


def generate_trajectories(args):
    with utils.TfContext(args.n_cpu):
        with utils.EnvironmentContext(
                env_name=args.env,
                n_envs=args.num_envs,
                seed=args.seed,
                **environments.atari_modifiers
        ) as context:
            policy = policies.EnvPolicy.load(args.expert_path, context.environments)
            ts = policies.sample_trajectories(
                model=policy.model,
                environments=policy.envs,
                n_trajectories=args.num_trajectories,
                one_hot_code=args.one_hot_code,
                render=args.render
            )

    pickle.dump(ts, open(args.trajectories_file, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_atari_args(parser)
    parser.add_argument(
        EXPERT_POLICY_FILENAME_ARG,
        help='file for the expert policy',
        default='experts/expert.pkl'
    )
    parser.add_argument(
        '--render', help='whether or not to render the sampled trajectories', default=True
    )
    add_trajectory_args(parser)

    args = parser.parse_args()
    generate_trajectories(args)