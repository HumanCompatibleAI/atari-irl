from atari_irl import utils, policies, environments, irl
import pickle
from arguments import add_atari_args, add_trajectory_args, add_expert_args, tf_context_for_args, env_context_for_args
import argparse


def generate_trajectories(args):
    # environments are not one hot coded, so we don't wrap this
    env_modifiers = environments.env_mapping[args.env]
    if args.expert_type == 'irl':
        env_modifiers = environments.one_hot_wrap_modifiers(env_modifiers)

    utils.logger.configure()
    with utils.TfContext(ncpu=args.n_cpu):
        with utils.EnvironmentContext(
            env_name=args.env,
            n_envs=args.num_envs,
            seed=args.seed,
            **env_modifiers
        ) as context:
            if args.expert_type == 'baselines_ppo':
                policy = policies.EnvPolicy.load(args.expert_path, context.environments)
                model = policy.model
                envs = policy.envs
            elif args.expert_type == 'irl':
                irl_policy = irl.make_irl_policy(
                    irl.policy_config(
                        init_location=args.expert_path
                    ),
                    envs=irl.rllab_wrap_venv(context.environments),
                    baseline_wrappers=[]
                )
                model = irl_policy.learner.policy.model
                envs = irl_policy.learner.runner.env
            else:
                raise NotImplementedError

            ts = policies.sample_trajectories(
                model=model,
                environments=envs,
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
    add_expert_args(parser)
    add_trajectory_args(parser)

    args = parser.parse_args()
    generate_trajectories(args)