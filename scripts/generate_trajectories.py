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
                _, policy_cfg, _, _ = irl.get_training_kwargs(
                    venv=context.environments,
                    policy_cfg=dict(init_location=args.expert_path, name='other_policy'),
                    reward_model_cfg=dict(expert_trajs=None)
                )
                policy_cfg['name'] = 'policy'
                irl_policy = irl.make_irl_policy(
                    policy_cfg,
                    wrapped_venv=irl.rllab_wrap_venv(context.environments),
                    baselines_venv=context.environments
                )
                model = irl_policy.model
                envs = context.environments
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