from atari_irl import utils, environments, training, policies
import argparse
from arguments import add_atari_args, EXPERT_POLICY_FILENAME_ARG, add_bool_feature
from baselines.ppo2.policies import MlpPolicy, CnnPolicy
import os.path as osp
import os


def train_expert(args):
    utils.logger.configure()

    modifiers = environments.atari_modifiers
    if not args.atari:
        modifiers = environments.mujoco_modifiers
    print(args.atari, modifiers)

    with utils.TfContext():
        with utils.EnvironmentContext(
                env_name=args.env,
                n_envs=args.num_envs,
                seed=args.seed,
                **modifiers
        ) as context:
            learner = training.Learner(
                CnnPolicy if args.policy_type == 'cnn' else MlpPolicy,
                context.environments,
                total_timesteps=args.num_timesteps,
                vf_coef=0.5, ent_coef=0.01,
                nsteps=128, noptepochs=4, nminibatches=4,
                gamma=0.99, lam=0.95,
                lr=lambda alpha: alpha * 2.5e-4,
                cliprange=lambda alpha: alpha * 0.1
            )

            for policy, update, mean_reward in learner.learn_and_yield(
                    lambda l: (l.policy, l.update, l.mean_reward), 100,
                    log_freq=1
            ):
                checkdir = osp.join(utils.logger.get_dir(), 'checkpoints')
                os.makedirs(checkdir, exist_ok=True)
                savepath = osp.join(checkdir, 'update-{}'.format(update))
                print('Saving to', savepath)
                policy.save(
                    savepath,
                    mean_reward=learner.mean_reward,
                    update=update,
                    seed=args.seed
                )

            policy.save(
                args.expert_path,
                mean_reward=learner.mean_reward, update=update, seed=args.seed
            )

            policies.run_policy(model=policy.model, environments=policy.envs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_atari_args(parser)
    parser.add_argument(
        EXPERT_POLICY_FILENAME_ARG,
        help='file for the expert policy',
        default='experts/new_expert'
    )

    add_bool_feature(parser, 'atari')

    parser.add_argument(
        '--is_atari',
        help='Whether or not this is an atari environment',
        type=bool, default=True
    )
    args = parser.parse_args()
    train_expert(args)