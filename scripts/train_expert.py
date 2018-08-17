from atari_irl import utils, training, policies
import argparse
from arguments import add_atari_args, add_expert_args, env_context_for_args, tf_context_for_args
from baselines.ppo2.policies import MlpPolicy, CnnPolicy
import os.path as osp
import os


def train_expert(args):
    utils.logger.configure()
    with tf_context_for_args(args):
        with env_context_for_args(args) as context:
            learner = training.Learner(
                CnnPolicy if args.policy_type == 'cnn' else MlpPolicy,
                context.environments,
                total_timesteps=args.num_timesteps,
                vf_coef=0.5, ent_coef=0.01,
                nsteps=args.nsteps, noptepochs=4, nminibatches=4,
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
    add_expert_args(parser)
    args = parser.parse_args()

    assert not args.one_hot_code
    train_expert(args)