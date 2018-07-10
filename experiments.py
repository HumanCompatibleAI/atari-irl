import utils
import environments
import training
import policies
import irl
from baselines.ppo2.policies import MlpPolicy, CnnPolicy
import os.path as osp
import os
import argparse
import tensorflow as tf
import pickle
from baselines import logger


def atari_arg_parser():
    # see baselines.common.cmd_util
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='PongNoFrameskip-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num_timesteps', type=int, default=int(10e6))

    parser.add_argument('--num_envs', type=int, default=8)
    return parser


def train_expert(args):
    utils.logger.configure()
    with utils.TfContext():
        with utils.EnvironmentContext(
                env_name=args.env,
                n_envs=args.num_envs,
                seed=args.seed,
                **environments.atari_modifiers
        ) as context:
            learner = training.Learner(
                CnnPolicy, context.environments,
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

            policies.run_policy(model=policy.model, environments=policy.envs)


def generate_trajectories(args):
    with tf.device('/gpu:1'):
        with utils.TfContext():
            with utils.EnvironmentContext(
                    env_name=env_name,
                    n_envs=8,
                    seed=21,
                    **environments.atari_modifiers
            ) as context:
                policy = policies.EnvPolicy.load(args.expert_path, context.environments)
                policies.run_policy(model=policy.model, environments=policy.envs)
                ts = policies.sample_trajectories(
                    model=policy.model,
                    environments=policy.envs,
                    n_trajectories=8,
                    one_hot_code=True
                )

    pickle.dump(ts, open(args.trajectories_file, 'wb'))


def train_airl(args):
    env_name = args.env

    tf_cfg = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=args.ncpu,
        inter_op_parallelism_threads=args.ncpu,
        device_count={'GPU': 1},
        log_device_placement=True
    )

    ts = pickle.load(open(args.trajectories_file, 'rb'))

    with utils.EnvironmentContext(
            env_name=env_name,
            n_envs=args.num_envs,
            seed=args.env_seed,
            **environments.one_hot_atari_modifiers
    ) as context:
        logger.configure()
        reward, policy_params = irl.airl(
            context.environments, ts, args.discount, args.irl_seed, logger.get_dir(),
            tf_cfg=tf_cfg,
            training_cfg={'n_itr': args.n_iter},
            policy_cfg=irl.policy_config(args)
        )

        import pickle
        pickle.dump(policy_params, open(args.irl_policy_name, 'wb'))
        # policy.step = lambda obs: (policy.get_actions(obs)[0], None, None, None)
        # policies.run_policy(model=policy, environments=context.environments)


if __name__ == '__main__':
    parser = atari_arg_parser()
    args = parser.parse_args()
    train_expert(args)
