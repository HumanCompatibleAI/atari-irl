from atari_irl import utils, environments, irl
from arguments import add_atari_args, add_trajectory_args, add_irl_args, env_context_for_args
import argparse
from baselines import logger
import tensorflow as tf
import pickle
import joblib
from baselines.ppo2.policies import CnnPolicy, MlpPolicy
from atari_irl.irl import cnn_net
from airl.models.architectures import relu_net


def train_airl(args):
    tf_cfg = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=args.n_cpu,
        inter_op_parallelism_threads=args.n_cpu,
        device_count={'GPU': 1},
        log_device_placement=False
    )
    tf_cfg.gpu_options.allow_growth = True

    logger.configure()
    reward, policy_params = irl.airl(
        logger.get_dir(),
        tf_cfg=tf_cfg,
        training_cfg={
            'n_itr': args.n_iter,
            'batch_size': args.batch_size,
            'entropy_weight': args.entropy_wt
        },
        env_config={
            'env_name': args.env,
            'n_envs': args.num_envs,
            'seed': args.seed,
            'one_hot_code': args.one_hot_code
        },
        policy_cfg={
            'init_location': None if args.init_location == 'none' else args.init_location,
            'policy_model': CnnPolicy if args.policy_type == 'cnn' else MlpPolicy
        },
        reward_model_cfg={
            'expert_trajs': joblib.load(open(args.trajectories_file, 'rb')),
            'state_only': args.state_only,
            'drop_framestack': args.drop_discriminator_framestack,
            'only_show_scores': args.only_show_discriminator_scores,
            'reward_arch': cnn_net if args.policy_type == 'cnn' else relu_net,
            'value_fn_arch': cnn_net if args.policy_type == 'cnn' else relu_net,
            'encoder_loc': None if not args.encoder else args.encoder
        },
        ablation=args.ablation
    )

    pickle.dump(policy_params, open(args.irl_policy_file, 'wb'))
    pickle.dump(reward, open(args.irl_reward_file, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_atari_args(parser)
    add_trajectory_args(parser)
    add_irl_args(parser)

    args = parser.parse_args()
    train_airl(args)