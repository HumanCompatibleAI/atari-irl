from atari_irl import utils, environments, irl
import pickle
from arguments import add_atari_args, add_trajectory_args, add_irl_args, tf_context_for_args, env_context_for_args
import argparse
import tensorflow as tf
from sandbox.rocky.tf.envs.base import TfEnv


def run_irl_policy(args):
    with tf_context_for_args(args):
        with env_context_for_args(args) as context:
            envs = environments.VecGymEnv(context.environments)
            envs = TfEnv(envs)

            policy = irl.make_irl_policy(
                irl.policy_config(args),
                envs=envs,
                sess=tf.get_default_session()
            )
            policy.restore_param_values(args.irl_policy_file)
            policy.show_run_in_gym_env(context.environments)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_atari_args(parser)
    add_irl_args(parser)

    args = parser.parse_args()
    run_irl_policy(args)