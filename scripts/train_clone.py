import argparse
import tensorflow as tf
from atari_irl import utils, behavioral_cloning
import os.path as osp
import joblib
from arguments import add_atari_args, add_trajectory_args, add_expert_args, tf_context_for_args, env_context_for_args

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_atari_args(parser)
    add_expert_args(parser)
    add_trajectory_args(parser)
    
    parser.add_argument('--clone_path', default='clone.pkl')
    parser.add_argument('--epochs', default=500, type=int)
    
    args = parser.parse_args()

    tf_cfg = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=args.n_cpu,
        inter_op_parallelism_threads=args.n_cpu,
        device_count={'GPU': 1},
        log_device_placement=False
    )
    tf_cfg.gpu_options.allow_growth = True

    env_config = {
         'env_name': args.env,
         'n_envs': args.num_envs,
         'seed': args.seed,
         'one_hot_code': args.one_hot_code
     }
    with utils.TfEnvContext(tf_cfg, env_config) as context:
        utils.logger.configure()
        #encoder = encoding.NextStepVariationalAutoEncoder.load('../scripts/encoders/run3/vae_850.pkl')
        expert_obs_base, expert_obs_next_base, expert_acts, expert_acts_next, _ = \
            joblib.load(args.trajectories_file)

        del expert_obs_next_base
        del expert_acts_next
        del _

        clone = behavioral_cloning.Cloner(
            obs_shape=expert_obs_base.shape[1:],
            n_actions=context.env_context.environments.action_space.n
        )

        tf.get_default_session().run(tf.local_variables_initializer())
        tf.get_default_session().run(tf.global_variables_initializer())


        obs = expert_obs_base
        act = expert_acts
        clone.train(obs=obs, act=act, epochs=args.epochs)
        clone.save(args.clone_path)