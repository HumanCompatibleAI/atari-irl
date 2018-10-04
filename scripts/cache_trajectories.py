from atari_irl import sampling, irl, utils
from arguments import add_atari_args, add_trajectory_args, add_irl_args, env_context_for_args
import argparse
from baselines import logger
import tensorflow as tf
import numpy as np
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
    env_config = {
         'env_name': args.env,
         'n_envs': args.num_envs,
         'seed': args.seed,
         'one_hot_code': args.one_hot_code
     }
    with utils.TfEnvContext(tf_cfg, env_config) as context:
        ts = joblib.load(open(args.trajectories_file, 'rb'))
        training_kwargs, _, _, _ = irl.get_training_kwargs(
            venv=context.env_context.environments,
            training_cfg={
                'n_itr': args.n_iter,
                'batch_size': args.batch_size,
                'entropy_weight': args.entropy_wt
            },
            policy_cfg={
                'init_location': None if args.init_location == 'none' else args.init_location,
                'policy_model': CnnPolicy if args.policy_type == 'cnn' else MlpPolicy
            },
            reward_model_cfg={
                'expert_trajs': ts,
                'state_only': args.state_only,
                'drop_framestack': args.drop_discriminator_framestack,
                'only_show_scores': args.only_show_discriminator_scores,
                'reward_arch': cnn_net if args.policy_type == 'cnn' else relu_net,
                'value_fn_arch': cnn_net if args.policy_type == 'cnn' else relu_net
            },
            ablation=args.ablation
        )
        algo = irl.IRLRunner(
            **training_kwargs,
            sampler_cls=sampling.PPOBatchSampler,
        )

        def fill_trajectories(paths):
            algo.irl_model.eval_expert_probs(paths, algo.policy, insert=True)
            algo.irl_model._insert_next_state(paths)

        fill_trajectories(ts)
        for t in ts:
            del t['agent_infos']

        T = len(ts)
        ans = []
        keys = ('observations', 'observations_next', 'actions', 'actions_next', 'a_logprobs')
        for key in keys:
            print(key)
            batch = []
            for i in range(T):
                batch.append(ts[i][key].copy())
                del ts[i][key]
            ans.append(np.concatenate(batch).astype(np.float32))
            for i in reversed(range(len(batch))):
                del batch[i]
            del batch
        joblib.dump(ans, open(args.cache_path, 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_atari_args(parser)
    add_trajectory_args(parser)
    add_irl_args(parser)
    parser.add_argument('--cache_path', default='cache.pkl')
    args = parser.parse_args()
    train_airl(args)