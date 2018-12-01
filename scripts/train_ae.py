from atari_irl import utils, encoding
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from baselines.ppo2 import ppo2
import joblib
import os
import os.path as osp
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--env', help='environment', type=str, default='PongNoFrameskip-v4')
parser.add_argument('--num_envs', help='number of environments', type=int, default=8)
parser.add_argument('--seed', help='random seed', type=int, default=0)
parser.add_argument(
    '--encoder_type',
    help='type of encoder',
    choices=['score_trimmed', 'next_step', 'non_pixel_class', 'pixel_class'],
    type=str, default='pixel_class'
)
parser.add_argument('--out_dir', help='name of output directory')
args = parser.parse_args()

encoder = encoding.VariationalAutoEncoder
if args.encoder_type == 'score_trimmed':
    encoder = encoding.ScoreTrimmedVariationalAutoEncoder
elif args.encoder_type == 'next_step':
    encoder = encoding.NextStepVariationalAutoEncoder
elif args.encoder_type == 'non_pixel_class':
    encoder = encoding.NormalAutoEncoder

tf_cfg = tf.ConfigProto(
    allow_soft_placement=True,
    intra_op_parallelism_threads=8,
    inter_op_parallelism_threads=8,
    device_count={'GPU': 1},
    log_device_placement=False
)
tf_cfg.gpu_options.allow_growth = True

env_cfg = {
    'env_name': args.env,
    'n_envs': args.num_envs,
    'seed': args.seed,
    'one_hot_code': False
}

extra_args = {}
if args.encoder_type == 'non_pixel_class' and 'Pong' in args.env:
    extra_args['trim_score'] = True
    
with utils.TfEnvContext(tf_cfg, env_cfg) as context:
    utils.logger.configure()
    os.makedirs(args.out_dir)
    dirname = args.out_dir
    print(f"logging in {dirname}")
    vae = encoder(
        obs_shape=context.env_context.environments.observation_space.shape,
        d_embedding=40,
        d_classes=30,
        embedding_weight=0.001,
        **extra_args
    )
    LR = 1e-4
    
    def args(*, lr, obs, noise_scale=.2):
        return {
            'lr': lr,
            'obs': obs,
            'noise': np.random.randn(obs.shape[0], vae.d_embedding) * noise_scale
        }

    tf.get_default_session().run(tf.local_variables_initializer())
    tf.get_default_session().run(tf.global_variables_initializer())

    buffer = deque(maxlen=100)

    env = context.env_context.environments
    env.reset()
    num_timesteps = 2500
    losses = []
    for i in range(num_timesteps):
        lr = LR * (num_timesteps - i) * 1.0 / num_timesteps
        obs = []
        for t in range(32):
            acts = [env.action_space.sample() for _ in range(env.num_envs)]
            obs.append(env.step(acts)[0])
            
        obs_arr = np.array(obs).astype(np.uint8)
        obs.clear()
        del obs
        obs_batch = ppo2.sf01(obs_arr)
        del obs_arr
                              
        if i % 100 == 0:
            img, loss = vae.compare(obs_batch, disp_p=.1)
            losses.append((i, loss))
            utils.logger.info(losses[-1])
            fname = osp.join(dirname, f'vae_{i}.pkl')
            utils.logger.info(f"Saving vae in {fname}")
            vae.save(fname)
            joblib.dump(img, open(osp.join(dirname, f'img_sample_{i}.pkl'), 'wb'))
            joblib.dump(losses, open(osp.join(dirname, f'losses_{i}.pkl'), 'wb'))

        utils.logger.info(f"{i}: {vae.train_step(**args(lr=lr, obs=obs_batch, noise_scale=0.0))}")
        buffer.append(obs_batch)
        for bi in range(len(buffer)):
            loss = vae.train_step(**args(lr=lr, obs=buffer[bi], noise_scale=0.0))