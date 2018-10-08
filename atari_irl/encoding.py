from atari_irl import utils
import tensorflow as tf
import numpy as np

import joblib
import os.path as osp

from baselines.a2c.utils import conv, fc, conv_to_fc
from baselines.ppo2 import ppo2

from collections import deque

def batch_norm(name, x):
    shape = (1, *x.shape[1:])
    with tf.variable_scope(name):
        mean = tf.get_variable('mean', shape, initializer=tf.constant_initializer(0.0))
        variance = tf.get_variable('variance', shape, initializer=tf.constant_initializer(1.0))
        offset = tf.get_variable('offset', shape, initializer=tf.constant_initializer(0.0))
        scale = tf.get_variable('scale', shape, initializer=tf.constant_initializer(1.0))
    return tf.nn.batch_normalization(
        x, mean, variance, offset, scale, 0.001, name
    )


def dcgan_cnn(images, dout, **conv_kwargs):
    activ = lambda name, inpt: tf.nn.leaky_relu(batch_norm(name, inpt), alpha=0.2)
    l1 = activ('l1', conv(images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2), **conv_kwargs))
    l2 = activ('l2', conv(l1, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    l3 = activ('l3', conv(l2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    out = fc(conv_to_fc(l3), nh=dout, scope='final')
    return out, l3.shape


def decoder_cnn(embedding, start_conv_shape, dclasses):
    activ = lambda name, inpt: tf.nn.relu(inpt)  # batch_norm(name, inpt))
    enhance = fc(embedding, 'out_shape', nh=np.prod(start_conv_shape))
    start_conv = tf.reshape(enhance, [-1, *start_conv_shape])
    tf.layers.conv2d_transpose(start_conv, 64, 3, strides=1)
    l1 = activ('l3inv', tf.layers.conv2d_transpose(start_conv, 64, 3, strides=1))
    l2 = activ('l2inv', tf.layers.conv2d_transpose(l1, 32, 4, strides=2))
    l3 = tf.layers.conv2d_transpose(l2, dclasses + 1, 8, strides=4)
    return l3

class NormalAutoEncoder:
    unk_mean = 255.0 / 2
    
    @staticmethod
    def _check_obs(obs):
        assert (obs >= 0).all()
        assert (obs <= 255).all()
        if not (obs[:, :10, :, :] == 87).all():
            obs[:, :10, :, :] = 87

    @staticmethod
    def _process_obs_tensor(obs):
        return tf.cast(obs, tf.float32)

    def _get_frame(self):
        return self.obs_t[:, :, :, -1:]

    def _get_final_encoding(self):
        self.noise = tf.placeholder(tf.float32, [None, self.d_embedding], name='noise')
        return self.cnn_embedding + self.noise

    def __init__(
            self, *,
            obs_shape, d_embedding,
            embedding_weight=0.01,
            obs_dtype=tf.int16,
            d_classes=0,
            **conv_kwargs
    ):
        self.kwargs = {
            'obs_shape': obs_shape,
            'd_embedding': d_embedding,
            'embedding_weight': embedding_weight,
            'obs_dtype': tf.int32
        }
        self.obs_dtype = obs_dtype
        self.obs_shape = obs_shape
        self.d_embedding = d_embedding
        self.embedding_weight = embedding_weight
        self.obs_dtype = obs_dtype

        with tf.variable_scope('autoencoder') as scope:
            with tf.variable_scope('encoder') as _es:
                self.obs_t = tf.placeholder(self.obs_dtype, list((None,) + self.obs_shape), name='obs')
                processed_obs_t = self._process_obs_tensor(self.obs_t)
                h_obs, final_conv_shape = dcgan_cnn(processed_obs_t, d_embedding, **conv_kwargs)

                self.cnn_embedding = h_obs
                self._final_conv_shape = tuple(s.value for s in final_conv_shape[1:])
                self.encoding = self._get_final_encoding()
                self.encoder_scope = _es

            self.encoding_shape = tuple(s.value for s in self.encoding.shape[1:])

            with tf.variable_scope('decoder') as _ds:
                # This part of the observation model handles our class predictions
                self.preds = decoder_cnn(
                    self.encoding,
                    self._final_conv_shape,
                    0
                )
                self.decoder_scope = _ds
            self.params = tf.trainable_variables(scope='autoencoder')

        with tf.variable_scope('optimization') as _os:
            # self.nobs_t = tf.placeholder(obs_dtype, list((None,) + self.dOshape), name='nobs')
            # processed_jobs_t = self.nobs_t

            self.frame = self._get_frame()
            s = -1*(self.preds - tf.cast(self.frame, tf.float32)) ** 2
            self.loss = -tf.reduce_mean(
                # add the log probabilities to get the probability for a whole
                # image
                tf.reduce_sum(s, axis=[1, 2])
            ) + self.embedding_weight * tf.reduce_mean(
                # regularize the encoding
                tf.reduce_sum(self.encoding ** 2, axis=1)
            )

            self.lr = tf.placeholder(tf.float64, (), name='lr')
            self.step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            self.optimization_scope = _os

    def train_step(self, *, lr, obs, noise=None):
        self._check_obs(obs)
        if noise is None:
            noise = np.zeros((obs.shape[0], self.d_embedding))
        loss, _ = tf.get_default_session().run(
            [self.loss, self.step], feed_dict={
                self.lr: lr,
                self.obs_t: obs,
                self.noise: noise
            }
        )
        return loss

    def encode(self, obs, *args):
        self._check_obs(obs)
        noise = np.zeros((obs.shape[0], self.d_embedding))
        return tf.get_default_session().run(self.encoding, feed_dict={
            self.obs_t: obs,
            self.noise: noise
        })

    def decode(self, encoding):
        preds = tf.get_default_session().run(self.preds, feed_dict={
            self.encoding: encoding
        })
        img = preds
        return img[:, :, :, -1]
    
    def base_vector(self, obs, *args):
        self._check_obs(obs)
        return tf.get_default_session().run(self.cnn_embedding, feed_dict={self.obs_t: obs})

    def compare(self, obs, disp_p=0.01):
        img = self.decode(self.encode(obs))
        full_img = []
        for i in range(len(img)):
            if np.random.random() < disp_p:
                full_img.append(np.hstack([img[i], obs[i, :, :, -1]]))
        return np.vstack(full_img), np.mean((img - obs[:, :, :, -1]) ** 2)

    def save(self, save_path):
        ps = tf.get_default_session().run(self.params)
        joblib.dump({'params': ps, 'kwargs': self.kwargs}, save_path)
        
    @classmethod
    def load(cls, load_path):
        data = joblib.load(load_path)
        self = cls(**data['kwargs'])
        loaded_params = data['params']
        restores = []
        for p, loaded_p in zip(self.params, loaded_params):
            restores.append(p.assign(loaded_p))
        tf.get_default_session().run(restores)
        return self
    
class VariationalAutoEncoder:
    unk_mean = 255.0 / 2
    
    @staticmethod
    def _check_obs(obs):
        assert (obs >= 0).all()
        assert (obs <= 255).all()

    @staticmethod
    def _process_obs_tensor(obs):
        return tf.cast(obs, tf.float32)

    def _get_frame(self):
        return self.obs_t[:, :, :, -1:]

    def _get_final_encoding(self):
        self.noise = tf.placeholder(tf.float32, [None, self.d_embedding], name='noise')
        return self.cnn_embedding + self.noise

    def __init__(
            self, *,
            obs_shape, d_classes, d_embedding,
            embedding_weight=0.01,
            obs_dtype=tf.int16,
            **conv_kwargs
    ):
        self.kwargs = {
            'obs_shape': obs_shape,
            'd_classes': d_classes,
            'd_embedding': d_embedding,
            'embedding_weight': embedding_weight,
            'obs_dtype': tf.int32
        }
        self.obs_dtype = obs_dtype
        self.obs_shape = obs_shape
        self.d_classes = d_classes
        self.d_embedding = d_embedding
        self.embedding_weight = embedding_weight
        self.obs_dtype = obs_dtype

        with tf.variable_scope('autoencoder') as scope:
            with tf.variable_scope('encoder') as _es:
                self.obs_t = tf.placeholder(self.obs_dtype, list((None,) + self.obs_shape), name='obs')
                processed_obs_t = self._process_obs_tensor(self.obs_t)
                h_obs, final_conv_shape = dcgan_cnn(processed_obs_t, d_embedding, **conv_kwargs)

                self.cnn_embedding = h_obs
                self._final_conv_shape = tuple(s.value for s in final_conv_shape[1:])
                self.encoding = self._get_final_encoding()
                self.encoder_scope = _es

            self.encoding_shape = tuple(s.value for s in self.encoding.shape[1:])

            with tf.variable_scope('decoder') as _ds:
                # This part of the observation model handles our class predictions
                self.logits = decoder_cnn(
                    self.encoding,
                    self._final_conv_shape,
                    self.d_classes
                )
                #self.logits = tf.clip_by_value(self.logits, -1e6, 1e6)
                self.logp_class = tf.nn.log_softmax(self.logits)

                # this part of the observation model handles our softclass parameters
                means = tf.get_variable(
                    'mean', [self.d_classes], dtype=tf.float32,
                    initializer=tf.random_uniform_initializer(
                        #[self.d_classes],
                        maxval=255,
                        minval=-255
                    )
                )
                sigsqs = tf.clip_by_value(
                    tf.get_variable(
                        'sigsq', [self.d_classes], dtype=tf.float32,
                        initializer=tf.random_normal_initializer([self.d_classes], stddev=0.1)
                    ), 0, 10
                )
                self.dist = tf.distributions.Normal(loc=means, scale=sigsqs)

                # if we want to assign an "unknown" class, give a uniform
                # distribution over pixel values
                unk_value = tf.constant((1.0 / 255))
                if self.unk_mean == 0.0:
                    unk_value = tf.constant((1.0 / (255 * 2)))
                
                self.decoder_scope = _ds
            self.params = tf.trainable_variables(scope='autoencoder')

        with tf.variable_scope('optimization') as _os:
            # self.nobs_t = tf.placeholder(obs_dtype, list((None,) + self.dOshape), name='nobs')
            # processed_jobs_t = self.nobs_t

            self.frame = self._get_frame()

            # Calculate the log probability for the pixel value for each
            # individual class
            self.logps = tf.concat([
                # For the normal classes, it's based on the gaussian
                # distribution for each class
                self.logp_class[:, :, :, :-1] + self.dist.log_prob(
                    tf.cast(self.frame, tf.float32)
                ),
                # For the "unk" class, it's a uniform probability over pixel values
                self.logp_class[:, :, :, -1:] + tf.log(unk_value)
            ], axis=3)

            # get the log probabilities by marginalizing over our class probabiltiies
            s = tf.reduce_logsumexp(self.logps, axis=3)
            self.loss = -tf.reduce_mean(
                # add the log probabilities to get the probability for a whole
                # image
                tf.reduce_sum(s, axis=[1, 2])
            ) + self.embedding_weight * tf.reduce_mean(
                # regularize the encoding
                tf.reduce_sum(self.encoding ** 2, axis=1)
            )

            self.lr = tf.placeholder(tf.float64, (), name='lr')
            self.step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            self.optimization_scope = _os

    def train_step(self, *, lr, obs, noise=None):
        self._check_obs(obs)
        if noise is None:
            noise = np.zeros((obs.shape[0], self.d_embedding))
        loss, _ = tf.get_default_session().run(
            [self.loss, self.step], feed_dict={
                self.lr: lr,
                self.obs_t: obs,
                self.noise: noise
            }
        )
        return loss

    def encode(self, obs, *args):
        self._check_obs(obs)
        noise = np.zeros((obs.shape[0], self.d_embedding))
        return tf.get_default_session().run(self.encoding, feed_dict={
            self.obs_t: obs,
            self.noise: noise
        })

    def decode(self, encoding):
        preds, means, stds = tf.get_default_session().run(
            [self.logp_class, self.dist.loc, self.dist.scale], feed_dict={
            self.encoding: encoding
        })
        means = np.hstack([means, np.array([self.unk_mean])])
        img = (np.exp(preds) * means).sum(axis=3)
        
        return img
    
    def base_vector(self, obs, *args):
        self._check_obs(obs)
        return tf.get_default_session().run(self.cnn_embedding, feed_dict={self.obs_t: obs})

    def compare(self, obs, disp_p=0.01):
        img = self.decode(self.encode(obs))
        full_img = []
        for i in range(len(img)):
            if np.random.random() < disp_p:
                full_img.append(np.hstack([img[i], obs[i, :, :, -1]]))
        return np.vstack(full_img), np.mean((img - obs[:, :, :, -1]) ** 2)

    def save(self, save_path):
        ps = tf.get_default_session().run(self.params)
        joblib.dump({'params': ps, 'kwargs': self.kwargs}, save_path)
        
    @classmethod
    def load(cls, load_path):
        data = joblib.load(load_path)
        self = cls(**data['kwargs'])
        loaded_params = data['params']
        restores = []
        for p, loaded_p in zip(self.params, loaded_params):
            restores.append(p.assign(loaded_p))
        tf.get_default_session().run(restores)
        return self

class ScoreTrimmedVariationalAutoEncoder(VariationalAutoEncoder):
    @staticmethod
    def _check_obs(obs):
        assert (obs >= 0).all()
        assert (obs <= 255).all()
        if not (obs[:, :10, :, :] == 87).all():
            obs[:, :10, :, :] = 87

class NextStepVariationalAutoEncoder(VariationalAutoEncoder):
    unk_mean = 0.0
    
    def __init__(self, num_actions=6, **kwargs):
        self.num_actions = num_actions
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.kwargs['num_actions'] = num_actions

    def _check_acts(self, acts):
        assert len(acts.shape) == 1
        assert (acts >= 0).all()
        assert (acts < self.num_actions).all()

    def _get_frame(self):
        # Now our frame is the difference between the current
        self.nobs_t = tf.placeholder(self.obs_dtype, list((None,) + self.obs_shape), name='obs')
        return self.nobs_t[:, :, :, -1:] - self.obs_t[:, :, :, -1:]

    def _get_final_encoding(self):
        embedding = super()._get_final_encoding()
        self.acts_t = tf.placeholder(tf.int32, [None], name='actions')
        self.action_embeddings = tf.get_variable(
            'action_embeddings',
            [self.num_actions, self.d_embedding],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(
                [self.d_embedding],
                stddev=0.1
            )
        )
        self.action_modifier = tf.nn.embedding_lookup(
            self.action_embeddings, self.acts_t
        )
        return embedding + self.action_modifier

    def train_step(self, *, lr, obs, acts, nobs, noise=None):
        self._check_obs(obs)
        if noise is None:
            noise = np.zeros((obs.shape[0], self.d_embedding))
        frame, loss, _ = tf.get_default_session().run(
            [self.frame, self.loss, self.step], feed_dict={
                self.lr: lr,
                self.obs_t: obs,
                self.acts_t: acts,
                self.nobs_t: nobs,
                self.noise: noise
            }
        )
        return loss, frame

    def encode(self, obs, acts):
        self._check_obs(obs)
        self._check_acts(acts)
        noise = np.zeros((obs.shape[0], self.d_embedding))
        return tf.get_default_session().run(self.encoding, feed_dict={
            self.obs_t: obs,
            self.noise: noise,
            self.acts_t: acts
        })

    def compare(self, obs, acts, nobs, disp_p=0.01):
        img = self.decode(self.encode(obs, acts))
        full_img = []
        for i in range(len(img)):
            if np.random.random() < disp_p:
                full_img.append(np.hstack([img[i], nobs[i, :, :, -1] - obs[i, :, :, -1]]))
        return np.vstack(full_img)


def autoencode(*, tf_cfg, env_cfg):
    with utils.TfEnvContext(tf_cfg, env_cfg) as context:
        utils.logger.configure()
        vae = VariationalAutoEncoder(
            obs_shape=context.env_context.environments.observation_space.shape,
            d_classes=20,
            d_embedding=30,
            embedding_weight=0.01
        )
        LR = 1e-4

        tf.get_default_session().run(tf.local_variables_initializer())
        tf.get_default_session().run(tf.global_variables_initializer())

        buffer = deque(maxlen=500)

        env = context.env_context.environments
        env.reset()
        num_timesteps = 10000
        for i in range(num_timesteps):
            lr = LR * (num_timesteps - i) * 1.0 / num_timesteps
            obs = []
            for t in range(128):
                acts = [env.action_space.sample() for _ in range(env.num_envs)]
                obs.append(env.step(acts)[0])
            obs = np.array(obs).astype(np.uint8)
            obs[:, :, :10, :, :] = 87.0
            obs_batch = ppo2.sf01(obs)

            if i == 0:
                initial_obs = obs[:, 0, :, :, :]
                for n in range(500):
                    loss = vae.train_step(lr=2.5e-4, obs=initial_obs[:100])
                    if n % 100 == 0:
                        print(f"Initial burn in {n}/1000: {loss}")
                joblib.dump(
                    vae.compare(initial_obs[:120], disp_p=1),
                    osp.join(utils.logger.get_dir(), 'overfit_check.pkl')
                )

            if i % 100 == 0:
                joblib.dump(
                    vae.compare(obs_batch),
                    osp.join(utils.logger.get_dir(), f'img_{i}.pkl')
                )
                #for epoch in range(4):
                #    for idx in np.random.permutation([i for i in range(len(buffer))]):
                #        vae.train_step(lr=lr, obs=buffer[idx])
                if i < 1000 or i % 1000 == 0:
                    vae.save(osp.join(utils.logger.get_dir(), f'vae_{i}.pkl'))

            buffer.append(obs_batch)
            utils.logger.logkv(
                'score',
                vae.train_step(
                    lr=lr,
                    obs=buffer[np.random.randint(len(buffer))]
                )
            )

            utils.logger.dumpkvs()


if __name__ == '__main__':
    tf_config = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=8,
        inter_op_parallelism_threads=8,
        device_count={'GPU': 1},
        log_device_placement=False
    )
    tf_config.gpu_options.allow_growth = True

    env_config = {
        'env_name': 'PongNoFrameskip-v4',
        'n_envs': 8,
        'seed': 32,
        'one_hot_code': False
    }
    autoencode(tf_cfg=tf_config, env_cfg=env_config)
