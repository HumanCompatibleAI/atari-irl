from atari_irl import utils
import tensorflow as tf
import numpy as np
from baselines.a2c.utils import conv, fc, conv_to_fc


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


class VariationalAutoEncoder:
    @staticmethod
    def _check_obs(obs):
        assert (obs >= 0).all()
        assert (obs <= 255).all()

    @staticmethod
    def _process_obs_tensor(obs):
        return tf.cast(obs, tf.float32)

    def __init__(self, *, obs_shape, d_classes, d_embedding, obs_dtype=tf.int32, **conv_kwargs):
        self.obs_dtype = obs_dtype
        self.obs_shape = obs_shape
        self.d_classes = d_classes
        self.d_embedding = d_embedding

        with tf.variable_scope('encoder') as _es:
            self.obs_t = tf.placeholder(obs_dtype, list((None,) + self.obs_shape), name='obs')
            processed_obs_t = self._process_obs_tensor(self.obs_t)
            h_obs, final_conv_shape = dcgan_cnn(processed_obs_t, d_embedding, **conv_kwargs)

            self.encoding = h_obs
            self._final_conv_shape = tuple(s.value for s in final_conv_shape[1:])
            self.encoder_scope = _es

        self.encoding_shape = tuple(s.value for s in self.encoding.shape[1:])

        with tf.variable_scope('decoder') as _ds:
            self.noise = tf.placeholder(tf.float32, [None, self.d_embedding], name='noise')
            # This part of the observation model handles our class predictions
            self.logits = decoder_cnn(
                self.encoding + self.noise,
                self._final_conv_shape,
                self.d_classes
            )
            self.logp_class = tf.nn.log_softmax(self.logits)

            # this part of the observation model handles our softclass parameters
            means = tf.get_variable(
                'mean', [self.d_classes], dtype=tf.float32,
                initializer=tf.random_uniform_initializer([self.d_classes], maxval=255)
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
            self.decoder_scope = _ds

        with tf.variable_scope('optimization') as _os:
            # self.nobs_t = tf.placeholder(obs_dtype, list((None,) + self.dOshape), name='nobs')
            # processed_jobs_t = self.nobs_t

            frame = self.obs_t[:, :, :, -1:]

            # Calculate the log probability for the pixel value for each
            # individual class
            self.logps = tf.concat([
                # For the normal classes, it's based on the gaussian
                # distribution for each class
                self.logp_class[:, :, :, :-1] + self.dist.log_prob(tf.cast(frame, tf.float32)),
                # For the "unk" class, it's a uniform probability over pixel values
                self.logp_class[:, :, :, -1:] + tf.log(unk_value)
            ], axis=3)

            # get the log probabilities by marginalizing over our class probabiltiies
            s = tf.reduce_logsumexp(self.logps, axis=3)
            self.loss = -tf.reduce_mean(
                # add the log probabilities to get the probability for a whole
                # image
                tf.reduce_sum(s, axis=[1, 2])
            ) + .01 * tf.reduce_mean(
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

    def encode(self, obs):
        self._check_obs(obs)
        return tf.get_default_session().run(self.encoding, feed_dict={
            self.obs_t: obs
        })

    def decode(self, encoding):
        noise = np.zeros(encoding.shape)
        return tf.get_default_session().run([self.logp_class], feed_dict={
            self.encoding: encoding,
            self.noise: noise
        })


def autoencode(*, tf_cfg, env_config):
    with utils.TfEnvContext(tf_cfg, env_config) as context:
        utils.logger.configure()
        vae = VariationalAutoEncoder(
            obs_shape=context.env_context.environments.observation_space.shape,
            d_classes=20,
            d_embedding=30
        )


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
    autoencode(tf_cfg=tf_config, env_config=env_config)