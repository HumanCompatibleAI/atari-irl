from atari_irl import encoding, utils
import tensorflow as tf
import joblib
from airl.models.architectures import relu_net
import os.path as osp
import numpy as np
from gym.spaces import Discrete
from baselines.common.distributions import make_pdtype

cnn_fn = lambda obs_tensor, n_actions: encoding.dcgan_cnn(obs_tensor, n_actions)[0]
relu_fn = lambda obs_tensor, n_actions: relu_net(obs_tensor, layers=4, dout=n_actions)

class Cloner:
    def __init__(self, *, obs_shape, n_actions, encoding_fn=cnn_fn, **conv_kwargs):
        self.obs_dtype = tf.float32
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        
        self.kwargs = {
            'obs_shape': obs_shape,
            'n_actions': n_actions
        }
        
        with tf.variable_scope('behavioral_cloning') as scope:
            self.obs_t = tf.placeholder(self.obs_dtype, list((None,) + self.obs_shape), name='obs')
            self.logits = encoding_fn(self.obs_t, self.n_actions)
            self.logp_class = tf.nn.log_softmax(self.logits)
            
            self.act_t = tf.placeholder(tf.float32, [None, self.n_actions], name='act')
            
            # Optimization
            log_loss_example = tf.reduce_sum(self.act_t * self.logp_class, 1, keepdims=True)#[1, 2])
            print(log_loss_example.shape)
            self.loss = -tf.reduce_mean(log_loss_example)
            self.lr = tf.placeholder(tf.float64, (), name='lr')
            self.opt_step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            
            # Actions
            self.pd, self.pi = make_pdtype(Discrete(n_actions)).pdfromlatent(self.logp_class)
            self.a = self.pd.sample()
            self.neglogp = self.pd.neglogp(self.a)
            
        self.params = tf.trainable_variables(scope='behavioral_cloning')
            
    def save(self, save_path):
        print(f"Saving to {save_path}")
        ps = tf.get_default_session().run(self.params)
        joblib.dump({'params': ps, 'kwargs': self.kwargs}, save_path)
        
    @classmethod
    def load(cls, load_path, **kwargs):
        data = joblib.load(load_path)
        kwargs.update(data['kwargs'])
        self = cls(**kwargs)
        loaded_params = data['params']
        restores = []
        for p, loaded_p in zip(self.params, loaded_params):
            restores.append(p.assign(loaded_p))
        tf.get_default_session().run(restores)
        return self

            
    def train_step(self, *, lr, obs, act):
        loss, _ = tf.get_default_session().run(
            [self.loss, self.opt_step], feed_dict={
                self.obs_t: obs,
                self.act_t: act,
                self.lr: lr
            }
        )
        return loss
        
    def step(self, obs):
        actions, neglogps = tf.get_default_session().run(
            [self.a, self.neglogp], feed_dict={
                self.obs_t: obs
            }
        )
        return actions, np.zeros(obs.shape[0]), None, neglogps
    
    def check(self, obs):
        actions, neglogps, logits = tf.get_default_session().run(
            [self.a, self.neglogp, self.logits], feed_dict={
                self.obs_t: obs
            }
        )
        return actions, neglogps, logits
    
    def train(self, *, obs, act, lr=1e-4, batch_size=1024, epochs=500):
        T = obs.shape[0]
        n_batches = (T // batch_size) - 1
        print(f"Splitting {T} timesteps into {n_batches} batches")
        
        for e in range(epochs):
            order = np.random.permutation(T)
            obs = obs[order]
            act = act[order]
            losses = []

            for b in range((T // batch_size) - 1):
                lr *= .9995
                s = slice(batch_size*b, batch_size*(b+1))
                loss = self.train_step(lr=lr, obs=obs[s], act=act[s])
                if b % 50 == 0:
                    print(f"Epoch {e} batch {b}: {loss}")
                losses.append(loss)
            print(f"Epoch {e}: {np.mean(losses)}")
