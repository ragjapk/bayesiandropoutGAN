import numpy as np
import tensorflow as tf
from collections import OrderedDict, defaultdict
from bgan_util import AttributeDict
import tensorflow_probability as tfp
from dcgan_ops import *

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

class BGAN(object):

    def __init__(self, x_dim, z_dim, dataset_size, batch_size=64, alpha=0.001, lr=0.0002,
                 optimizer='adam'):
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.optimizer = optimizer.lower()
        self.alpha = alpha
        self.lr = lr
        self.weight_dims = OrderedDict([("g_h0_lin_W", (self.z_dim, 100)),
                                        ("g_h0_lin_b", (100,)),
                                        ("g_h1_lin_W", (100, 100)),
                                        ("g_h1_lin_b", (100,)),
                                        ("g_lin_W", (100, self.x_dim[0])),
                                        ("g_lin_b", (self.x_dim[0]))])

        self.K = 1 # 1 means unsupervised, label == 0 always reserved for fake
        self.build_bgan_graph()

    def _get_optimizer(self, lr):
        if self.optimizer == 'adam':
            return tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
        elif self.optimizer == 'sgd':
            return tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.5)
        else:
            raise ValueError("Optimizer must be either 'adam' or 'sgd'")

    def build_bgan_graph(self):
    
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + self.x_dim, name='real_images')        
        self.labeled_inputs = tf.placeholder(tf.float32, [self.batch_size] + self.x_dim, name='real_images_w_labels')        
        self.labels = tf.placeholder(tf.float32, [self.batch_size, self.K+1], name='real_targets')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.drop_prob = tf.placeholder(tf.float32, name='keep_prob')
        #self.z_sum = histogram_summary("z", self.z) TODO looks cool

        self.gen_param_list = []

        #Creating a generator parameter list under generator scope
        with tf.variable_scope("generator") as scope:
            self.gen_params = AttributeDict() #A dictionary to dump the generator parameters
            for name, shape in self.weight_dims.items():
                if('_W' in name):
                    self.gen_params[name] = tf.get_variable("%s" % (name), shape, initializer=tf.random_normal_initializer(stddev=1. / tf.sqrt(shape[0] / 2.)))
                else:
                    self.gen_params[name] = tf.get_variable("%s" % (name), shape, initializer=tf.random_normal_initializer(stddev=0.02))

        self.D, self.D_logits = self.discriminator(self.inputs, self.K)

        self.d_loss_real = -tf.reduce_mean(tf.log(self.D + 1e-8))        
        self.generation = defaultdict(list)        
        self.generation["generators"].append(self.generator(self.z, self.gen_params,self.drop_prob))
        self.D_, D_logits_ = self.discriminator(self.generator(self.z, self.gen_params, self.drop_prob), self.K, reuse=True)
        self.generation["d_logits"].append(D_logits_)
        self.generation["d_probs"].append(self.D_)            

        self.d_loss_fake = -tf.reduce_mean(tf.log(1-self.generation["d_probs"][0] + 1e-8))
        #print(d_loss_fake)

        g_loss_ = -tf.reduce_mean(tf.log(self.generation["d_probs"][0] + 1e-8))

        t_vars = tf.trainable_variables()

        reg_loss_g=0
        reg_loss_d=0
        with tf.variable_scope("generator") as scope:
            for name, value in self.gen_params.items():
                if('_W'  in name):
                    reg_loss_g += tf.nn.l2_loss(value)

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        
        clip_d = [w.assign(tf.clip_by_value(w, -0.01, 0.01)) for w in self.d_vars]
        self.clip_d = clip_d

        self.cnt=0
        for var in t_vars:
                if('_W'  in var.name and 'd_' in var.name):
                    reg_loss_d += tf.nn.l2_loss(var)
                    self.cnt+=1
        self.d_loss = tf.reduce_mean(self.d_loss_real + self.d_loss_fake + self.alpha * reg_loss_d)

        
        self.g_vars = []
        self.g_vars.append([var for var in t_vars if 'g_' in var.name])    

        self.g_loss = tf.reduce_mean(g_loss_ + self.alpha * reg_loss_g)

        self.d_learning_rate = tf.placeholder(tf.float32, shape=[])

        d_opt_adam = tf.train.AdamOptimizer(learning_rate=self.d_learning_rate)
        self.d_optim_adam = d_opt_adam.minimize(self.d_loss, var_list=self.d_vars)
        
        self.g_learning_rate = tf.placeholder(tf.float32, shape=[])

        g_opt_adam = tf.train.AdamOptimizer(learning_rate=self.g_learning_rate)
        self.g_optims_adam = g_opt_adam.minimize(self.g_loss, var_list=self.g_vars)
            
    def discriminator(self, x, K, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            h0 = lrelu(linear(x, 100, 'd_lin_0'))
            drop = dropout(h0, dropout_rate=0.01, name='dropout_layer', training=False)
            h1 = linear(drop, K, 'd_lin_1')
            return tf.nn.sigmoid(h1), h1
        
    '''def generator3(self, z, gen_params):
        with tf.variable_scope("generator") as scope:
            h0 = lrelu(linear(z, 2000, 'g_h0_lin', matrix=gen_params.g_h0_lin_W, bias=gen_params.g_h0_lin_b))
            drop = dropout(h0, dropout_rate=0.95, name='dropout_layer', training=True)
            h2 = linear(drop, self.x_dim[0], 'g_lin', matrix=gen_params.g_lin_W, bias=gen_params.g_lin_b)
            self.x_ = tanh(h2)
            return self.x_'''

    def generator(self, z, gen_params, drop_prob):
        with tf.variable_scope("generator") as scope:
            self.h0 = lrelu(linear(z, 100, 'g_h0_lin', matrix=gen_params.g_h0_lin_W, bias=gen_params.g_h0_lin_b))
            self.drop = dropout(self.h0, dropout_rate=drop_prob, name='dropout_layer', training=True)
            self.h1 = lrelu(linear(self.drop, 100, 'g_h1_lin', matrix=gen_params.g_h1_lin_W, bias=gen_params.g_h1_lin_b))
            self.drop = dropout(self.h1, dropout_rate=drop_prob, name='dropout_layer', training=True)
            self.x = linear(self.drop, self.x_dim[0], 'g_lin', matrix=gen_params.g_lin_W, bias=gen_params.g_lin_b)
            #self.x_ = tanh(h2)
            return self.x
