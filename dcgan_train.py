
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[6]:


# Noise dimensions
zdim = 50

# Samples
Nsamples = 100


# In[7]:


# Load MNIST data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)


# In[8]:


def leaky_relu(x, leak=0.2):
    return tf.maximum(x, x * leak)

def conv2d(inputs, filters, bn=True):
    out = tf.layers.conv2d(inputs, filters, [5, 5], strides=(2, 2), padding='SAME')    
    if bn:
        return leaky_relu(tf.layers.batch_normalization(out))
    else:
        return leaky_relu(out)
        
def conv2dtrans(inputs, filters, activation, bn=True):
    out = tf.layers.conv2d_transpose(inputs, filters, [5, 5], strides=(2, 2), padding='SAME')                     
    if bn:
        return activation(tf.layers.batch_normalization(out))
    else:
        return activation(out)


# In[9]:


def generator(input, ch, reuse=False): # input is z, shape = [None, 100]
    with tf.variable_scope("generator", reuse=reuse):
        g = tf.layers.dense(input, 7*7*ch[0])  # layer1: project        
        g = tf.reshape(g, [-1, 7, 7, ch[0]])
        g = tf.nn.relu(tf.layers.batch_normalization(g))
        
        g = conv2dtrans(g, ch[1], tf.nn.relu)  # layer2: conv_trans1 (128 --> 64)
        g = conv2dtrans(g, ch[2], tf.nn.relu)  # layer3: conv_trans2 (64 --> 1)
        return g
    
def discriminator(input, ch, reuse=False): # input is x or G(z), shape = [None, 1, 28 ,28]
    with tf.variable_scope("discriminator", reuse=reuse):
        d = conv2d(input, ch[0]) # layer1: conv1 (1 --> 64)
        d = conv2d(d, ch[1]) # layer2: conv2 (64 --> 128)
        d = tf.reshape(d, [-1, 7*7*128])
        d = tf.layers.dense(d, ch[2])
        prob = tf.nn.sigmoid(d, name="d_out")
        return d, prob


# In[10]:


def inference(x, z, g_ch, d_ch):
    g = generator(z, g_ch)
    g_demo = tf.identity(generator(z, g_ch, reuse=True), name="g_out")
    d0, prob0 = discriminator(g, d_ch)
    d1, prob1 = discriminator(x, d_ch, reuse=True)    
    return g, g_demo, d0, d1

def loss(d0, d1):    
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d0, labels=tf.ones_like(d0)))
    d_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d1, labels=tf.ones_like(d1)))
    d_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d0, labels=tf.zeros_like(d0)))
    d_loss = d_loss1 + d_loss2    
    return g_loss, d_loss

def training(g_loss, d_loss, lr=0.0002, beta1=0.5):
    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if 'generator' in var.name] # Generator variables (used during G update)
    d_vars = [var for var in t_vars if 'discriminator' in var.name] # Discriminator variables (used during D update)

    g_opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1)
    d_opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1)
    g_opt_op = g_opt.minimize(loss=g_loss, var_list=g_vars)
    d_opt_op = d_opt.minimize(loss=d_loss, var_list=d_vars)

    return g_opt_op, d_opt_op


# In[11]:


def draw_sample(m, n):
    return np.random.uniform(-1.0, 1.0, size=[m, n])

def placeholder_inputs(Nsamples):
    # Input to the discriminator
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='x')
    # Input to the generator
    z = tf.placeholder(tf.float32, shape=[None, zdim], name='z')    
    return x, z

def fill_feed_dict_d(x_pl, z_pl, dataset, Nsamples, zdim):
    x_feed, _ = dataset.next_batch(Nsamples)
    x_feed = np.reshape(x_feed, [Nsamples, 28, 28, 1])
    z_feed = draw_sample(Nsamples, zdim)
    return {x_pl: x_feed, z_pl: z_feed }

def fill_feed_dict_g(z_pl, Nsamples, zdim):
    z_feed = draw_sample(Nsamples, zdim)
    return {z_pl: z_feed }


# In[ ]:


# DCGAN architecture
g_channels = [128, 64, 1];
d_channels = [64, 128, 1];

tf.reset_default_graph()

x, z = placeholder_inputs(Nsamples)
g, g_demo, d0, d1 = inference(x ,z, g_channels, d_channels)
g_loss, d_loss = loss(d0, d1)
g_opt, d_opt = training(g_loss, d_loss)


# In[ ]:


saver = tf.train.Saver()

log_d_loss = list()
log_g_loss = list()
log_iteration = list()

with tf.Session() as sess:
    
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # training iterations
    for i in range(int(1e3)):
        fd_d = fill_feed_dict_d(x, z, mnist.train, Nsamples, zdim)
        fd_g = fill_feed_dict_g(z, Nsamples, zdim)

        # 1. Update the discriminator using real and generated samples
        _, d_loss_val = sess.run([d_opt, d_loss], feed_dict=fd_d)

        # 2. Update the generator
        _, g_loss_val = sess.run([g_opt, g_loss], feed_dict=fd_g)
        
        if (i+1) % int(1e2) == 0:            
            log_iteration.append(i+1)
            log_d_loss.append(d_loss_val)
            log_g_loss.append(g_loss_val)
            
        if (i+1) % int(1e3) == 0:
        print('Iteration {}, Discriminator Loss {:.3}, Generator Loss {:.3}'.format(i+1, d_loss_val, g_loss_val))

        if (i+1) % int(1e3) == 0:
            print('Saving model...')
            saver.save(sess, './models/model_iter', global_step=i+1)
            
                    
log_iteration = np.asarray(log_iteration)
log_d_loss = np.asarray(log_d_loss)
log_g_loss = np.asarray(log_g_loss)
log = np.column_stack((log_iteration, log_d_loss, log_g_loss))
np.save('./models/loss_log', log)


# In[ ]:




