import tensorflow as tf
import numpy as np

conv_size = 4
deconv_size = 4
ndf= 64

def prelu(_x):  # code from https://stackoverflow.com/questions/39975676/how-to-implement-prelu-activation-in-tensorflow
    alphas = tf.get_variable('alpha', _x.get_shape()[-1],
            initializer=tf.constant_initializer(0.0),
            dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg

def lrelu(x, leak=0.2, name="lrelu"):
    # code from https://github.com/tensorflow/tensorflow/issues/4079
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def add_white_noise(input_tensor, mean=0, stddev=0.01):
    noise = tf.random_normal(shape= tf.shape(input_tensor), mean, stddev, dtype=tf.float32)
    return input_tensor+noise 

def encoder_all(input_tensor, reuse=False):
    if reuse:
        scope.reuse_variables()     
    output = tf.contrib.layers.conv2d(
        input_tensor, 64, conv_size, scope='convlayer1', stride =2, padding='SAME',
        activation_fn=prelu, normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params={'scale': True})
    output = tf.contrib.layers.conv2d(
        output, 128, conv_size, scope='convlayer2', stride =2, padding='SAME',
        activation_fn=prelu, normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params={'scale': True})
    output = tf.contrib.layers.conv2d(
        output, 256, conv_size, scope='convlayer3', stride =2, padding='SAME',
        activation_fn=prelu, normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params={'scale': True})
    output = tf.contrib.layers.conv2d(
        output, 512, conv_size, scope='convlayer4', stride =2, padding='SAME',
        activation_fn=prelu, normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params={'scale': True})
    output = tf.contrib.layers.conv2d(
        output, 1024, conv_size, scope='convlayer5', stride =2, padding='SAME',
        activation_fn=prelu, normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params={'scale': True})
    output = tf.contrib.layers.conv2d(
        output, 1024, conv_size, scope='convlayer6', stride =2, padding='SAME',
        activation_fn=None, normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params={'scale': True}) 
    output = tf.contrib.layers.flatten(output)
    output = prelu(output)
    return output


def encoder_x_r(input_tensor, output_size, reuse=False):
    if reuse:
        scope.reuse_variables()
    output = tf.contrib.layers.fully_connected(input_tensor, output_size, activation_fn=None,
        normalizer_fn=tf.contrib.layers.batch_norm, normalizer_params={'scale': True})
 #   output = tf.contrib.layers.dropout(output, 0.9, scope='dropout1')
    return output

def encoder_x_r_s(input_tensor, output_size_r, nclass, output_size_s):
    output_y = tf.contrib.layers.fully_connected(input_tensor, nclass, scope='class_deconv_1',
        activation_fn= tf.nn.log_softmax, normalizer_fn=tf.contrib.layers.batch_norm, 
        normalizer_params={'scale': True})
    output_s = tf.contrib.layers.fully_connected(input_tensor, output_size_s,
        scope='s_deconv_1', activation_fn= None, normalizer_fn=tf.contrib.layers.batch_norm, 
        normalizer_params={'scale': True})
    output_r = tf.contrib.layers.fully_connected(input_tensor, output_size_r,
        scope='r_deconv_1', activation_fn= None, normalizer_fn=tf.contrib.layers.batch_norm, 
        normalizer_params={'scale': True})
    return output_y, output_s, output_r

def decoder_all(input_sensor, chan_out, reuse=False):
    if reuse:
        scope.reuse_variables()    
    output = tf.expand_dims(input_sensor ,1)
    output = tf.expand_dims(output,1)
    output = tf.contrib.layers.fully_connected(ouput,1024*4*4, activation_fn=None)
    output = tf.reshape(output,[-1,1024,4,4])
    output = prelu(output)
    output = tf.contrib.layers.conv2d_transpose(    
        output, 1024, deconv_size, scope='deconv1', stride = 2, padding='SAME',
        activation_fn=prelu, normalizer_fn=tf.contrib.layers.batch_norm, 
        normalizer_params={'scale': True})
    output = tf.contrib.layers.conv2d_transpose(
        output, 512, deconv_size, scope='deconv2', stride = 2, padding='SAME',
        activation_fn=prelu, normalizer_fn=tf.contrib.layers.batch_norm, 
        normalizer_params={'scale': True})
    output = tf.contrib.layers.conv2d_transpose(
        output, 256, deconv_size, scope='deconv3', stride = 2, padding='SAME',
        activation_fn=prelu, normalizer_fn=tf.contrib.layers.batch_norm, 
        normalizer_params={'scale': True})
    output = tf.contrib.layers.conv2d_transpose(
        output, 128, deconv_size, scope='deconv4', stride = 2, padding='SAME',
        activation_fn=prelu, normalizer_fn=tf.contrib.layers.batch_norm, 
        normalizer_params={'scale': True})
    output = tf.contrib.layers.conv2d_transpose(
        output, 64, deconv_size, scope='deconv5', stride=2, padding='SAME',
        activation_fn=prelu, normalizer_fn=tf.contrib.layers.batch_norm, 
        normalizer_params={'scale': True})
    output = tf.contrib.layers.conv2d_transpose(
        output, chan_out, deconv_size, scope='deconv6', stride=2, padding='SAME',
        activation_fn=tf.nn.sigmoid, normalizer_fn=tf.contrib.layers.batch_norm, 
        normalizer_params={'scale': True})    
    return output

def createAdversary(input_tensor, reuse= False):
    if reuse:
        scope.reuse_variables()        
    output = tf.contrib.layers.fully_connected(input_tensor, 1024, scope='full1',
        activation_fn=lrelu)
    output = tf.contrib.layers.fully_connected(output, 1024, scope='full2',
        activation_fn=lrelu, normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params={'scale': True})
    output = tf.contrib.layers.fully_connected(output, 512, scope='full3',
        activation_fn=lrelu, normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params={'scale': True})
    output = tf.contrib.layers.fully_connected(output, 1, scope='full4',
        activation_fn=tf.nn.sigmoid, normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params={'scale': True})
    return output


def createAdversary_Dec(input_tensor, gan_noise=0.01, noise_bool=False, reuse=False):
    if reuse:
        scope.reuse_variables()    
    if gan_noise > 0:
        input_tensor = add_white_noise(input_tensor)
    output = tf.contrib.layers.conv2d(
        input_tensor, ndf, conv_size, scope='convlayer1', stride =2, padding='SAME',
        activation_fn=lrelu)
    if noise_bool == True and gan_noise>0:
        output = add_white_noise(output)
    output = tf.contrib.layers.conv2d(
        output, ndf*2, conv_size, scope='convlayer2', stride =2, padding='SAME',
        activation_fn=lrelu, normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params={'scale': True})
    if noise_bool == True and gan_noise>0:
        output = add_white_noise(output)    
    output = tf.contrib.layers.conv2d(
        output, ndf*4, conv_size, scope='convlayer3', stride =2, padding='SAME',
        activation_fn=lrelu, normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params={'scale': True})        
    if noise_bool == True and gan_noise>0:
        output = add_white_noise(output)    
    output = tf.contrib.layers.conv2d(
        output, ndf*8, conv_size, scope='convlayer4', stride =2, padding='SAME',
        activation_fn=lrelu, normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params={'scale': True})       
    if noise_bool == True and gan_noise>0:
        output = add_white_noise(output)    
    output = tf.contrib.layers.conv2d(
        output, ndf*8, conv_size, scope='convlayer5', stride =2, padding='SAME',
        activation_fn=lrelu, normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params={'scale': True})  
    if gan_noise >0:
        output=add_white_noise(output)
    return output

def Adv_dec_x_r(input_tensor, reuse=False):
    if reuse:
        scope.reuse_variables()  
    output= tf.contrib.layers.flatten(input_tensor)
    output = tf.contrib.layers.fully_connected(output, 1, scope='decd_r_full1',
        activation_fn=tf.nn.sigmoid)
    return output

def Adv_dec_x_r_s(input_tensor, nclass, reuse=False):
    if reuse:
        scope.reuse_variables()
    output = tf.contrib.layers.flatten(input_tensor)
    output = tf.contrib.layers.fully_connected(output, nclass+1, scope='decd_rs_full1',
        activation_fn=tf.nn.log_softmax)
    return output

def log_likelihood_gaussian(sample, mean, sigma):
    '''
    compute log(sample~Gaussian(mean, sigma^2))
    '''
    return -log2pi*tf.cast(sample.shape[1].value, tf.float32)/2\
        -tf.reduce_sum(tf.square((sample-mean)/sigma) + 2*tf.log(sigma), 1)/2

def log_likelihood_prior(sample):
    '''
    compute log(sample~Gaussian(0, I))
    '''
    return -log2pi*tf.cast(sample.shape[1].value, tf.float32)/2\
         -tf.reduce_sum(tf.square(sample), 1)/2

def parzen_cpu_batch(x_batch, samples, sigma, batch_size, num_of_samples, data_size):
    '''
    x_batch:    a data batch (batch_size, data_size), data_size = h*w*c for images
    samples:    generated data (num_of_samples, data_size)
    sigma:      standard deviation (float32)
    '''
    x = x_batch.reshape((batch_size, 1, data_size))
    mu = samples.reshape((1, num_of_samples, data_size))
    a = (x - mu)/sigma # (batch_size, num_of_samples, data_size)

    # sum -0.5*a^2
    tmp = -0.5*(a**2).sum(2) # (batch_size, num_of_samples)
    # log_mean_exp trick
    max_ = np.amax(tmp, axis=1, keepdims=True) # (batch_size, 1)
    E = max_ + np.log(np.mean(np.exp(tmp - max_), axis=1, keepdims=True)) # (batch_size, 1)
    # Z = dim * log(sigma * sqrt(2*pi)), dim = data_size
    Z = data_size * np.log(sigma * np.sqrt(np.pi * 2))
    return E-Z