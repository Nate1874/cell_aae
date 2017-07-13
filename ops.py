import tensorflow as tf
import numpy as np

conv_size = 4
deconv_size = 4
ndf= 64

def prelu(_x, scope='prelu/'):  # code from https://stackoverflow.com/questions/39975676/how-to-implement-prelu-activation-in-tensorflow
    alphas = tf.get_variable(scope+'alpha', _x.get_shape()[-1],
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
    noise = tf.random_normal(shape= tf.shape(input_tensor), mean= mean, stddev =stddev, dtype=tf.float32)
    return input_tensor+noise 

def encoder_all(input_tensor):  
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


def encoder_x_r(input_tensor, output_size):
    
    output = tf.contrib.layers.fully_connected(input_tensor, output_size, activation_fn=None,
        normalizer_fn=tf.contrib.layers.batch_norm, normalizer_params={'scale': True})
 #   output = tf.contrib.layers.dropout(output, 0.9, scope='dropout1')
    return output

def encoder_x_r_s(input_tensor, output_size_r, nclass, output_size_s):
    output_y = tf.contrib.layers.fully_connected(input_tensor, nclass, scope='class_deconv_1',
        activation_fn= None, normalizer_fn=tf.contrib.layers.batch_norm, 
        normalizer_params={'scale': True})
    output_s = tf.contrib.layers.fully_connected(input_tensor, output_size_s,
        scope='s_deconv_1', activation_fn= None, normalizer_fn=tf.contrib.layers.batch_norm, 
        normalizer_params={'scale': True})
    output_r = tf.contrib.layers.fully_connected(input_tensor, output_size_r,
        scope='r_deconv_1', activation_fn= None, normalizer_fn=tf.contrib.layers.batch_norm, 
        normalizer_params={'scale': True})
    return output_y, output_s, output_r

def decoder_all(input_sensor, chan_out):
 #   output = tf.expand_dims(input_sensor ,1)
#    output = tf.expand_dims(output,1)
    output = tf.contrib.layers.fully_connected(input_sensor,1024*4*4, activation_fn=None)
    print(output.get_shape())
    output = tf.reshape(output,[-1,1024,4,4])
    output = tf.transpose(output, perm=[0, 2, 3 ,1])
    print(output.get_shape())
    output = prelu(output)
    print(output.get_shape())
    output = tf.contrib.layers.conv2d_transpose(    
        output, 1024, deconv_size, scope='deconv1', stride = 2, padding='SAME',
        activation_fn=prelu, normalizer_fn=tf.contrib.layers.batch_norm, 
        normalizer_params={'scale': True})
    print(output.get_shape())
    output = tf.contrib.layers.conv2d_transpose(
        output, 512, deconv_size, scope='deconv2', stride = 2, padding='SAME',
        activation_fn=prelu, normalizer_fn=tf.contrib.layers.batch_norm, 
        normalizer_params={'scale': True})
    print(output.get_shape())
    output = tf.contrib.layers.conv2d_transpose(
        output, 256, deconv_size, scope='deconv3', stride = 2, padding='SAME',
        activation_fn=prelu, normalizer_fn=tf.contrib.layers.batch_norm, 
        normalizer_params={'scale': True})
    print(output.get_shape())
    output = tf.contrib.layers.conv2d_transpose(
        output, 128, deconv_size, scope='deconv4', stride = 2, padding='SAME',
        activation_fn=prelu, normalizer_fn=tf.contrib.layers.batch_norm, 
        normalizer_params={'scale': True})
    print(output.get_shape())
    output = tf.contrib.layers.conv2d_transpose(
        output, 64, deconv_size, scope='deconv5', stride=2, padding='SAME',
        activation_fn=prelu, normalizer_fn=tf.contrib.layers.batch_norm, 
        normalizer_params={'scale': True})
    print(output.get_shape())
    output = tf.contrib.layers.conv2d_transpose(
        output, chan_out, deconv_size, scope='deconv6', stride=2, padding='SAME',
        activation_fn=tf.nn.sigmoid, normalizer_fn=None)
    print(output.get_shape())  # use sigmoid?
    return output

def super_resolution(inputs , out_ch):
    conv1 = conv2d(inputs, 32, 5, 1, '/conv1', False) # or use false?
    print('conv1======================', conv1.get_shape())
    residual1 = residual_block(conv1, 32, 3, '/residual1')
    print('residual1======================', residual1.get_shape())
    residual2 = residual_block(residual1, 32, 3, '/residual2')
    print('residual2======================', residual2.get_shape())
    residual3 = residual_block(residual2, 32, 3, '/residual3')
    print('residual3======================', residual3.get_shape())
    residual4 = residual_block(residual3, 32, 3, '/residual4')
    print('residual4======================', residual4.get_shape())
    residual5 = residual_block(residual4, 32, 3, '/residual5')
    print('residual5======================', residual5.get_shape())
    conv2 = conv2d(residual5, 32, 3, 1, '/conv2', True)
    print('conv2======================', conv2.get_shape())
    conv2 = conv2 + conv1
    conv3 = conv2d(conv2, 128, 3, 1, '/conv3', True)
    print('conv3======================', conv3.get_shape())
    conv5 = conv2d(conv3, 32, 3, 1, '/conv5', False)
    print('conv5======================', conv5.get_shape())
    conv6 = conv2d(conv5, out_ch, 3, 1, '/conv6', False)
    print('conv6======================', conv6.get_shape())
    result = tf.nn.sigmoid(conv6)
    return result 

def residual_block(incoming, num_outputs, kernel_size, scope, data_format = 'NHWC'):
    conv1 = tf.contrib.layers.conv2d(
        incoming, num_outputs, kernel_size, scope=scope+'/conv1',
        data_format=data_format, activation_fn=None, biases_initializer=None)
    conv1_bn = tf.contrib.layers.batch_norm(
            conv1, decay=0.9, center=True, activation_fn=prelu,
            updates_collections=None, epsilon=1e-5, scope=scope+'/batch_norm1',
            data_format=data_format)
    conv2 = tf.contrib.layers.conv2d(
        conv1, num_outputs, kernel_size, scope=scope+'/conv2',
        data_format=data_format, activation_fn=None, biases_initializer=None)
    conv2_bn = tf.contrib.layers.batch_norm(
            conv2, decay=0.9, center=True, activation_fn=None,
            updates_collections=None, epsilon=1e-5, scope=scope+'/batch_norm2',
            data_format=data_format)
    incoming += conv2_bn
    return prelu(incoming,scope)


def conv2d(inputs, num_outputs, kernel_size, stride, scope, norm=True, ac_fn = prelu, 
           d_format='NHWC'):
    outputs = tf.contrib.layers.conv2d(
        inputs, num_outputs, kernel_size, scope=scope, stride=stride,
        data_format=d_format, activation_fn=None, biases_initializer=None)
    if norm:
        outputs = tf.contrib.layers.batch_norm(
            outputs, decay=0.9, center=True, activation_fn=ac_fn,
            updates_collections=None, epsilon=1e-5, scope=scope+'/batch_norm',
            data_format=d_format)
    else:
        outputs = prelu(outputs,scope)
    return outputs



def createAdversary(input_tensor):        
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


def createAdversary_Dec(input_tensor, gan_noise=0.01, noise_bool=False):
    if gan_noise > 0:
        input_tensor = add_white_noise(input_tensor)
    output = tf.contrib.layers.conv2d(
        input_tensor, ndf, conv_size, scope='convlayer1', stride =2, padding='SAME',
        activation_fn=lrelu)
    if noise_bool and gan_noise>0:
        output = add_white_noise(output)
    output = tf.contrib.layers.conv2d(
        output, ndf*2, conv_size, scope='convlayer2', stride =2, padding='SAME',
        activation_fn=lrelu, normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params={'scale': True})
    if noise_bool  and gan_noise>0:
        output = add_white_noise(output)    
    output = tf.contrib.layers.conv2d(
        output, ndf*4, conv_size, scope='convlayer3', stride =2, padding='SAME',
        activation_fn=lrelu, normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params={'scale': True})        
    if noise_bool and gan_noise>0:
        output = add_white_noise(output)    
    output = tf.contrib.layers.conv2d(
        output, ndf*8, conv_size, scope='convlayer4', stride =2, padding='SAME',
        activation_fn=lrelu, normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params={'scale': True})       
    if noise_bool and gan_noise>0:
        output = add_white_noise(output)    
    output = tf.contrib.layers.conv2d(
        output, ndf*8, conv_size, scope='convlayer5', stride =2, padding='SAME',
        activation_fn=lrelu, normalizer_fn=tf.contrib.layers.batch_norm,
        normalizer_params={'scale': True})  
    if gan_noise >0:
        output=add_white_noise(output)
    return output

def Adv_dec_x_r(input_tensor):
    output= tf.contrib.layers.flatten(input_tensor)
    output = tf.contrib.layers.fully_connected(output, 1, scope='decd_r_full1',
        activation_fn=tf.nn.sigmoid) #
    return output

def Adv_dec_x_r_s(input_tensor, nclass):
    output = tf.contrib.layers.flatten(input_tensor)
    output = tf.contrib.layers.fully_connected(output, nclass+1, scope='decd_rs_full1',
        activation_fn=None)  # should be None?
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

def get_ssim_loss(img1, img2, size=11, sigma=1.5):
    img1s = tf.split(img1, img1.shape[-1].value, axis=3)
    img2s = tf.split(img2, img2.shape[-1].value, axis=3)
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 1
    C1 = (K1 * L)**2
    C2 = (K2 * L)**2
    values = []
    for index in range(img1.shape[-1].value):
        mu1 = tf.nn.conv2d(img1s[index], window, strides=[1,1,1,1], padding='VALID')
        mu2 = tf.nn.conv2d(img2s[index], window, strides=[1,1,1,1], padding='VALID')
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = tf.nn.conv2d(img1s[index]*img1s[index], window, strides=[1,1,1,1], padding='VALID') - mu1_sq
        sigma2_sq = tf.nn.conv2d(img2s[index]*img2s[index], window, strides=[1,1,1,1], padding='VALID') - mu2_sq
        sigma12 = tf.nn.conv2d(img1s[index]*img2s[index], window, strides=[1,1,1,1], padding='VALID') - mu1_mu2
        value = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))
        values.append(tf.reduce_mean(value))
    return (1-tf.reduce_mean(values))/2

def fspecial_gauss(size, sigma):
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)
    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)
    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)