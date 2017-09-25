from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
import numpy as np
import math
from ops import *
import time
from data_reader import data_reader
from progressbar import ETA, Bar, Percentage, ProgressBar
from scipy.misc import imsave, imread
from data_reader import data_reader

class GAN(object):

    def __init__(self, sess, flag):
        self.conf = flag
        self.sess = sess
        self.chan_out_r = 2
        self.chan_out_r_s =3
        if not os.path.exists(self.conf.modeldir):
            os.makedirs(self.conf.modeldir)
        if not os.path.exists(self.conf.logdir):
            os.makedirs(self.conf.logdir)
        if not os.path.exists(self.conf.sampledir):
            os.makedirs(self.conf.sampledir)
        print("Start building network=================")
        self.configure_networks()
        print("Finishing building network=================")
    
    def configure_networks(self):
        self.build_network()
        variables = tf.trainable_variables()

        self.var_gen = [var for var in variables if var.name.startswith('Generator')]
        self.var_disc = [var for var in variables if var.name.startswith('Discriminator')]

        self.train_disc = tf.contrib.layers.optimize_loss(self.dis_loss, tf.contrib.framework.get_or_create_global_step(), 
            learning_rate=self.conf.learning_rate, optimizer='Adam', variables=self.var_disc, update_ops=[])
        self.train_gen = tf.contrib.layers.optimize_loss(self.gen_loss, tf.contrib.framework.get_or_create_global_step(), 
            learning_rate=self.conf.learning_rate, optimizer='Adam', variables=self.var_gen, update_ops=[])    
  #      self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)
        self.writer = tf.summary.FileWriter(self.conf.logdir, self.sess.graph)
        self.train_summary = self.config_summary()
    #    self.train_con_summary =self.config_con_summary()
    #    self.test_summary = self.config_test_summary()

    def build_network(self):

        self.sampled_z_s = tf.placeholder(tf.float32,[None, self.conf.hidden_size])
        self.input_y = tf.placeholder(tf.int32,[None,self.conf.n_class])    
        self.input_latent_r = tf.placeholder(tf.float32,[None, self.conf.hidden_size])
        self.input_x = tf.placeholder(tf.float32,[None, self.conf.height, self.conf.width, 3])

        self.input_y= tf.cast(self.input_y, tf.float32)

        print("Start building the generator of the ConGAN========================")
        #build the conditional auto encoder
        with tf.variable_scope('Generator') as scope:
            self.X_rec = generator(self.sampled_z_s, self.input_latent_r, self.input_y, self.conf.batch_size) # only s channel
        print("=========================Now split and insert")
        self.ch1, self.ch2_, self.ch3 = tf.split(self.input_x, num_or_size_splits=3, axis= 3)
        print(self.X_rec.get_shape())
        print(self.ch1.get_shape())
        self.X_rec = tf.concat([self.ch1, self.X_rec, self.ch3], axis= 3) 
        print(self.X_rec.get_shape())

        with tf.variable_scope('Discriminator') as scope:
            self.out_real = discriminator(self.input_x, self.input_y, self.conf.batch_size)
            scope.reuse_variables()
            self.out_fake = discriminator(self.X_rec,  self.input_y, self.conf.batch_size)
        

        # the loss for the conditional auto encoder
        self.d_loss_real = self.get_bce_loss(self.out_real, tf.ones_like(self.out_real))
        self.d_loss_fake = self.get_bce_loss(self.out_fake, tf.zeros_like(self.out_fake))
        # Do we need to add the classification loss??????????????????????????
        self.g_loss = self.get_bce_loss(self.out_fake, tf.ones_like(self.out_fake))
        self.rec_loss = self.get_mse_loss(self.X_rec, self.input_x)

        # build the model for the final conditional generation
        
        self.dis_loss= self.d_loss_fake+self.d_loss_real
        self.gen_loss= self.rec_loss + self.g_loss*self.conf.gamma_gen

        self.test_r = tf.placeholder(tf.float32,[None, self.conf.hidden_size])
        self.test_y = tf.placeholder(tf.int32,[None,self.conf.n_class])
   #     self.test_label = tf.placeholder(tf.float32,[None, self.conf.height, self.conf.width, 3])
        random_s_test= tf.random_normal([self.conf.batch_size,self.conf.hidden_size])
        self.test_y = tf.cast(self.test_y, tf.float32)
        with tf.variable_scope('Generator', reuse= True) as scope:
            self.test_out = generator(random_s_test, self.test_y, self.test_r, self.conf.batch_size)

        
       

    def config_summary(self):
        summarys = []                      
        summarys.append(tf.summary.scalar('/Rec_loss', self.rec_loss))
        summarys.append(tf.summary.scalar('/d_loss_real', self.d_loss_real))
        summarys.append(tf.summary.scalar('/d_loss_fake', self.d_loss_fake))
        summarys.append(tf.summary.scalar('/d_loss', self.dis_loss))
        summarys.append(tf.summary.scalar('/g_loss', self.g_loss)) 
        summarys.append(tf.summary.scalar('/generator_loss', self.gen_loss))
        summarys.append(tf.summary.image('input_X', self.input_x, max_outputs = 10))
        summarys.append(tf.summary.image('recon_X', self.X_rec, max_outputs = 10))        
        summary = tf.summary.merge(summarys)
        return summary

    # def config_test_summary(self):
    #     summarys= []
    #     input_ch1, input_ch2 = tf.split(self.test_input, num_or_size_splits=2, axis=3)
    #     test_input = tf.concat([input_ch1, tf.zeros_like(input_ch1), input_ch2], axis = 3)
    #     summarys.append(tf.summary.image('test_input', test_input, max_outputs = 10))
    #     summarys.append(tf.summary.image('test_label', self.test_label, max_outputs = 10))
    #     summarys.append(tf.summary.image('test_out', self.test_out, max_outputs = 10))
    #     summary = tf.summary.merge(summarys)
    #     return summary
        

    def get_bce_loss(self, output_tensor, target_tensor, epsilon=1e-10):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= output_tensor, labels = target_tensor))
   #     return tf.reduce_mean(-target_tensor * tf.log(output_tensor + epsilon) -(1.0 - target_tensor) * tf.log(1.0 - output_tensor + epsilon))

    def get_log_softmax(self, x, y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= x, labels = y))

    def get_mse_loss(self, x, y):
        return tf.losses.mean_squared_error(predictions= x, labels= y)

    def get_l1_loss(self,x, y):
        return tf.losses.absolute_difference(x, y, scope='l1_loss')

    def save(self, step):
        print('---->saving', step)
        checkpoint_path = os.path.join(
            self.conf.modeldir, 'model')
        self.saver.save(self.sess, checkpoint_path, global_step=step)
    
    def save_summary(self, summary, step):
         print('---->summarizing', step)
         self.writer.add_summary(summary, step)

    def train(self):
        if self.conf.checkpoint >0:
            self.reload(self.conf.checkpoint)
        data = data_reader()
        iterations = 1
        max_epoch = int (self.conf.max_epoch - (self.conf.checkpoint)/ 500)

        for epoch in range(max_epoch):
            pbar = ProgressBar()
            for i in pbar(range(self.conf.updates_per_epoch)):
                inputs, labels, latent_r = data.next_batch(self.conf.batch_size)
             #   inputs_only_r = data.extract(inputs)
                sampled_zs = np.random.normal(size= (self.conf.batch_size,self.conf.hidden_size))
                feed_dict = {self.sampled_z_s: sampled_zs, self.input_y: labels, self.input_latent_r:latent_r, self.input_x: inputs}
                _ , d_loss = self.sess.run([self.train_disc,self.dis_loss], feed_dict= feed_dict)
                _ , g_loss, summary = self.sess.run([self.train_gen, self.gen_loss, self.train_summary], feed_dict = feed_dict)
                if iterations %self.conf.summary_step == 1:
                    self.save_summary(summary, iterations+self.conf.checkpoint)
                if iterations %self.conf.save_step == 0:
                    self.save(iterations+self.conf.checkpoint)
                iterations = iterations +1
           #     self.save_image(test_out, test_x, epoch)
            print("g_loss is ===================", g_loss, "d_loss is =================", d_loss)
            test_x, test_y, test_r = data.next_test_batch(self.conf.batch_size)
            test_out = self.sess.run([self.test_out], feed_dict= {self.test_r: test_r,  self.test_y: test_y})
            self.save_image(test_out, test_x, epoch)
    #           print("encd_s_loss is  ================", encd_s_loss, "decd_s_loss is =============", decd_s_loss)
     #       self.generate_con_image()
     #   self.evaluate(data)

    def save_image(self, imgs, inputs, epoch):
        imgs_test_folder = os.path.join(self.conf.working_directory, 'imgs_GAN')
        if not os.path.exists(imgs_test_folder):
            os.makedirs(imgs_test_folder)
        for k in range(self.conf.batch_size):
            temp_test_dir= os.path.join(imgs_test_folder, 'epoch_%d_#img_%d.png'%(epoch,k))
            res = np.zeros((self.conf.height, self.conf.height*5+8, 3))
            res[:,0:self.conf.height,:]= inputs[k,:,:,:]
            res[:,self.conf.height+2:self.conf.height*2+2,0]=inputs[k,:,:,0]
            res[:,self.conf.height+2:self.conf.height*2+2,2]=inputs[k,:,:,2]
            res[:,self.conf.height*2+4:self.conf.height*3+4, 1]= inputs[k,:,:,1]
            res[:,self.conf.height*3+6:self.conf.height*4+6, 1]= imgs[k,:,:,1]
            res[:,self.conf.height*4+8:self.conf.height*5+8, 0]= inputs[k,:,:,2]
            res[:,self.conf.height*4+8:self.conf.height*5+8, 2]= inputs[k,:,:,2]
            res[:,self.conf.height*4+8:self.conf.height*5+8, 1]= imgs[k,:,:,1]
            imsave(temp_test_dir, res)
        print("Evaluation images generated！==============================") 


    

    def generate_con_image(self):
        
        for i in range(self.conf.n_class):
            sampled_y = np.zeros((self.conf.batch_size, self.conf.n_class), dtype=np.float32)
            sampled_y[:,i]=1
            imgs = self.sess.run(self.generate_con_out, {self.generated_y: sampled_y})
            for k in range(imgs.shape[0]):
                imgs_folder = os.path.join(self.conf.working_directory, 'imgs_con_parallel', str(i))
                if not os.path.exists(imgs_folder):
                    os.makedirs(imgs_folder)   
                imsave(os.path.join(imgs_folder,'%d.png') % k,
                    imgs[k,:,:,:])
        print("conditional generated imgs saved!!!!==========================")               
    
    def evaluate(self, data):        
     #   data = data_reader()
        pbar = ProgressBar()
        imgs_original_folder = os.path.join(self.conf.working_directory, 'imgs_original_parallel')
        if not os.path.exists(imgs_original_folder):
            os.makedirs(imgs_original_folder)
        imgs_test_folder = os.path.join(self.conf.working_directory, 'imgs_test_parallel')
        if not os.path.exists(imgs_test_folder):
            os.makedirs(imgs_test_folder)
        for i in pbar(range(self.conf.max_test_epoch)):
            x, y, r = data.next_test_batch(self.conf.batch_size)
            x_extracted = data.extract(x)
     #       print("x==============================", x.shape)
     #       print("x_ex============================", x_extracted.shape)
            y_label  = np.argmax(y, axis= 1)
            for j in range (self.conf.max_generated_imgs):
                output_test, summary = self.sess.run([self.test_out, self.test_summary], feed_dict={self.test_input: x_extracted, self.test_r: r,  self.test_y: y, self.test_label: x})
                for k in range(output_test.shape[0]):                    
                    # res = np.ones([self.conf.height, self.conf.width*3 +4, 3])
                    # res[:,0:self.conf.width,:]= x[k,:,:,:]
                    # res[:,self.conf.width+2:self.conf.width*2+2,(0,2)] = x_extracted[k,:,:,:]
                    # res[:,self.conf.width+2:self.conf.width*2+2,1] = 0
                    # res[:,self.conf.width*2+4:, :] = output_test[k,:,:,:]
                    temp_test_dir = os.path.join(imgs_test_folder, 'epoch_%d_#img_%d'%(i,k))
                    if not os.path.exists(temp_test_dir):
                        os.makedirs(temp_test_dir)
                    imsave(os.path.join(imgs_original_folder,'epoch_%d_#img_%d_cls_%d.png') %(i,k,y_label[k]),
                        x[k,:,:,:])
                    imsave(os.path.join(temp_test_dir,'imgs_%d.png') %j,
                        output_test[k,:,:,:])
           #     self.save_summary(summary, i*10*50+k*50+j)
        print("Evaluation images generated！==============================")

    
    def generate_and_save(self):
        imgs = self.sess.run(self.generated_out)
        for k in range(imgs.shape[0]):
            imgs_folder = os.path.join(self.conf.working_directory, 'imgs_parallel')
            if not os.path.exists(imgs_folder):
                os.makedirs(imgs_folder)      
            res= np.zeros([imgs.shape[1],imgs.shape[2],3])         
            res[:,:,0]=imgs[k,:,:,0]
            res[:,:,1]= 0
            res[:,:,2]=imgs[k,:,:,1]                
            imsave(os.path.join(imgs_folder,'%d.png') % k,
                res)
            imsave(os.path.join(imgs_folder,'%d_ch0.png') % k,
                imgs[k,:,:,0]) 
            imsave(os.path.join(imgs_folder,'%d_ch1.png') % k,
                imgs[k,:,:,1])    
        print("generated imgs saved!!!!==========================")


    def test(self):
        return
        # model.reload(FLAGS.checkpoint)
        # samples = model.generate_samples()
        # sigmas = np.logspace(-1.0, 0.0, 10)
        # lls = []
        # for sigma in sigmas:
        #     print("sigma: ", sigma)
        #     nlls = []
        #     for i in range(1, 10 + 1):
        #         X = data.next_test_batch(FLAGS.batch_size)
        #         nll = parzen_cpu_batch(
        #             X,
        #             samples,
        #             sigma=sigma,
        #             batch_size=FLAGS.batch_size,
        #             num_of_samples=10000,
        #             data_size=12288)
        #         nlls.extend(nll)
        #     nlls = np.array(nlls).reshape(1000)  # 1000 valid images
        #     print("sigma: ", sigma)
        #     print("ll: %d" % (np.mean(nlls)))
        #     lls.append(np.mean(nlls))
        # sigma = sigmas[np.argmax(lls)]

        # nlls = []
        # data.reset()
        # for i in range(1, 100 + 1):  # number of test batches = 100
        #     X = data.next_test_batch(FLAGS.batch_size)
        #     nll = parzen_cpu_batch(
        #         X,
        #         samples,
        #         sigma=sigma,
        #         batch_size=FLAGS.batch_size,
        #         num_of_samples=10000,
        #         data_size=12288)
        #     nlls.extend(nll)
        # nlls = np.array(nlls).reshape(10000)  # 10000 test images
        # print("sigma: ", sigma)
        # print("ll: %d" % (np.mean(nlls)))
        # print("se: %d" % (nlls.std() / np.sqrt(10000)))

    def reload(self, epoch):
        checkpoint_path = os.path.join(
            self.conf.modeldir, 'model')
        model_path = checkpoint_path +'-'+str(epoch)
        if not os.path.exists(model_path+'.meta'):
            print('------- no such checkpoint', model_path)
            return       
        self.saver.restore(self.sess, model_path)
        print("model load successfully===================")

   # def evaluate
    def log_marginal_likelihood_estimate(self):
        x_mean = tf.reshape(self.input_tensor, [self.conf.batch_size, self.conf.width*self.conf.height])
        x_sample = tf.reshape(self.out_put, [self.conf.batch_size,self.conf.width*self.conf.height])
        x_sigma = tf.multiply(1.0, tf.ones(tf.shape(x_mean)))
        return log_likelihood_gaussian(x_mean, x_sample, x_sigma)+\
                log_likelihood_prior(self.latent_sample)-\
                log_likelihood_gaussian(self.latent_sample, self.mean, self.stddev)        

    def evaluate_nll(self, test_input):
        sample_ll= []
        for j in range (1000):
            res= self.sess.run(self.lle,{self.input_tensor: test_input})
            sample_ll.append(res)
        sample_ll = np.array(sample_ll)
        m = np.amax(sample_ll, axis=1, keepdims=True)
        log_marginal_estimate = m + np.log(np.mean(np.exp(sample_ll - m), axis=1, keepdims=True))
        return np.mean(log_marginal_estimate)

    def generate_samples(self):
        samples = []
        for i in range(100): # generate 100*100 samples
            samples.extend(self.sess.run(self.sample_out))
        samples = np.array(samples)
        print (samples.shape)
        return samples



        

