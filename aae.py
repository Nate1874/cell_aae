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


class AAE(object):

    def __init__(self, sess, flag):
        self.conf = flag
        self.sess = sess
        self.chan_out_r = 2
        self.chan_out_r_s =3
        self.input_sr = tf.placeholder(tf.float32,[None, self.conf.height, self.conf.width, 3])
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
        self.var_enc_r = [var for var in variables if var.name.startswith('ENC_R')]
        self.var_encd_r = [var for var in variables if var.name.startswith('ENCD_R')]
        self.var_dec_r = [var for var in variables if var.name.startswith('DEC_R')]
        self.var_decd_r = [var for var in variables if var.name.startswith('DECD_R')]
        # TO DO : add variables for bottom part
        self.train_decd_r = tf.contrib.layers.optimize_loss(self.decdr_loss, tf.contrib.framework.get_or_create_global_step(), 
            learning_rate=self.conf.learning_rate, optimizer='Adam', variables=self.var_decd_r, update_ops=[])
        self.train_encd_r = tf.contrib.layers.optimize_loss(self.encdr_loss, tf.contrib.framework.get_or_create_global_step(), 
            learning_rate=self.conf.learning_rate, optimizer='Adam', variables=self.var_encd_r, update_ops=[])
        self.train_enc_r = tf.contrib.layers.optimize_loss(self.encr_loss, tf.contrib.framework.get_or_create_global_step(), 
            learning_rate=self.conf.learning_rate, optimizer='Adam', variables=self.var_enc_r, update_ops=[])
        self.train_dec_r = tf.contrib.layers.optimize_loss(self.decr_loss, tf.contrib.framework.get_or_create_global_step(), 
            learning_rate=self.conf.learning_rate, optimizer='Adam', variables=self.var_dec_r, update_ops=[])
  #      self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)
        self.writer = tf.summary.FileWriter(self.conf.logdir, self.sess.graph)
        self.train_summary = self.config_summary()

    def build_network(self):
        self.input_r = tf.placeholder(tf.float32,[None, self.conf.height, self.conf.width, 2])
        self.sampled_z_r = tf.placeholder(tf.float32,[None, self.conf.hidden_size])
    #    self.sampled_x_r = tf.placeholder(tf.float32,[None, self.height, self.width, 2])
        with tf.variable_scope('ENC_R') as scope:
            intermediate_out_r = encoder_all(self.input_r)
            self.latent_z_r = encoder_x_r(intermediate_out_r, self.conf.hidden_size)
        with tf.variable_scope('DEC_R') as scope:
            print(self.latent_z_r.get_shape())
            self.out_x_r = decoder_all(self.latent_z_r, self.chan_out_r)
            print(self.out_x_r.get_shape())
            scope.reuse_variables()
            print(self.sampled_z_r.get_shape())
            self.out_x_r_sampled = decoder_all(self.sampled_z_r, self.chan_out_r)
            print(self.out_x_r_sampled.get_shape())
        with tf.variable_scope('ENCD_R') as scope:
            self.d_enc_out_p = createAdversary(self.sampled_z_r) #postive samples # V_gen_encdr
            scope.reuse_variables()
            self.d_enc_out_n = createAdversary(self.latent_z_r)#negative samples # v_obs_encdr
        with tf.variable_scope('DECD_R') as scope:
            output1 = createAdversary_Dec(self.input_r) # positive samples # v_obs_decdr
    #        print(self.input_r.get_shape())
            self.d_dec_out_p = Adv_dec_x_r(output1)
    #        print(output1.get_shape())
            scope.reuse_variables()
            output2 = createAdversary_Dec(self.out_x_r_sampled) #negative samples # v_gen_decdr
    #        print(self.out_x_r_sampled.get_shape())
            self.d_dec_out_n = Adv_dec_x_r(output2)

            output3 = createAdversary_Dec(self.out_x_r) # sample for autoencoder loss
            self.d_dec_out_rec = Adv_dec_x_r(output3)
            
            
        generated_latent = tf.random_normal([self.conf.batch_size,self.conf.hidden_size])
        with tf.variable_scope('DEC_R', reuse= True) as scope:
            self.generated_out= decoder_all(generated_latent, self.chan_out_r)


        self.decdr_loss = self.get_bce_loss(self.d_dec_out_p, tf.ones_like(self.d_dec_out_p)) + self.get_bce_loss(self.d_dec_out_n, tf.zeros_like(self.d_dec_out_n))
        self.encdr_loss = self.get_bce_loss(self.d_enc_out_p, tf.ones_like(self.d_enc_out_n)) + self.get_bce_loss(self.d_enc_out_n,  tf.zeros_like(self.d_enc_out_p))
        self.rec_loss = self.get_bce_loss(self.out_x_r, self.input_r)
        self.encdr_loss_enc = self.get_bce_loss(self.d_enc_out_n, tf.ones_like(self.d_enc_out_n))
        self.decdr_loss_dec = self.get_bce_loss(self.d_dec_out_n, tf.ones_like(self.d_dec_out_n)) + self.get_bce_loss(self.d_dec_out_rec, tf.ones_like(self.d_dec_out_rec))

        self.encr_loss = self.rec_loss + self.conf.gamma_enc*self.encdr_loss_enc
        self.decr_loss = self.rec_loss + self.conf.gamma_dec*self.decdr_loss_dec

    def config_summary(self):
        summarys = []                      
        summarys.append(tf.summary.scalar('/Rec_loss', self.rec_loss))
        summarys.append(tf.summary.scalar('/encoder_loss', self.encr_loss))
        summarys.append(tf.summary.scalar('/decoder_loss', self.decr_loss))
        summarys.append(tf.summary.scalar('/enc_adv_loss', self.encdr_loss))
        summarys.append(tf.summary.scalar('/dec_adv_loss', self.decdr_loss))
        input_ch1, input_ch2 = tf.split(self.input_r, num_or_size_splits=2, axis=3)
        summarys.append(tf.summary.image('input_channel1', input_ch1, max_outputs = 10))
        summarys.append(tf.summary.image('input_channel2', input_ch2, max_outputs = 10))
        out_ch1, out_ch2 = tf.split(self.out_x_r, num_or_size_splits=2, axis =3)
        summarys.append(tf.summary.image('output_channel1', out_ch1, max_outputs = 10))
        summarys.append(tf.summary.image('output_channel2', out_ch2, max_outputs = 10))
        summary = tf.summary.merge(summarys)
        return summary

    def get_bce_loss(self, x, y):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= x, labels = y))

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
            self.load(self.conf.checkpoint)
        data = data_reader()
        iterations = 1
    #    epoch = 0
        for epoch in range(self.conf.max_epoch):
            pbar = ProgressBar()
            for i in pbar(range(self.conf.updates_per_epoch)):            
                inputs, labels= data.next_batch(self.conf.batch_size)
                inputs_only_r = data.extract(inputs)
            #    sampled_zr = tf.random_normal([self.conf.batch_size,self.conf.hidden_size])
                sampled_zr = np.random.normal(size= (self.conf.batch_size,self.conf.hidden_size))
                feed_dict= {self.input_r:inputs_only_r, self.sampled_z_r: sampled_zr}
                _, decd_r_loss, = self.sess.run([self.train_decd_r, self.decdr_loss], feed_dict= feed_dict)
                _, encd_r_loss, = self.sess.run([self.train_encd_r, self.encdr_loss], feed_dict= feed_dict) 
                _, enc_r_loss = self.sess.run([self.train_enc_r, self.encr_loss], feed_dict =  feed_dict)
                _, dec_r_loss, summary = self.sess.run([self.train_dec_r, self.decr_loss, self.train_summary], feed_dict = feed_dict)
                if iterations %self.conf.summary_step == 1:
                    self.save_summary(summary, iterations+self.conf.checkpoint)
                if iterations %self.conf.save_step == 0:
                    self.save(iterations+self.conf.checkpoint)
                iterations = iterations + 1
            print("decd_r_loss is =============", decd_r_loss)
            print("encd_r_loss is  ==============", encd_r_loss)
            print("enc_r_loss is  ================", enc_r_loss)
            print("dec_r_loss is =====================",dec_r_loss)            
            self.generate_and_save()
    
    def generate_and_save(self):
            imgs = self.sess.run(self.generated_out)
            for k in range(imgs.shape[0]):
                imgs_folder = os.path.join(self.conf.working_directory, 'imgs')
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


    def update_params(self, input_tensor, step):
        loss, summary, kl_loss, rec_loss =  self.sess.run([self.train, self.train_summary, self.kl_loss, self.rec_loss], {self.input_tensor: input_tensor})
        self.writer.add_summary(summary, step)
        return loss, kl_loss, rec_loss

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

    def evaluate(self, test_input):
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



        

