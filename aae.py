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
        #variables for the first autoencoder
        # self.var_enc_r = [var for var in variables if var.name.startswith('ENC_R')]
        # self.var_encd_r = [var for var in variables if var.name.startswith('ENCD_R')]
        # self.var_dec_r = [var for var in variables if var.name.startswith('DEC_R')]
        # self.var_decd_r = [var for var in variables if var.name.startswith('DECD_R')]
        #variables for the conditional autoencoder
        self.var_enc_r_s = [var for var in variables if var.name.startswith('ENC_R_S')]
        self.var_encd_r_s = [var for var in variables if var.name.startswith('ENCD_R_S')]
        self.var_dec_r_s = [var for var in variables if var.name.startswith('DEC_R_S')]
        self.var_decd_r_s = [var for var in variables if var.name.startswith('DECD_R_S')]
        self.var_dec_r_s_super = [var for var in variables if var.name.startswith('DEC_R_S_SUPER')]
        # opt for the first autoencoder
        # self.train_decd_r = tf.contrib.layers.optimize_loss(self.decdr_loss, tf.contrib.framework.get_or_create_global_step(), 
        #     learning_rate=self.conf.learning_rate, optimizer='Adam', variables=self.var_decd_r, update_ops=[])
        # self.train_encd_r = tf.contrib.layers.optimize_loss(self.encdr_loss, tf.contrib.framework.get_or_create_global_step(), 
        #     learning_rate=self.conf.learning_rate, optimizer='Adam', variables=self.var_encd_r, update_ops=[])
        # self.train_enc_r = tf.contrib.layers.optimize_loss(self.encr_loss, tf.contrib.framework.get_or_create_global_step(), 
        #     learning_rate=self.conf.learning_rate, optimizer='Adam', variables=self.var_enc_r, update_ops=[])
        # self.train_dec_r = tf.contrib.layers.optimize_loss(self.decr_loss, tf.contrib.framework.get_or_create_global_step(), 
        #     learning_rate=self.conf.learning_rate, optimizer='Adam', variables=self.var_dec_r, update_ops=[])
        # opt for the con autoencoder
        self.train_decd_r_s = tf.contrib.layers.optimize_loss(self.decdr_s_loss, tf.contrib.framework.get_or_create_global_step(), 
            learning_rate=self.conf.learning_rate, optimizer='Adam', variables=self.var_decd_r_s, update_ops=[])
        self.train_encd_r_s = tf.contrib.layers.optimize_loss(self.encdr_s_loss, tf.contrib.framework.get_or_create_global_step(), 
            learning_rate=self.conf.learning_rate, optimizer='Adam', variables=self.var_encd_r_s, update_ops=[])

        self.train_enc_r_s = tf.contrib.layers.optimize_loss(self.encr_s_loss, tf.contrib.framework.get_or_create_global_step(), 
            learning_rate=self.conf.learning_rate, optimizer='Adam', variables=self.var_enc_r_s, update_ops=[])
        self.train_dec_r_s = tf.contrib.layers.optimize_loss(self.decr_s_loss, tf.contrib.framework.get_or_create_global_step(), 
            learning_rate=self.conf.learning_rate, optimizer='Adam', variables=self.var_dec_r_s, update_ops=[])
        
        self.train_dec_r_s_super = tf.contrib.layers.optimize_loss(self.decr_super_loss, tf.contrib.framework.get_or_create_global_step(), 
            learning_rate=self.conf.learning_rate, optimizer='Adam', variables=self.var_dec_r_s_super, update_ops=[])
    
  #      self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)
        self.writer = tf.summary.FileWriter(self.conf.logdir, self.sess.graph)
    #    self.train_summary = self.config_summary()
        self.train_con_summary =self.config_con_summary()
        self.test_summary = self.config_test_summary()

    def build_network(self):

        #build the first autoencoder
    #    self.input_r = tf.placeholder(tf.float32,[None, self.conf.height, self.conf.width, 2])

     #   self.sampled_z_r = tf.placeholder(tf.float32,[None, self.conf.hidden_size])

        self.input_r_s = tf.placeholder(tf.float32,[None, self.conf.height, self.conf.width, 3])

        self.sampled_z_s = tf.placeholder(tf.float32,[None, self.conf.hidden_size])

        self.input_y = tf.placeholder(tf.int32,[None,self.conf.n_class])

        self.input_extracted = tf.placeholder(tf.float32,[None, self.conf.height, self.conf.width, 2])
    
        self.input_latent_r = tf.placeholder(tf.float32,[None, self.conf.hidden_size])
    #    self.sampled_x_r = tf.placeholder(tf.float32,[None, self.height, self.width, 2])
        # with tf.variable_scope('ENC_R') as scope:
        #     intermediate_out_r = encoder_all(self.input_r)
        #     self.latent_z_r = encoder_x_r(intermediate_out_r, self.conf.hidden_size)
        # with tf.variable_scope('DEC_R') as scope:
        #     self.out_x_r = decoder_all(self.latent_z_r, self.chan_out_r) #reconstructed
        #     self.out_x_r_super = super_resolution(self.out_x_r, self.chan_out_r)
        #     scope.reuse_variables()
        #     self.out_x_r_sampled = decoder_all(self.sampled_z_r, self.chan_out_r)
        #     print('=========================',self.out_x_r_sampled.get_shape())
        #     self.out_x_r_sampled_super =super_resolution(self.out_x_r_sampled, self.chan_out_r)

        # with tf.variable_scope('ENCD_R') as scope:
        #     self.d_enc_out_p = createAdversary(self.sampled_z_r) #postive samples # V_gen_encdr
        #     scope.reuse_variables()
        #     self.d_enc_out_n = createAdversary(self.latent_z_r)#negative samples # v_obs_encdr
        # with tf.variable_scope('DECD_R') as scope:
        #     output1 = createAdversary_Dec(self.input_r) # positive samples # v_obs_decdr # use the same one or draw another one?
        #     self.d_dec_out_p = Adv_dec_x_r(output1)
        #     scope.reuse_variables()
        #     print('==============================',self.out_x_r_sampled_super.get_shape())
        #     output2 = createAdversary_Dec(self.out_x_r_sampled_super) #negative samples # v_gen_decdr 
        #     self.d_dec_out_n = Adv_dec_x_r(output2)
        #     output3 = createAdversary_Dec(self.out_x_r_super) # sample for autoencoder loss
        #     self.d_dec_out_rec = Adv_dec_x_r(output3)
            
            
        # generated_latent = tf.random_normal([self.conf.batch_size,self.conf.hidden_size])
        # with tf.variable_scope('DEC_R', reuse= True) as scope:
        #     self.generated_out= decoder_all(generated_latent, self.chan_out_r)
        #     self.generated_out= super_resolution(self.generated_out, self.chan_out_r)

        # # the loss for the top autoencoder
        # self.decdr_loss = self.get_bce_loss(self.d_dec_out_p, tf.ones_like(self.d_dec_out_p)) + self.get_bce_loss(self.d_dec_out_n, tf.zeros_like(self.d_dec_out_n))
        # self.encdr_loss = self.get_bce_loss(self.d_enc_out_p, tf.ones_like(self.d_enc_out_p)) + self.get_bce_loss(self.d_enc_out_n,  tf.zeros_like(self.d_enc_out_n))
        # self.rec_loss = (self.get_bce_loss(self.out_x_r, self.input_r)+ get_ssim_loss(self.out_x_r_super, self.input_r))/2
        # self.encdr_loss_enc = self.get_bce_loss(self.d_enc_out_n, tf.ones_like(self.d_enc_out_n))
        # self.decdr_loss_dec = self.get_bce_loss(self.d_dec_out_n, tf.ones_like(self.d_dec_out_n)) + self.get_bce_loss(self.d_dec_out_rec, tf.ones_like(self.d_dec_out_rec))
        # self.encr_loss = self.rec_loss + self.conf.gamma_enc*self.encdr_loss_enc
        # self.decr_loss = self.rec_loss + self.conf.gamma_dec*self.decdr_loss_dec
        print("finish building the first autoencoder===========Now the conditional one")
        #build the conditional auto encoder
        with tf.variable_scope('ENC_R_S') as scope:
            intermediate_out_r_s = encoder_all(self.input_r_s)
            self.latent_y_raw, self.latent_s, self.latent_z_rs = encoder_x_r_s(intermediate_out_r_s, 
                self.conf.hidden_size, self.conf.n_class, self.conf.hidden_size)
            self.latent_y = tf.nn.log_softmax(self.latent_y_raw)

        with tf.variable_scope('DEC_R_S') as scope:
            dec_input = tf.concat([self.latent_z_rs, self.latent_y, self.latent_s], 1)
            self.out_x_r_s = decoder_all(dec_input, self.chan_out_r_s) #x_r_s_head, reconstrcuted image
     #       self.out_x_r_s_super = super_resolution(self.out_x_r_s, self.chan_out_r_s)
            scope.reuse_variables()
            dec_inpt_gen = tf.concat([self.latent_z_rs, self.latent_y, self.sampled_z_s],1)
            self.out_x_rs_gen = decoder_all(dec_inpt_gen, self.chan_out_r_s) # dec output using randomed z_s
     #       self.out_x_rs_gen_super= super_resolution(self.out_x_rs_gen, self.chan_out_r_s)
        with tf.variable_scope('DEC_R_S_SUPER') as scope:
            self.out_x_r_s_super = super_resolution(self.out_x_r_s, self.chan_out_r_s)

        with tf.variable_scope('ENCD_R_S') as scope:
            self.d_encds_out_p = createAdversary(self.sampled_z_s) #positive samples. V_gen_encds
            scope.reuse_variables()
            self.d_encds_out_n = createAdversary(self.latent_s)# negative smaples, v_obs_encds
        with tf.variable_scope('DECD_R_S') as scope:
            output1_con = createAdversary_Dec(self.input_r_s)
            self.d_decds_out_p = Adv_dec_x_r_s(output1_con, self.conf.n_class) # positive, Y_obs, original images
            scope.reuse_variables()
            output2_con = createAdversary_Dec(self.out_x_rs_gen)
            self.d_decds_out_n =Adv_dec_x_r_s(output2_con, self.conf.n_class) #negative, Y_gen
            output3_con = createAdversary_Dec(self.out_x_r_s)
            self.d_decds_out_recon = Adv_dec_x_r_s(output3_con, self.conf.n_class)
        
        # with tf.variable_scope('ENC_R', reuse= True) as scope:
        #     inter_out_con = encoder_all(self.input_extracted)
        #     self.z_r_con = encoder_x_r(inter_out_con, self.conf.hidden_size)
            
        generated_latent_s = tf.random_normal([self.conf.batch_size,self.conf.hidden_size])
        generated_latent_r = tf.random_normal([self.conf.batch_size,self.conf.hidden_size])
        self.generated_y = tf.placeholder(tf.float32,[None,self.conf.n_class])
        gen_input_con= tf.concat([generated_latent_r,self.generated_y, generated_latent_s],1)
        with tf.variable_scope('DEC_R_S', reuse= True) as scope:
            self.generate_con_out = decoder_all(gen_input_con, self.chan_out_r_s)
        
        with tf.variable_scope('DEC_R_S_SUPER', reuse= True) as scope:
            self.generate_con_out = super_resolution(self.generate_con_out, self.chan_out_r_s)
        

        # the loss for the conditional auto encoder
        self.encdr_s_loss = self.get_bce_loss(self.d_encds_out_p, tf.ones_like(self.d_encds_out_p))+ \
            self.get_bce_loss(self.d_encds_out_n, tf.zeros_like(self.d_encds_out_n))
        self.y_head = tf.concat([self.input_y, tf.zeros([self.conf.batch_size, 1], tf.int32)],1)
        self.y_gen = tf.concat([tf.zeros([self.conf.batch_size, self.conf.n_class], tf.int32), 
            tf.ones([self.conf.batch_size, 1], tf.int32)], 1)
        self.decdr_s_loss = self.get_log_softmax(self.d_decds_out_p, self.y_head)+ \
            self.get_log_softmax(self.d_decds_out_n, self.y_gen)
        self.rec_loss_rs = self.get_bce_loss(self.out_x_r_s, self.input_r_s)
        self.y_loss = self.get_log_softmax(self.latent_y_raw, self.input_y)
        self.decr_super_loss = self.get_l1_loss(self.input_r_s, self.out_x_r_s_super) 

        self.zr_loss = self.get_mse_loss(self.latent_z_rs, self.input_latent_r)  #MSE error

        self.encds_loss_enc = self.get_bce_loss(self.d_encds_out_n, tf.ones_like(self.d_encds_out_n))
        self.decdrs_loss_dec = self.get_log_softmax(self.d_decds_out_n, self.y_head) + \
            self.get_log_softmax(self.d_decds_out_recon, self.y_head)        
        self.encr_s_loss = self.rec_loss_rs+ self.zr_loss+ self.y_loss+ self.conf.gamma_enc*self.encds_loss_enc
        self.decr_s_loss = self.rec_loss_rs + self.conf.gamma_dec* self.decdrs_loss_dec

        # build the model for the final conditional generation
        
        self.test_input = tf.placeholder(tf.float32,[None, self.conf.height, self.conf.width, 2])
        self.test_label = tf.placeholder(tf.float32,[None, self.conf.height, self.conf.width, 3]) # desired results
        self.test_y = tf.placeholder(tf.int32,[None,self.conf.n_class])
        self.test_r = tf.placeholder(tf.float32,[None, self.conf.hidden_size])

        randomed_s_test = tf.random_normal([self.conf.batch_size,self.conf.hidden_size])
        self.test_y= tf.cast(self.test_y, tf.float32)
        gen_input_test= tf.concat([self.test_r,self.test_y, randomed_s_test],1)

        with tf.variable_scope('DEC_R_S', reuse= True) as scope:
            self.test_out = decoder_all(gen_input_test, self.chan_out_r_s)

        with tf.variable_scope('DEC_R_S_SUPER', reuse= True) as scope:
            self.test_out = super_resolution(self.test_out, self.chan_out_r_s)

        
       

    def config_summary(self):
        summarys = []                      
        summarys.append(tf.summary.scalar('/Rec_loss', self.rec_loss))
        summarys.append(tf.summary.scalar('/encoder_loss', self.encdr_loss_enc))
        summarys.append(tf.summary.scalar('/decoder_loss', self.decdr_loss_dec))
        summarys.append(tf.summary.scalar('/enc_adv_loss', self.encdr_loss))
        summarys.append(tf.summary.scalar('/dec_adv_loss', self.decdr_loss)) 
        input_ch1, input_ch2 = tf.split(self.input_r, num_or_size_splits=2, axis=3)
        summarys.append(tf.summary.image('input_channel_one', input_ch1, max_outputs = 10))
        summarys.append(tf.summary.image('input_channel_two', input_ch2, max_outputs = 10))
        out_ch1, out_ch2 = tf.split(self.out_x_r_super, num_or_size_splits=2, axis =3)
        summarys.append(tf.summary.image('output_channel_one', out_ch1, max_outputs = 10))
        summarys.append(tf.summary.image('output_channel_two', out_ch2, max_outputs = 10))
        inter_ch1, inter_ch2 = tf.split(self.out_x_r, num_or_size_splits=2, axis =3)
        summarys.append(tf.summary.image('inter_channel_one', inter_ch1, max_outputs = 10))
        summarys.append(tf.summary.image('inter_channel_two', inter_ch2, max_outputs = 10))        
        summary = tf.summary.merge(summarys)
        return summary


    def config_con_summary(self):
        summarys = []    

        summarys.append(tf.summary.scalar('/Rec_loss_con', self.rec_loss_rs))
        summarys.append(tf.summary.scalar('/encoder_loss_con', self.encds_loss_enc)) 
        summarys.append(tf.summary.scalar('/r_loss', self.zr_loss)) 
        summarys.append(tf.summary.scalar('/y_loss', self.y_loss)) 
        summarys.append(tf.summary.scalar('/decoder_loss_con', self.decdrs_loss_dec))
        summarys.append(tf.summary.scalar('/enc_adv_loss_con', self.encdr_s_loss))
        summarys.append(tf.summary.scalar('/dec_adv_loss_con', self.decdr_s_loss))
        summarys.append(tf.summary.scalar('/l1_loss', self.decr_super_loss))
        summarys.append(tf.summary.image('input_con', self.input_r_s, max_outputs = 10))
        summarys.append(tf.summary.image('inter_con', self.out_x_r_s, max_outputs = 10))
        summarys.append(tf.summary.image('output_con', self.out_x_r_s_super, max_outputs = 10))
        summary = tf.summary.merge(summarys)

        return summary

    def config_test_summary(self):
        summarys= []
        input_ch1, input_ch2 = tf.split(self.test_input, num_or_size_splits=2, axis=3)
        test_input = tf.concat([input_ch1, tf.zeros_like(input_ch1), input_ch2], axis = 3)
        summarys.append(tf.summary.image('test_input', test_input, max_outputs = 10))
        summarys.append(tf.summary.image('test_label', self.test_label, max_outputs = 10))
        summarys.append(tf.summary.image('test_out', self.test_out, max_outputs = 10))
        summary = tf.summary.merge(summarys)
        return summary
        

    def get_bce_loss(self, output_tensor, target_tensor, epsilon=1e-10):
   #     return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= x, labels = y))
        return tf.reduce_mean(-target_tensor * tf.log(output_tensor + epsilon) -(1.0 - target_tensor) * tf.log(1.0 - output_tensor + epsilon))

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
    #    epoch = 0
        # max_epoch = int (max(self.conf.max_epoch - self.conf.checkpoint/1000, 0))

        # print("The epochs  for the first model to be trained is ", max_epoch)

        # for epoch in range(max_epoch):
        # #    pbar = ProgressBar()
        #     for i in range(self.conf.updates_per_epoch):            
        #         inputs, labels= data.next_batch(self.conf.batch_size)
        #         inputs_only_r = data.extract(inputs)
        #     #    sampled_zr = tf.random_normal([self.conf.batch_size,self.conf.hidden_size])
        #         sampled_zr = np.random.normal(size= (self.conf.batch_size,self.conf.hidden_size))
        #         feed_dict= {self.input_r:inputs_only_r, self.sampled_z_r: sampled_zr}
        #         _, decd_r_loss, = self.sess.run([self.train_decd_r, self.decdr_loss], feed_dict= feed_dict)
        #         _, encd_r_loss, = self.sess.run([self.train_encd_r, self.encdr_loss], feed_dict= feed_dict)
        #         _, enc_r_loss = self.sess.run([self.train_enc_r, self.encr_loss], feed_dict =  feed_dict)
        #         loss, _, dec_r_loss, summary = self.sess.run([self.decdr_loss_dec, self.train_dec_r, self.decr_loss, self.train_summary], feed_dict = feed_dict)
        #         if iterations %self.conf.summary_step == 1:
        #             self.save_summary(summary, iterations+self.conf.checkpoint)
        #             print("summary_saved")
        #         if iterations %self.conf.save_step == 0:
        #             self.save(iterations+self.conf.checkpoint)
        #         iterations = iterations + 1
        #     print("enc_r_loss is  ================", enc_r_loss)
        #     print("dec_r_loss is =====================",dec_r_loss)            
        #     self.generate_and_save()
        # print("the first model is well trained, now the second one !================")
        # max_con_epoch = int (self.conf.max_con_epoch - (self.conf.checkpoint- 75000)/ 1000)
        # print("The epochs  for the first model to be trained is ", max_con_epoch)
        max_con_epoch = int (self.conf.max_con_epoch - (self.conf.checkpoint)/ 500)

        for epoch in range(max_con_epoch):
            pbar = ProgressBar()
            for i in pbar(range(self.conf.updates_per_epoch)):
                inputs, labels, latent_r = data.next_batch(self.conf.batch_size)
             #   inputs_only_r = data.extract(inputs)
                sampled_zs = np.random.normal(size= (self.conf.batch_size,self.conf.hidden_size))
                feed_dict_1 = {self.input_r_s: inputs, self.input_y: labels, self.sampled_z_s:sampled_zs}
                feed_dict_2 = {self.input_r_s: inputs, self.input_latent_r:latent_r, self.input_y: labels, self.sampled_z_s:sampled_zs}
                _ , encd_s_loss = self.sess.run([self.train_encd_r_s,self.encdr_s_loss], feed_dict= feed_dict_1)
                _ , decd_s_loss = self.sess.run([self.train_decd_r_s, self.decdr_s_loss], feed_dict = feed_dict_1)
                _ , enc_s_loss = self.sess.run([self.train_enc_r_s, self.encr_s_loss], feed_dict= feed_dict_2)
                _ , decr_super_loss = self.sess.run([self.train_dec_r_s_super, self.decr_super_loss], feed_dict =feed_dict_2)
           #     _ = self.sess.run(self.decr_s_loss,feed_dict = feed_dict_2)
                _ , dec_s_loss, summary_con = self.sess.run([self.train_dec_r_s, self.decr_s_loss, self.train_con_summary],feed_dict = feed_dict_2)
         #       _ , dec_s_loss = self.sess.run([self.train_dec_r_s, self.decr_s_loss],feed_dict = feed_dict_2)
                if iterations %self.conf.summary_step == 1:
                    self.save_summary(summary_con, iterations+self.conf.checkpoint)
                if iterations %self.conf.save_step == 0:
                    self.save(iterations+self.conf.checkpoint)
                iterations = iterations +1
       #         print("encd_s_loss is  ================", encd_s_loss, "decd_s_loss is =============", decd_s_loss)
            self.generate_con_image()
        self.evaluate(data)


    

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
     #       x_extracted = data.extract(x)
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



        

