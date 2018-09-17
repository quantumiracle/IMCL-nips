import tensorflow as tf
import tensorlayer as tl
import numpy as np
import opensim as osim
import cPickle as pickle


import matplotlib.pyplot as plt
plt.switch_backend('agg')

import threading as th
import multiprocessing as mp

from rl.agents import tf_util
from rl.agents import kfac
from rl.models import Actor, Critic
from rl.processor2 import transform_observation
from rl.noise import NormalNoise, GaussianWhiteNoiseProcess
from copy import copy
from math import *

import time
import random
import traceback

class DDPGAgent_t2p_kfac(object):
    def __init__(self, memory, noise, observation_dims, action_dims, batch_size, 
        nb_max_episodes, gamma=.99):
        # Parameters
        self.observation_dims = observation_dims
        self.action_dims = action_dims
        self.batch_size = batch_size
        self.nb_max_episodes = nb_max_episodes
        self.gamma = gamma

        self.train_multiplier = 2
        self.action_multiplier = 0.5
        self.action_bias = 0.5
        self.tau = 1e-3
        self.actor_lr = 3e-4
        self.critic_lr = 3e-4
        self.epi_horizon = 500
        self.epi_num_pool = []
        self.epi_reward_pool= []
        self.epirew_lower_bound = 0
        self.epirew_higher_bound = 0
        self.learn_counter = 0

        # Related objects
        self.memory = memory
        self.noise = noise
        self.lock = th.Lock()
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        self.sess = tf.Session(config=config)
        
        # State
        self.training = False

        # Tensor
        with tf.name_scope('S'):
            self.S = tf.placeholder(tf.float32, shape=[None, self.observation_dims], name='s')
        with tf.name_scope('A'):
            self.A = tf.placeholder(tf.float32, shape=[None, self.action_dims], name='a')
        with tf.name_scope('R'):
            self.R = tf.placeholder(tf.float32, shape=[None, 1], name='r')
        with tf.name_scope('S_'): 
            self.S_ = tf.placeholder(tf.float32, shape=[None, self.observation_dims], name='s_')
        
        # Networks
        with tf.variable_scope('Actor'):
            self.a = self._build_actor_net(self.S, scope_name='eval_net', reuse=False,)     #used in forward
            self.a_ = self._build_actor_net(self.S_, scope_name='target_net', reuse=False)    #used in BP
        with tf.variable_scope('Critic'):
            self.q = self._build_critic_net(self.S, self.a, scope_name='eval_net', reuse=False)
            self.q_predict = self._build_critic_net(self.S, self.A, scope_name='eval_net', reuse=True)  #used in forward
            self.q_ = self._build_critic_net(self.S_, self.a_, scope_name='target_net', reuse=False)  #used in BP
        
        # Network Parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')
        
        self.generate_train_op()

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        # Async the target nets
        self.sess.run([[tf.assign(ct, ce), tf.assign(at, ae)] 
            for ct, ce, at, ae in zip(self.ct_params, self.ce_params, self.at_params, self.ae_params)])


    def generate_train_op(self):
        stddv = 0.5
        #actor_optimizer
        self.actor_loss = - tf.reduce_mean(self.q)   #critic use batch sample from memory to predict  a list of q value
        actor_grads = tf.gradients(self.actor_loss, self.ae_params)
        '''
        #adam
        grads = list(zip(actor_grads, self.ae_params))
        trainer = tf.train.AdamOptimizer(learning_rate=self.actor_lr, beta1=0.9, beta2=0.999, epsilon=1e-08)
        self.atrain = trainer.apply_gradients(grads)
        self.actor_q_runner = None


        '''
        #kfac
        sampled_ac_na = tf.random_normal(tf.shape(self.a)) * stddv + self.a
        logprobsampled_n = - 0.5 * tf_util.mean(   #U.mean:just mean
                tf.square(self.a - tf.stop_gradient(sampled_ac_na)) / (tf.square(stddv)),  
                axis=1)  # Logprob of sampled action
        self.ac_fisher = - tf_util.mean(logprobsampled_n)  # Sampled loss of the policy
        with tf.device('/cpu:0'):  ##need set
                self.actor_optim = actor_optim = kfac.KfacOptimizer(learning_rate=self.actor_lr, clip_kl=0.001, \
                                                        momentum=0.9, kfac_update=1, epsilon=0.01, \
                                                        stats_decay=0.99, async=1, cold_iter=10,
                                                        max_grad_norm=10)
                update_stats_op = actor_optim.compute_and_apply_stats(self.ac_fisher, var_list=self.ae_params)  #update step!
                self.atrain, self.actor_q_runner = actor_optim.apply_gradients(list(zip(actor_grads, self.ae_params)))
        
        
        #Critic optimizer
        '''
        #adam
        self.q_target = self.R + self.gamma * self.q_
        self.critic_loss = tf.losses.mean_squared_error(labels=self.q_target, predictions=self.q_predict)
        self.ctrain = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss, 
            var_list=self.ce_params)

        '''
        #kfac
        self.q_target = self.R + self.gamma * self.q_
        self.critic_loss = tf.losses.mean_squared_error(labels=self.q_target, predictions=self.q_predict)
        
        #l2 loss regularization
        '''
        critic_reg = tc.layers.apply_regularization(   #??l2 regularization
                tc.layers.l2_regularizer(self.critic_l2_reg),   #need parameter critic_l2_reg to scale
                weights_list=critic_reg_vars
            )
            '''
         #Fisher loss
        sample_net = self.q_predict + tf.random_normal(tf.shape(self.q_predict))
        self.vf_fisher = - tf.reduce_mean(
            tf.pow(self.q_predict - tf.stop_gradient(sample_net), 2))
        with tf.device('/cpu:0'):  ##need set
            self.critic_optim = critic_optim = kfac.KfacOptimizer(learning_rate=self.critic_lr, clip_kl=0.001, \
                                                    momentum=0.9, kfac_update=1, epsilon=0.01, \
                                                    stats_decay=0.99, async=1, cold_iter=10,
                                                    max_grad_norm=10)
            self.critic_grads = critic_grads = tf.gradients(self.critic_loss, self.ce_params)
            update_stats_op = critic_optim.compute_and_apply_stats(self.vf_fisher, var_list=self.ce_params)
            self.ctrain, self.critic_q_runner = critic_optim.apply_gradients(list(zip(critic_grads, self.ce_params)))
        
        
        # 3.update the target net
        self.update_critic_target = [tf.assign(t, (1-self.tau)*t+self.tau*e) 
            for t, e in zip(self.ct_params, self.ce_params)]
        self.update_actor_target = [tf.assign(t, (1-self.tau)*t+self.tau*e) 
            for t, e in zip(self.at_params, self.ae_params)]
        


    '''
    #origin adam for actor&critic
    def generate_train_op(self):   #where the computation nodes are defined
        # 1.update the critic
        self.q_target = self.R + self.gamma * self.q_
        self.critic_loss = tf.losses.mean_squared_error(labels=self.q_target, predictions=self.q_predict)
        self.ctrain = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss, 
            var_list=self.ce_params) #actually all these nodes are computation functions and values after computing is retrun back to the parameters so as to update

        # 2.update the actor
        self.actor_loss = - tf.reduce_mean(self.q)   #critic use batch sample from memory to predict  a list of q value
        self.atrain = tf.train.AdamOptimizer(self.actor_lr).minimize(self.actor_loss, 
            var_list=self.ae_params)

    '''
    def _build_actor_net(self, actor_input, scope_name, reuse=False):
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            tl.layers.set_name_reuse(reuse)

            actor = tl.layers.InputLayer(actor_input, name='actor_input_layer')
            actor = tl.layers.DenseLayer(actor, n_units=96, name='dense_layer1', 
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            actor = tl.layers.ReshapeLayer(actor, [-1, 96, 1], name='reshape_layer')     
            actor = tl.layers.Conv1dLayer(actor, shape=[5, 1, 8], name='conv_layer1', padding='VALID',
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            actor = tl.layers.Conv1dLayer(actor, shape=[3, 8, 4], name='conv_layer2', padding='VALID',
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            actor = tl.layers.Conv1dLayer(actor, shape=[3, 4, 2], name='conv_layer3', padding='VALID',
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            actor = tl.layers.FlattenLayer(actor,name ='flatten_layer')
            actor = tl.layers.DenseLayer(actor, n_units=128, name='dense_layer2', 
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            actor = tl.layers.DenseLayer(actor, n_units=48, name='dense_layer3', 
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            actor = tl.layers.DenseLayer(actor, n_units=self.action_dims, act=tf.nn.tanh, name='dense_layer4')
            actor = tl.layers.LambdaLayer(actor, lambda a: a*self.action_multiplier + self.action_bias, name='lambda_layer')
            actor_output = actor.outputs
            print 'action shape', actor_output.get_shape()
            # actor.print_layers()
            # actor.print_params()
        return actor_output

    def _build_critic_net(self, critic_obs_input, critic_act_input, scope_name, reuse=False):
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            tl.layers.set_name_reuse(reuse)

            critic1 = tl.layers.InputLayer(critic_obs_input, name='obs_input_layer')
            critic2 = tl.layers.InputLayer(critic_act_input, name='act_input_layer')
            critic = tl.layers.ConcatLayer(layer=[critic1, critic2], concat_dim=1, name='concat_input_layer')
            critic = tl.layers.DenseLayer(critic, n_units=128, name='dense_layer1', 
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            critic = tl.layers.ReshapeLayer(critic, [-1, 128, 1], name ='reshape_layer')
            critic = tl.layers.Conv1dLayer(critic, shape=[5, 1, 8], name='conv_layer1', padding='VALID',
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            critic = tl.layers.Conv1dLayer(critic, shape=[3, 8, 4], name='conv_layer2', padding='VALID',
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            critic = tl.layers.Conv1dLayer(critic, shape=[3, 4, 2], name='conv_layer3', padding='VALID',
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            critic = tl.layers.FlattenLayer(critic,name ='flatten_layer')
            critic = tl.layers.DenseLayer(critic, n_units=128, name='dense_layer2', 
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            critic = tl.layers.DenseLayer(critic, n_units=96, name='dense_layer3',
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            critic = tl.layers.DenseLayer(critic, n_units=1, name='dense_layer4' )
            critic_output = critic.outputs
            print 'q-value shape', critic_output.get_shape()
            # critic.print_layers()
            # critic.print_params()
        return critic_output

    def learn(self, s, a, r, s_):  #core of training, [..]is a fetch, the self._ variables in [] are defined elsewhere as computation nodes in a graph, 
    #their values could be computed and return back to them
    #feed_dict is some variables needed in the process
        results = self.sess.run([self.critic_loss, self.actor_loss, self.ctrain, self.atrain,
                                self.update_critic_target, self.update_actor_target],
                                feed_dict={self.S:s, self.A:a, self.R:r, self.S_:s_})
        # self.learn_counter += 1
        # if self.learn_counter%10 == 0:
        #     print(' '*30, 'closs: {:.5f} aloss: {:.5f}'.format(
        #         results[0]*1000,results[1]*1000))

    def choose_action(self, s):
        return self.sess.run(self.a, feed_dict={self.S:s})
    
    def memory_append(self, epi_exps):
        for i in xrange(len(epi_exps)):
            # if np.random.uniform()>0.5:
            self.memory.append(epi_exps[i][0], epi_exps[i][1], epi_exps[i][2], 
                            epi_exps[i][3], epi_exps[i][4], training=self.training)
   
   

   #plot section
    def smooth_reward_curve(self, x, y):
        halfwidth = min(31, int(np.ceil(len(x)/30))) # Halfwidth of our smoothing convolution
        k = halfwidth
        xsmoo = x[k:-k]
        ysmoo = np.convolve(y, np.ones(2*k+1), mode='valid') / np.convolve(np.ones_like(y), np.ones(2*k+1), mode='valid')  #valid output the real overlap are of the convolution rusult
        downsample = max(int(np.floor(len(xsmoo)/1e3)),1)  #bigger or equal than 1
        return xsmoo[::downsample], ysmoo[::downsample]

    def fix_point(self, x, y, interval):
        np.insert(x, 0, 0)
        np.insert(y, 0, 0)

        fx, fy = [], []
        pointer = 0
        ninterval = int(max(x) / interval + 1)

        for i in range(ninterval):
            tmpx = interval * i

            while pointer + 1 < len(x) and tmpx > x[pointer + 1]:
                pointer += 1

            if pointer + 1 < len(x):
                alpha = (y[pointer+1] - y[pointer]) / (x[pointer+1] - x[pointer])
                tmpy = y[pointer] + alpha * (tmpx - x[pointer])
                fx.append(tmpx)
                fy.append(tmpy)

        return fx, fy
    
    def process_data(self, x, y, bin_size):
        x, y = self.smooth_reward_curve(x, y)
        x, y = self.fix_point(x, y, bin_size)
        return [x, y]


    def plot_fig(self, filename):
        self.lock.acquire()
        save_path = filename + '.png'

        #plot of Wu yuhuai
        plt.figure(figsize=(13,10))
        nb_epi = range(1, len(self.epi_num_pool)+1)
        prac_epi_reward = [i/10 for i in self.epi_reward_pool ]
        tmpx, tmpy = [], []
        datas = []
        tx, ty = self.process_data(nb_epi, prac_epi_reward, bin_size=2)  #set!!  bin_size is average window size
        tmpx.append(tx)
        tmpy.append(ty)


        if len(tmpx) > 1:  #more than one trainings data set to generate average plot with std
            length = min([len(t) for t in tmpx])
            for j in range(len(tmpx)):
                tmpx[j] = tmpx[j][:length]
                tmpy[j] = tmpy[j][:length]
            x = np.mean(np.array(tmpx), axis=0)
            y_mean = np.mean(np.array(tmpy), axis=0)
            y_std = np.std(np.array(tmpy), dtype=np.float64, axis=0) 
            #axis =0: mean of row, std of row, but for single training only has 1 line, so row std=0
            # row std is what we want, std of different trainings at every x point

        else:

            x = np.array(tmpx).reshape(-1)
            y_mean = np.array(tmpy).reshape(-1)
            y_std = np.zeros(len(y_mean))
        
        color_defaults = [
            '#1f77b4',  # muted blue
            '#ff7f0e',  # safety orange
            '#2ca02c',  # cooked asparagus green
            '#d62728',  # brick red
            '#9467bd',  # muted purple
            '#8c564b',  # chestnut brown
            '#e377c2',  # raspberry yogurt pink
            '#7f7f7f',  # middle gray
            '#bcbd22',  # curry yellow-green
            '#17becf'  # blue-teal
        ]

        lines = []    
        color = color_defaults[1] 

        ################differnet from original wu. plot, no for-loop here because just one agent   
        y_upper = y_mean + y_std
        y_lower = y_mean - y_std
        print(y_mean, y_std )
        plt.fill_between(
            x, list(y_lower), list(y_upper), interpolate=True, facecolor=color, linewidth=0.0, alpha=0.3
        )
        line = plt.plot(x, list(y_mean), color=color)
        lines.append(line[0])            

        #plt.xticks([20e6, 40e6,80e6,120e6,160e6, 200e6], ["20M", "40M","80M","120M","160M", "200M"])
        plt.xlabel('Samples')
        plt.ylabel('Rewards')

        '''#original plot
        plt.plot(nb_epi, prac_epi_reward, color='orange')
        plt.xlabel('Samples') #episode
        plt.ylabel('Rewards') #episode
        '''
        plt.savefig(save_path)
        self.lock.release()




    def forward(self, obs):
        state = self.memory.get_recent_state(obs)
        action = self.choose_action(state)
        action = np.reshape(action, (self.action_dims,))
        assert action.shape == (self.action_dims,)

        # if self.training and self.noise is not None:
        #     noise_sample = self.noise.sample()
        #     action += noise_sample
        #     # action = np.clip(action, a_max=1., a_min=0.)

        return action 

    def backward(self, epi_selecting):
        if self.memory.nb_entries > self.batch_size*128:
            for _ in xrange(self.train_multiplier):
                # choose which experiences should be learned
                sample_count = 0
                sample_succeed = False
                if epi_selecting and len(self.epi_num_pool) > self.epi_horizon:
                    while not sample_succeed and sample_count < 5:
                        experiences = self.memory.sample(self.batch_size)
                        sample_count += 1
                        if experiences[-1].episode_reward > self.epirew_higher_bound:
                            sample_succeed = True
                        elif experiences[-1].episode_reward < self.epirew_lower_bound:
                            pass
                        else: 
                            prob = (experiences[-1].episode_reward - self.epirew_lower_bound) / (self.epirew_higher_bound - self.epirew_lower_bound)
                            if np.random.rand() < prob:
                                sample_succeed = True
                else:
                    experiences = self.memory.sample(self.batch_size)
                assert len(experiences) == self.batch_size

                state0 = []
                reward = []
                action = []
                terminal = []
                state1 = []
                for e in experiences:
                    state0.append(e.state0)
                    state1.append(e.state1)
                    reward.append([e.reward])
                    action.append(e.action)
                    terminal.append(0. if e.terminal1 else 1.)

                state0 = np.squeeze(state0, axis=1)
                state1 = np.squeeze(state1, axis=1)
                
                self.learn(state0, action, reward, state1)

    # play one episode; agent interact with one Env_Ins 
    def play(self, env, epi_index, episode_max_steps, noise_level, epi_sel, act_repetition=1):
        self.training = True
        timer = time.time()
        if act_repetition < 1:
            raise ValueError('act_repetition must be >= 1, is {}.'.format(act_repetition))

        obs = None
        old_obs = None
        episode_step = None
        episode_reward = None
        epi_horizon_pool = []
        scale_factor = 10

        noise_source = NormalNoise()
        for j in range(200):
            noise_source.sample((self.action_dims,),noise_level)

        try:
            if obs is None:
                episode_step = 0
                episode_reward = 0.
                episode_memory = []
            obs = env.reset()
            obs, old_obs = transform_observation(obs, old_obs, episode_step)
        except Exception as e:
            print('(Agent) something wrong on env.reset()')
            traceback.print_exc()
            print(e)
            return

        while episode_step < episode_max_steps:
            obs_before_act = obs
            action = self.forward(obs_before_act)
            noise_sample = noise_source.sample((self.action_dims,),noise_level)
            noise_sample *= self.action_multiplier
            action += noise_sample
            action = np.round(np.clip(action, a_min=0.0, a_max=1.0))
            reward = 0.
            done = False
            for _ in range(act_repetition):
                try:
                    obs, rew, done, info = env.step(action)
                    episode_step += 1
                    reward += rew
                    obs, old_obs = transform_observation(obs, old_obs, episode_step)
                    if episode_step > episode_max_steps:
                        done = True
                        break                        
                except Exception as e:
                    print('(Agent) something wrong on env.step')
                    traceback.print_exc()
                    print(e)
                    return
            reward *= scale_factor
            episode_reward += reward
            episode_memory.append([obs_before_act, action, reward, done])
            self.backward(epi_sel)
            if done:
                action = self.forward(obs)
                episode_memory.append([obs, action, 0, False])  #append the last sample for an episode
                self.backward(epi_sel)  #train after each episode or say each threading is done
                for j in range(len(episode_memory)):
                    episode_memory[j].append(episode_reward)
                assert episode_memory[-1][-1] == episode_memory[-2][-1]
                break

        totaltime = time.time() - timer
        
        self.lock.acquire()
        self.memory_append(episode_memory)
        self.epi_num_pool.append(epi_index)
        self.epi_reward_pool.append(episode_reward)
        if len(self.epi_reward_pool) > self.epi_horizon:
            epi_horizon_pool = self.epi_reward_pool[len(self.epi_reward_pool)-self.epi_horizon:len(self.epi_reward_pool)]
            epi_horizon_pool.sort()   #make samples in memory ordered via epi_reward
            self.epirew_lower_bound = epi_horizon_pool[int(self.epi_horizon*0.1)]
            self.epirew_higher_bound = epi_horizon_pool[int(self.epi_horizon*0.9)]
        print('(FINISH)No.{}/{} episode done after {} steps in {:.2f} sec and got episode_reward: {}'
            .format(epi_index, self.nb_max_episodes, episode_step, totaltime, episode_reward/scale_factor))
        self.lock.release()

        return

    def test(self, env, episode_max_steps, act_repetition=1):
        self.training = False
        timer = time.time()
        if act_repetition < 1:
            raise ValueError('act_repetition must be >= 1, is {}.'.format(act_repetition))

        obs = None
        old_obs = None
        episode_step = None
        episode_reward = None

        try:
            if obs is None:
                episode_step = 0
                episode_reward = 0.
            obs = env.reset()
            obs, old_obs = transform_observation(obs, old_obs, episode_step)
        except Exception as e:
            print('(Agent) something wrong on env.reset()')
            traceback.print_exc()
            print(e)
            return

        while episode_step < episode_max_steps:
            obs_before_act = obs
            action = self.forward(obs_before_act)
            reward = 0.
            done = False
            for _ in range(act_repetition):
                try:
                    obs, rew, done, info = env.step(action)
                    episode_step += 1
                    reward += rew
                    obs, old_obs = transform_observation(obs, old_obs, episode_step)
                    if episode_step > episode_max_steps:
                        done = True
                        break                        
                except Exception as e:
                    print('(Agent) something wrong on env.step')
                    traceback.print_exc()
                    print(e)
                    return
            episode_reward += reward
            if done:    
                break

        totaltime = time.time() - timer
        print('One episode for testing done after {} steps in {:.2f} sec and got episode_reward: {}'
                .format(episode_step, totaltime, episode_reward))

    def submit(self, client, token):
        # Initialization
        self.training = False
        old_obs = None
        episode_num = 1
        episode_step = 0
        episode_reward = 0.

        # Creat environment
        obs = client.env_create(token)
        obs, old_obs = transform_observation(obs, old_obs, episode_step)

        while True:
            [obs, rew, done, info] = client.env_step(self.forward(obs).tolist(), True)
            obs, old_obs = transform_observation(obs, old_obs, episode_step)
            episode_reward += rew
            episode_step += 1
            if done:
                print ('\n>>>>>>>episode',episode_num,' DONE after',episode_step,'got_reward',episode_reward)
                obs = client.env_reset()
                if not obs:
                    break
                episode_num += 1
                old_obs = None
                episode_step = 0
                episode_reward = 0.
                obs, old_obs = transform_observation(obs, old_obs, episode_step)
        client.submit()


    def save_weights(self, filename):
        self.lock.acquire()
        save_path = filename + '.ckpt'
        self.saver.save(self.sess, save_path)
        print('(Agent) save model to {}.'.format(save_path))
        self.lock.release()

    def load_weights(self, filename):
        self.lock.acquire()
        save_path = filename + '.ckpt'
        self.saver.restore(self.sess, save_path)
        print('(Agent) load model from {}.'.format(save_path))
        self.lock.release()

    def save_memory(self, filename):
        self.lock.acquire()
        fn = filename + '.pkl'
        with open(fn, 'w') as f:
            pickle.dump(self.memory, f)
            print('(Agent) save memory to {}.'.format(fn))
        self.lock.release()

    def load_memory(self, filename):
        self.lock.acquire()
        fn = filename + '.pkl'
        with open(fn, 'r') as f:
            self.memory = pickle.load(f)
            print('(Agent) load memory from'.format(fn))
        self.lock.release()

