import tensorflow as tf
import tensorlayer as tl
import numpy as np
import opensim as osim
import cPickle as pickle

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import threading as th
import multiprocessing as mp

from rl.models import Actor, Critic
from rl.processor2 import transform_observation
from rl.noise import NormalNoise, GaussianWhiteNoiseProcess, OrnsteinUhlenbeckProcess
from copy import copy
from math import *

import time
import random
import traceback

class DDPGAgent_t2(object):
    def __init__(self, memory, observation_dims, action_dims, batch_size, 
        nb_max_episodes, gamma=.995):
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
        self.actor_lr = 1e-4
        self.critic_lr = 1e-3
        self.epi_horizon = 500
        self.epi_num_pool = []
        self.epi_reward_pool= []
        self.epirew_lower_bound = 0
        self.epirew_higher_bound = 0
        self.learn_counter = 0

        # Related objects
        self.memory = memory
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
            self.a = self._build_actor_net(self.S, scope_name='eval_net', reuse=False,)
            self.a_ = self._build_actor_net(self.S_, scope_name='target_net', reuse=False)
        with tf.variable_scope('Critic'):
            self.q = self._build_critic_net(self.S, self.a, scope_name='eval_net', reuse=False)
            self.q_predict = self._build_critic_net(self.S, self.A, scope_name='eval_net', reuse=True)
            self.q_ = self._build_critic_net(self.S_, self.a_, scope_name='target_net', reuse=False)
        
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
        # 1.update the critic
        self.q_target = self.R + self.gamma * self.q_
        self.critic_loss = tf.losses.mean_squared_error(labels=self.q_target, predictions=self.q_predict)
        self.ctrain = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss, 
            var_list=self.ce_params)

        # 2.update the actor
        self.actor_loss = - tf.reduce_mean(self.q)
        self.atrain = tf.train.AdamOptimizer(self.actor_lr).minimize(self.actor_loss, 
            var_list=self.ae_params)
        
        # 3.update the target net
        self.update_critic_target = [tf.assign(t, (1-self.tau)*t+self.tau*e) 
            for t, e in zip(self.ct_params, self.ce_params)]
        self.update_actor_target = [tf.assign(t, (1-self.tau)*t+self.tau*e) 
            for t, e in zip(self.at_params, self.ae_params)]

    def _build_actor_net(self, actor_input, scope_name, reuse=False):
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            tl.layers.set_name_reuse(reuse)

            actor = tl.layers.InputLayer(actor_input, name='actor_input_layer')
            actor = tl.layers.DenseLayer(actor, n_units=96, name='dense_layer1', 
                                        act= tf.nn.relu)
            actor = tl.layers.ReshapeLayer(actor, [-1, 96, 1], name='reshape_layer')     
            actor = tl.layers.Conv1dLayer(actor, shape=[5, 1, 8], name='conv_layer1', padding='VALID',
                                        act= tf.nn.relu)
            actor = tl.layers.Conv1dLayer(actor, shape=[3, 8, 4], name='conv_layer2', padding='VALID',
                                        act= tf.nn.relu)
            actor = tl.layers.Conv1dLayer(actor, shape=[3, 4, 2], name='conv_layer3', padding='VALID',
                                        act= tf.nn.relu)
            actor = tl.layers.FlattenLayer(actor,name ='flatten_layer')
            actor = tl.layers.DenseLayer(actor, n_units=128, name='dense_layer2', 
                                        act= tf.nn.relu)
            actor = tl.layers.DenseLayer(actor, n_units=48, name='dense_layer3', 
                                        act= tf.nn.relu)
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
                                        act= tf.nn.relu)
            critic = tl.layers.ReshapeLayer(critic, [-1, 128, 1], name ='reshape_layer')
            critic = tl.layers.Conv1dLayer(critic, shape=[5, 1, 8], name='conv_layer1', padding='VALID',
                                        act= tf.nn.relu)
            critic = tl.layers.Conv1dLayer(critic, shape=[3, 8, 4], name='conv_layer2', padding='VALID',
                                        act= tf.nn.relu)
            critic = tl.layers.Conv1dLayer(critic, shape=[3, 4, 2], name='conv_layer3', padding='VALID',
                                        act= tf.nn.relu)
            critic = tl.layers.FlattenLayer(critic,name ='flatten_layer')
            critic = tl.layers.DenseLayer(critic, n_units=128, name='dense_layer2', 
                                        act= tf.nn.relu)
            critic = tl.layers.DenseLayer(critic, n_units=96, name='dense_layer3',
                                        act= tf.nn.relu)
            critic = tl.layers.DenseLayer(critic, n_units=1, name='dense_layer4' )
            critic_output = critic.outputs
            print 'q-value shape', critic_output.get_shape()
            # critic.print_layers()
            # critic.print_params()
        return critic_output

    def learn(self, s, a, r, s_):
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
            self.memory.append(epi_exps[i][0], epi_exps[i][1], epi_exps[i][2], 
                            epi_exps[i][3], epi_exps[i][4], training=self.training)

    def plot_fig(self, filename):
        self.lock.acquire()
        save_path = filename + '.png'
        nb_epi = range(1, len(self.epi_num_pool)+1)
        plt.plot(nb_epi, self.epi_reward_pool, color='orange')
        plt.xlabel('Episode Index')
        plt.ylabel('Episode Reward')
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

    # play one episode
    def play(self, env, epi_index, episode_max_steps, epi_sel, act_repetition=1):
        self.training = True
        timer = time.time()
        if act_repetition < 1:
            raise ValueError('act_repetition must be >= 1, is {}.'.format(act_repetition))

        obs = None
        old_obs = None
        episode_step = None
        episode_reward = None
        epi_horizon_pool = []
        noise_source = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.2, size=18)

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
            noise_sample = noise_source.sample()
            action += noise_sample
            action = np.clip(action, a_max=1.0, a_min=0.0)
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
            episode_memory.append([obs_before_act, action, reward, done])
            self.backward(epi_sel)
            if done:
                action = self.forward(obs)
                episode_memory.append([obs, action, 0., False])
                self.backward(epi_sel)
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
            epi_horizon_pool.sort()
            self.epirew_lower_bound = epi_horizon_pool[int(self.epi_horizon*0.1)]
            self.epirew_higher_bound = epi_horizon_pool[int(self.epi_horizon*0.9)]
        print('(FINISH)No.{}/{} episode done after {} steps in {:.2f} sec and got episode_reward: {}'
            .format(epi_index, self.nb_max_episodes, episode_step, totaltime, episode_reward))
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
        obs = client.env_creat(token)
        obs, old_obs = transform_observation(obs, old_obs, episode_step)

        while True:
            [obs, rew, done, info] = client.env_step(self.forward(obs).tolist(), True)
            obs, old_obs = transform_observation(obs, old_obs, act_rep)
            episode_reward += rew
            episode_step += 1
            if done:
                print ('\n>>>>>>>episode',episode_num,' DONE after',episode_step,'got_reward',episode_reward)
                obs = client.env_reset()
                if not observation:
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

