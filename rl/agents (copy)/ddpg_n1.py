import tensorflow as tf
import tensorlayer as tl
import numpy as np
import opensim as osim
import cPickle as pickle

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import threading as th
import multiprocessing as mp

from rl.memory import SequentialMemory
from rl.noise import OrnsteinUhlenbeckProcess
from rl.processor2 import transform_observation
from math import *

import time
import random
import traceback

class Actor(object):
    def __init__(self, sess, S, S_, act_dims, learning_rate, t_replace_iter=1):
        self.sess = sess
        self.S = S
        self.S_ = S_
        self.act_dims = act_dims
        self.learning_rate = learning_rate
        self.t_replace_iter = t_replace_iter

        self.t_replace_counter = 0
        self.tau = 1e-3
        self.action_multiplier = 0.5
        self.action_bias = 0.5

        with tf.variable_scope('Actor'):
            self.a = self._build_net(self.S, scope='eval_net')
            self.a_ = self._build_net(self.S_, scope='target_net')
        
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

        self.update_t = [tf.assign(t, (1-self.tau)*t+self.tau*e) for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, actor_input, scope):
        with tf.variable_scope(scope):
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
            actor = tl.layers.DenseLayer(actor, n_units=96, name='dense_layer2', 
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            actor = tl.layers.DenseLayer(actor, n_units=48, name='dense_layer3', 
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            actor = tl.layers.DenseLayer(actor, n_units=self.act_dims, act=tf.nn.tanh, name='dense_layer4')
            actor = tl.layers.LambdaLayer(actor, lambda a: a*self.action_multiplier + self.action_bias, name='lambda_layer')
            actor_output = actor.outputs
            print 'action shape', actor_output.get_shape()
            # actor.print_layers()
            # actor.print_params()
        return actor_output

    def learn(self, s, a):
        self.sess.run(self.train_op, feed_dict={self.S:s})

        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run(self.update_t)
        self.t_replace_counter += 1

    def choose_action(self, s):
        return self.sess.run(self.a, feed_dict={self.S:s})
    
    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)
        
        with tf.variable_scope('Actor_train'):
            opt = tf.train.AdamOptimizer(-self.learning_rate)
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


class Critic(object):
    def __init__(self, sess, S, S_, R, A, A_, obs_dims, act_dims, learning_rate, t_replace_iter=1, gamma=0.99):
        self.sess = sess
        self.S = S
        self.S_ = S_
        self.R = R
        self.A = A
        self.A_ = A_
        self.obs_dims = obs_dims
        self.act_dims = act_dims
        self.learning_rate = learning_rate
        self.t_replace_iter = t_replace_iter
        self.gamma = gamma

        self.t_replace_counter = 0
        self.tau = 1e-3

        with tf.variable_scope('Critic'):
            self.q = self._build_net(self.S, self.A, scope='eval_net')
            self.q_ = self._build_net(self.S_, self.A_, scope='target_net')

        with tf.variable_scope('target_q'):
            self.target_q = self.R + self.gamma * self.q_
        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))
        with tf.variable_scope('Critic_train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, self.A)[0]

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        self.update_t = [tf.assign(t, (1-self.tau)*t+self.tau*e) for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, critic_obs_input, critic_act_input, scope):
        with tf.variable_scope(scope):  
            critic1 = tl.layers.InputLayer(critic_obs_input, name='obs_input_layer')
            critic2 = tl.layers.InputLayer(critic_act_input, name='act_input_layer')
            critic = tl.layers.ConcatLayer(layer=[critic1, critic2], concat_dim=1, name='concat_input_layer')
            critic = tl.layers.DenseLayer(critic, n_units=96, name='dense_layer1', 
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            critic = tl.layers.ReshapeLayer(critic, [-1, 96, 1], name ='reshape_layer')
            critic = tl.layers.Conv1dLayer(critic, shape=[5, 1, 8], name='conv_layer1', padding='VALID',
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            critic = tl.layers.Conv1dLayer(critic, shape=[3, 8, 4], name='conv_layer2', padding='VALID',
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            critic = tl.layers.Conv1dLayer(critic, shape=[3, 4, 2], name='conv_layer3', padding='VALID',
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            critic = tl.layers.FlattenLayer(critic,name ='flatten_layer')
            critic = tl.layers.DenseLayer(critic, n_units=96, name='dense_layer2', 
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            critic = tl.layers.DenseLayer(critic, n_units=48, name='dense_layer3',
                                        act= lambda x : tl.act.lrelu(x, 0.2))
            critic = tl.layers.DenseLayer(critic, n_units=1, name='dense_layer4' )
            critic_output = critic.outputs
            print 'q-value shape', critic_output.get_shape()
            # critic.print_layers()
            # critic.print_params()
        return critic_output

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={self.S:s, self.A:a, self.R:r, self.S_:s_})

        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run(self.update_t)
        self.t_replace_counter += 1

class DDPGAgent_n1(object):
    def __init__(self, memory, noise, observation_dims, action_dims, batch_size, nb_max_episodes, gamma=.99):
        # Parameters
        self.observation_dims = observation_dims
        self.action_dims = action_dims
        self.batch_size = batch_size
        self.nb_max_episodes = nb_max_episodes
        self.gamma = gamma

        self.train_multiplier = 2
        self.actor_learning_rate = 2e-4
        self.critic_learning_rate = 3e-4
        self.epi_horizon = 500
        self.epi_num_pool = []
        self.epi_reward_pool= []
        self.epirew_lower_bound = 0
        self.epirew_higher_bound = 0

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
        with tf.name_scope('R'):
            self.R = tf.placeholder(tf.float32, shape=[None, 1], name='r')
        with tf.name_scope('S_'): 
            self.S_ = tf.placeholder(tf.float32, shape=[None, self.observation_dims], name='s_')
        
        # Networks
        self.actor = Actor(self.sess, self.S, self.S_, self.action_dims, self.actor_learning_rate)           
        self.critic = Critic(self.sess, self.S, self.S_, self.R, self.actor.a, self.actor.a_, 
                        self.observation_dims, self.action_dims, self.critic_learning_rate)
        self.actor.add_grad_to_graph(self.critic.a_grads)

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
    
    def memory_append(self, epi_exps):
        for i in xrange(len(epi_exps)):
            self.memory.append(epi_exps[i][0], epi_exps[i][1], epi_exps[i][2], 
                            epi_exps[i][3], epi_exps[i][4])

    def plot_fig(self, filename):
        self.lock.acquire()
        save_path = filename + '.png'
        nb_epi = range(1, len(self.epi_num_pool)+1)
        plt.plot(nb_epi, self.epi_reward_pool, color='orange')
        plt.xlabel('Episode Index')
        plt.ylabel('Episode Reward')
        plt.savefig(save_path)
        self.lock.release()

    def forward(self, obs, noise_level):
        state = self.memory.get_recent_state(obs)
        action = self.actor.choose_action(state)
        action = np.reshape(action, (self.action_dims,))
        assert action.shape == (self.action_dims,)

        if self.training and self.noise is not None:
            noise_sample = self.noise.sample((self.action_dims,), noise_level)
            action += noise_sample * self.actor.action_multiplier
            action = np.clip(action, a_max=1., a_min=0.)

        return action 

    def backward(self):
        if self.memory.nb_entries > self.batch_size*128:
            for _ in xrange(self.train_multiplier):
                # choose which experiences should be learned
                sample_count = 0
                sample_succeed = False
                if len(self.epi_num_pool) > self.epi_horizon:
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
                
                self.critic.learn(state0, action, reward, state1)
                self.actor.learn(state0, action)

    # play one episode
    def play(self, env, epi_index, episode_max_steps, noise_level, act_repetition=1):
        timer = time.time()
        episode_step = 0
        episode_reward = 0.
        episode_memory = []
        old_obs = None

        try:
            obs = env.reset()
            obs, old_obs = transform_observation(obs, old_obs, episode_step)
        except Exception as e:
            print('(Agent) something wrong on env.reset()')
            traceback.print_exc()
            print(e)
            return

        while episode_step <= episode_max_steps:
            obs_before_act = obs
            action = self.forward(obs_before_act, noise_level)
            for _ in range(act_repetition):
                try:
                    obs, rew, done, info = env.step(action)
                    episode_step += 1
                    episode_reward += rew
                    obs, old_obs = transform_observation(obs, old_obs, episode_step)
                    if episode_step > episode_max_steps:
                        done = True
                        break                        
                except Exception as e:
                    print('(Agent) something wrong on env.step')
                    traceback.print_exc()
                    print(e)
                    return
            if self.training == True:
                episode_memory.append([obs_before_act, action, rew, done])
                self.backward()
            if done:
                for j in range(len(episode_memory)):
                    episode_memory[j].append(episode_reward)
                assert episode_memory[-1][-1] == episode_memory[-2][-1]
                break

        totaltime = time.time() - timer
        if self.training == True:
            self.lock.acquire()
            self.memory_append(episode_memory)
            self.epi_num_pool.append(epi_index)
            self.epi_reward_pool.append(episode_reward)
            print('(FINISH)No.{}/{} episode done after {} steps in {:.2f} sec and got episode_reward: {}'
                .format(epi_index, self.nb_max_episodes, episode_step, totaltime, episode_reward))
            self.lock.release()
        else:
            print('One episode for testing done after {} steps in {:.2f} sec and got episode_reward: {}'
                .format(episode_step, totaltime, episode_reward))

        return

    def test(self, env, episode_max_steps, act_repetition=1):
        self.training = False
        self.play(env, episode_max_steps, act_repetition)

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
        self.saver.load(self.sess, save_path)
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

