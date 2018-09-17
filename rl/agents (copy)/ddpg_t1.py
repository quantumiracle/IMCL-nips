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
from copy import copy
from math import *

import time
import random
import traceback

class DDPGAgent_t1(object):
    def __init__(self, memory, noise, observation_dims, action_dims, batch_size, 
        nb_max_episodes, gamma=.99):
        # Parameters
        self.observation_dims = observation_dims
        self.action_dims = action_dims
        self.batch_size = batch_size
        self.nb_max_episodes = nb_max_episodes
        self.gamma = gamma

        self.learn_counter = 0
        self.train_multiplier = 2
        self.actor_lr = 3e-4
        self.critic_lr = 3e-4
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

        # Networks
        self.actor = Actor(self.action_dims, name='actor')
        self.actor_target = Actor(self.action_dims, name='actor_target')
        self.critic = Critic(name='critic')
        self.critic_target = Critic(name='critic_target')

        self.learn, self.choose_action, sync_target = self.generate_train_op()
        
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        sync_target()

    def generate_train_op(self):
        S = tf.placeholder(tf.float32, shape=[None, self.observation_dims], name='s')
        S_ = tf.placeholder(tf.float32, shape=[None, self.observation_dims], name='s_')
        A = tf.placeholder(tf.float32, shape=[None, self.action_dims], name='a')
        R = tf.placeholder(tf.float32, shape=[None, 1], name='r')
        isdone = tf.placeholder(tf.float32, shape=[None, 1], name='isdone')
        tau = tf.Variable(1e-3)

        a = self.actor(S)
        a_ = self.actor_target(S_)
        q = self.critic(a, S)
        q_ = self.critic_target(a_, S_)

        # 1.update the critic
        q_predict = self.critic(A, S, reuse=True)
        q_target = R + (1-isdone) * self.gamma * q_
        critic_loss = tf.reduce_mean((q_target - q_predict)**2)
    
        # 2.update the actor
        actor_loss = tf.reduce_mean(-q)

        ae_params = self.actor.vars
        at_params = self.actor_target.vars
        ce_params = self.critic.vars
        ct_params = self.critic_target.vars

        critic_train = tf.train.AdamOptimizer(self.critic_lr).minimize(critic_loss, 
            var_list=ce_params)
        actor_train = tf.train.AdamOptimizer(self.actor_lr).minimize(actor_loss, 
            var_list=ae_params)
        
        # 3.update the target net
        update_critic_target = [tf.assign(t, (1-tau)*t+tau*e) 
            for t, e in zip(ct_params, ce_params)]
        update_actor_target = [tf.assign(t, (1-tau)*t+tau*e) 
            for t, e in zip(at_params, ae_params)]

        def learn(sd, ad, rd, isdoned, s_d):
            results = self.sess.run([critic_loss, actor_loss, critic_train, actor_train, 
                update_critic_target, update_actor_target],
                feed_dict={S:sd, S_:s_d, A:ad, isdone:isdoned, R:rd, tau:1e-3})
                
            self.learn_counter += 1
            if self.learn_counter%10 == 0:
                print(' '*30, 'closs: {:6.4f} aloss: {:6.4f}'.format(
                    results[0],results[1])) 


        def choose_action(sd):
            return self.sess.run(a, feed_dict={S:sd})

        def sync_target():
            self.sess.run([update_critic_target, update_actor_target], feed_dict={tau:1.})
        
        return learn, choose_action, sync_target

    
    def memory_append(self, epi_exps):
        for i in xrange(len(epi_exps)):
            if np.random.uniform() > 0.5:
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

        if self.training and self.noise is not None:
            noise_sample = self.noise.sample()
            action += noise_sample
            action = np.clip(action, a_max=1., a_min=0.)

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
                isdone = []
                state1 = []
                for e in experiences:
                    state0.append(e.state0)
                    state1.append(e.state1)
                    reward.append([e.reward])
                    action.append(e.action)
                    isdone.append([1. if e.terminal1 else 0.])

                state0 = np.squeeze(state0, axis=1)
                state1 = np.squeeze(state1, axis=1)
                
                self.learn(state0, action, reward, isdone, state1)

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
                episode_memory.append([obs, action, 0, False])
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

