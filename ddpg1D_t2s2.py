import cPickle as cPickle
import tensorflow as tf
import numpy as np
import threading as th
import multiprocessing as mp

import sys
import math
import time
import argparse

import opensim as osim
from osim.env import *
from osim.http.client import Client

from rl.agents import DDPGAgent_t2p_kfac
from rl.memory import SequentialMemory
from rl.noise import NormalNoise, GaussianWhiteNoiseProcess
from rl.multiEnv import EnvPool

## Command line parameters
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false', default=False)
parser.add_argument('--maxepi', dest='maxepi', action='store', default=50000, type=int)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--model', dest='model', action='store', default="test")
parser.add_argument('--submit', dest='submit', action='store_true', default=False)
args = parser.parse_args()


## Env
env = RunEnv(args.visualize)
env_pool = EnvPool(16)  #used to control how many processes are created

## Parameters
memory = SequentialMemory(limit=1000000, window_length=1)
# random_process = GaussianWhiteNoiseProcess(mu=0., sigma=1., sigma_min=0.25, 
#     n_steps_annealing=1000, size=env.noutput)
random_process = NormalNoise()
observation_dims = 62
action_dims = env.action_space.shape[0]
batch_size = 64
nb_max_episodes = args.maxepi
episode_max_steps = env.timestep_limit


## Submition Setting
remote_base = 'http://grader.crowdai.org:1729'
# crowdai_token = '024126b127f6ded99757e4c904a05bb7'
# crowdai_token = 'f5a99af8502f6fae1a30e08984de6a78' #Ding
crowdai_token = '25dba1af410439b4458447394eefe2b7'  #LittleZ
client = Client(remote_base)

## Agent
agent = DDPGAgent_t2p_kfac(memory = memory,
                    noise = random_process,
                    observation_dims = observation_dims,
                    action_dims = action_dims,
                    batch_size = batch_size,
                    nb_max_episodes = nb_max_episodes)

def play_threading(env_ins, epi_index, epi_max_steps, noise_level, epi_sel, act_rep):
    agent.play(env_ins, epi_index, epi_max_steps, noise_level, epi_sel, act_rep)
    env_ins.release()
            
def play_if_available(env_pool, epi_index, epi_max_steps, noise_level, epi_sel, act_rep):
    while True:  #keep acquiring available env from pool
        env_ins = env_pool.acquire_env()    #env_ins is a class Env_Ins, containing a running env(a running process)
        if env_ins == False:
            pass
        else:
            t = th.Thread(target=play_threading,    #multithreading,target is a function to execute
                        args=(env_ins, epi_index, epi_max_steps, noise_level, epi_sel, act_rep))
            t.daemon = True
            t.start()
            break

def train(nb_train, env_pool, epi_max_steps, model_name, epi_sel, act_rep=1):
    agent.training = True
    noise_level = 2.
    noise_decay_rate = 0.005
    noise_floor = 0.05
    noiseless = 0.01

    for i in xrange(nb_train):  #xrange = list(range) is a range generator ; nb_train is max #episodes, default 50000
        print('(START)No.{}/{} episode has began...'.format(i+1,nb_train))
        epi_index = i+1

        noise_level *= (1-noise_decay_rate)
        noise_level = max(noise_floor, noise_level)
        nl = noise_level if np.random.uniform()>0.05 else noiseless    #95% probability with almost no noise 
        
        play_if_available(env_pool, epi_index, epi_max_steps, nl, epi_sel, act_rep)  #this is the core of train, it induces thread--play_threading and then induce agent.play
        time.sleep(0.05)
        #sava weights, memory and plot
        if epi_index % 3000 == 0 or epi_index == nb_train:
            filename = model_name + str(epi_index)
            agent.save_weights(filename)
            agent.save_memory(filename)
        if epi_index % 50 == 0 or epi_index == nb_train:
            # filename = model_name + str(epi_index)
            agent.plot_fig(model_name)

## Main
if args.train:
    #agent.load_memory(args.model)
    #agent.load_weights(args.model)
    train(nb_train = nb_max_episodes, env_pool = env_pool, epi_max_steps= episode_max_steps, 
        model_name = args.model, epi_sel=False, act_rep = 2)   #epi_sel is a sign for using sample method for bp process

if not args.train and not args.submit:
    agent.load_weights(args.model)
    agent.test(env, episode_max_steps)

if not args.train and args.submit:
    agent.load_weights(args.model)
    agent.submit(client=client, token=crowdai_token)





