#!/usr/bin/env python

from mpi4py import MPI
from baselines.common import set_global_seeds
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

def train(env_id, args):
    from baselines.ppo1 import cnn_policy
    import baselines.common.tf_util as U
    if args.nokl:
        from baselines.ppo1 import nokl_pposgd_simple as pposgd_simple
    else:
        from baselines.ppo1 import pposgd_simple

    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    print('_'.join([str(arg) for arg in vars(args)]))
    logdir = osp.join('./result/', '_'.join([str(getattr(args, arg)) for arg in vars(args)]))
    logger.configure(dir=logdir)
    workerseed = args.seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = make_atari(env_id)
    def policy_fn(name, ob_space, ac_space): #pylint: disable=W0613
        return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space)
    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    env = wrap_deepmind(env)
    env.seed(workerseed)

    pposgd_simple.learn(env, policy_fn,
        max_timesteps=int(args.num_timesteps * 1.1),
        timesteps_per_actorbatch=args.timesteps_per_actorbatch,
        clip_param=args.clip, entcoeff=args.entcoeff,
        optim_epochs=args.optim_epochs, optim_stepsize=args.optim_stepsize, optim_batchsize=args.optim_batchsize,
        gamma=0.99, lam=0.95,
        schedule='linear'
    )
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='EnduroNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e5))
    parser.add_argument('--nokl', type=int, default=0)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--timesteps-per-actorbatch', type=int, default=256)
    parser.add_argument('--optim-epochs', type=int, default=4)
    parser.add_argument('--optim-batchsize', type=int, default=64)
    parser.add_argument('--entcoeff', type=float, default=0.01)
    parser.add_argument('--optim-stepsize', type=float, default=1e-3)

    args = parser.parse_args()
    train(args.env, args)

if __name__ == '__main__':
    main()
