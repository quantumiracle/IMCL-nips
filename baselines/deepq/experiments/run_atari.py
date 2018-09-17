import gym

from baselines import deepq
from baselines.common import set_global_seeds
from baselines import bench
import argparse
import os.path as osp
from baselines import logger
from baselines.common.atari_wrappers import make_atari

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='PongNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=1)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    parser.add_argument('--logdir', type=str, default='~/.tmp/deepq')

    # General hyper-parameters 
    parser.add_argument('--isKfac', type=int, default=0)
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=160)
    parser.add_argument('--target_network_update_freq', type=int, default=1000)
    
    # Kfac parameters
    parser.add_argument('--kfac_fisher_metric', type=str, default='gn')        
    parser.add_argument('--kfac_momentum', type=float, default=0.9)
    parser.add_argument('--kfac_clip_kl', type=float, default=0.01)
    parser.add_argument('--kfac_epsilon', type=float, default=1e-2)
    parser.add_argument('--kfac_stats_decay', type=float, default=0.99)
    parser.add_argument('--kfac_cold_iter', type=float, default=10)

    args = parser.parse_args()
    print('_'.join([str(arg) for arg in vars(args)]))
    logdir = osp.join(args.logdir, '_'.join([str(getattr(args, arg)) for arg in vars(args) if arg != 'logdir']))
    logger.configure(dir=logdir)
    #Get parameters in kfac
    kfac_paras = {}
    for arg in vars(args):
        if arg[:4] == 'kfac':
            kfac_paras[arg[5:]] = getattr(args, arg)

    set_global_seeds(args.seed)
    env = make_atari(args.env)
    env = bench.Monitor(env, logger.get_dir())
    env = deepq.wrap_atari_dqn(env)
    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (32, 3, 1)],
        hiddens=[256],
        dueling=bool(args.dueling),
    )
    act = deepq.learn(
        env,
        q_func=model,
        isKfac=args.isKfac,
        kfac_paras=kfac_paras,
        lr=args.lr,
        max_timesteps=args.num_timesteps,
        batch_size=args.batch_size,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=args.target_network_update_freq,
        gamma=0.99,
        prioritized_replay=bool(args.prioritized)
    )
    # act.save("pong_model.pkl") XXX
    env.close()


if __name__ == '__main__':
    main()
