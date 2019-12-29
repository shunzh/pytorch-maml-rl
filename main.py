import random

import maml_rl.envs
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import json

from maml_rl.metalearner import MetaLearner, KPolicyMetaLearner, total_rewards
from maml_rl.policies import CategoricalMLPPolicy, NormalMLPPolicy
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler

from tensorboardX import SummaryWriter

def set_random_seed(rnd):
    torch.manual_seed(rnd)
    random.seed(rnd)
    np.random.seed(rnd)

def main(args):
    set_random_seed(args.random)

    continuous_actions = (args.env_name in ['AntVel-v1', 'AntDir-v1',
        'AntPos-v0', 'HalfCheetahVel-v1', 'HalfCheetahDir-v1',
        '2DNavigation-v0', '2DNavigationBiased-v0'])

    writer = SummaryWriter('./logs/{0}'.format(args.alg))
    save_folder = './saves/{0}'.format(args.alg)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(os.path.join(save_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)

    sampler = BatchSampler(args.env_name, batch_size=args.fast_batch_size,
        num_workers=args.num_workers, seed=args.random)
    if continuous_actions:
        policy = NormalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            int(np.prod(sampler.envs.action_space.shape)),
            hidden_sizes=(args.hidden_size,) * args.num_layers)
    else:
        policy = CategoricalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            sampler.envs.action_space.n,
            hidden_sizes=(args.hidden_size,) * args.num_layers)
    baseline = LinearFeatureBaseline(
        int(np.prod(sampler.envs.observation_space.shape)))

    if args.alg == 'simul':
        # vanilla maml
        metalearner = MetaLearner(sampler, policy, baseline, gamma=args.gamma,
            fast_lr=args.fast_lr, tau=args.tau, device=args.device)

        for batch in range(args.meta_policy_num * args.num_batches):
            # first sample tasks under the distribution
            tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
            episodes = metalearner.sample(tasks, first_order=args.first_order)
            metalearner.step(episodes, max_kl=args.max_kl, cg_iters=args.cg_iters,
                cg_damping=args.cg_damping, ls_max_steps=args.ls_max_steps,
                ls_backtrack_ratio=args.ls_backtrack_ratio)

            # Tensorboard
            writer.add_scalar('maml/before_update',
                total_rewards([ep.rewards for ep, _ in episodes]), batch)
            writer.add_scalar('maml/after_update',
                total_rewards([ep.rewards for _, ep in episodes]), batch)

            # Save policy network
            with open(os.path.join(save_folder,
                    'policy-{0}.pt'.format(batch)), 'wb') as f:
                torch.save(policy.state_dict(), f)

    elif args.alg == 'greedy':
        # multi-policy maml
        metalearner = KPolicyMetaLearner(sampler, policy, baseline, args.meta_policy_num, gamma=args.gamma,
            fast_lr=args.fast_lr, tau=args.tau, device=args.device)

        # visualize the poolicies' behavior
        trajectories = []
        for policy_idx in range(args.meta_policy_num):
            print(policy_idx)
            metalearner.optimize_policy_index(policy_idx)

            for batch in range(args.num_batches):
                print('batch num %d' % batch)

                tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
                metalearner.evaluate_optimized_policies(tasks)

                episodes = metalearner.sample(tasks, first_order=args.first_order)
                # loss is computed inside, then update policies
                metalearner.step(episodes, max_kl=args.max_kl, cg_iters=args.cg_iters,
                    cg_damping=args.cg_damping, ls_max_steps=args.ls_max_steps,
                    ls_backtrack_ratio=args.ls_backtrack_ratio)

                # not sure what to write in tensorboard...
                for epIdx in range(len(episodes)):
                    writer.add_scalar('kmaml/pi_' + str(policy_idx) + '_task_' + str(epIdx),
                        total_rewards([episodes[epIdx][1].rewards]), batch)
            # use a random task (no update here anyway) to visualize meta-policies
            tasks = sampler.sample_tasks(num_tasks=1)
            trajectories.append(metalearner.sample_meta_policy(tasks[0]))
        plotTrajectories(trajectories)

def plotTrajectories(trajectories):
    """
    plot a list of trajectories
    """
    for traj in trajectories:
        plt.plot(traj[:, 0].numpy(), traj[:, 1].numpy())
    plt.savefig('trajectories.pdf')
    plt.show()

if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML)')

    # General
    parser.add_argument('--env-name', type=str,
        help='name of the environment')
    parser.add_argument('--gamma', type=float, default=0.95,
        help='value of the discount factor gamma')
    parser.add_argument('--tau', type=float, default=1.0,
        help='value of the discount factor for GAE')
    parser.add_argument('--first-order', action='store_true',
        help='use the first-order approximation of MAML')
    parser.add_argument('--alg',type=str, default='maml',
        help='algorithm to run')
    parser.add_argument('--random', type=int, default=0,
        help='random seed')

    # Policy network (relu activation function)
    parser.add_argument('--hidden-size', type=int, default=100,
        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2,
        help='number of hidden layers')

    # Task-specific
    parser.add_argument('--fast-batch-size', type=int, default=20,
        help='batch size for each individual task')
    parser.add_argument('--fast-lr', type=float, default=0.5,
        help='learning rate for the 1-step gradient update of MAML')

    # Optimization
    parser.add_argument('--num-batches', type=int, default=200,
        help='number of batches')
    parser.add_argument('--meta-batch-size', type=int, default=40,
        help='number of tasks per batch')
    parser.add_argument('--max-kl', type=float, default=1e-2,
        help='maximum value for the KL constraint in TRPO')
    parser.add_argument('--cg-iters', type=int, default=10,
        help='number of iterations of conjugate gradient')
    parser.add_argument('--cg-damping', type=float, default=1e-5,
        help='damping in conjugate gradient')
    parser.add_argument('--ls-max-steps', type=int, default=15,
        help='maximum number of iterations for line search')
    parser.add_argument('--ls-backtrack-ratio', type=float, default=0.8,
        help='maximum number of iterations for line search')

    # Miscellaneous
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling')
    parser.add_argument('--device', type=str, default='cpu',
        help='set the device (cpu or cuda)')

    # For multi-policy
    parser.add_argument('--meta-policy-num', type=int, default=2,
        help='the number of policies to keep for meta-learning')

    args = parser.parse_args()

    # Create logs and saves folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./saves'):
        os.makedirs('./saves')
    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.alg += '-{0}'.format(os.environ['SLURM_JOB_ID'])

    main(args)
