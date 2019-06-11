#!/usr/bin/env python3
import functools
import os
import numpy as np
from numbers import Number

from baselines import logger
from mpi4py import MPI
import mpi_util
import tf_util
from cmd_util import make_atari_env, arg_parser
from policies.cnn_gru_policy_dynamics import CnnGruPolicy
from policies.cnn_policy_param_matched import CnnPolicy
from ppo_agent import PpoAgent
from utils import set_global_seeds
from vec_env import VecFrameStack

try:
    import nsml
    NSML = True
except:
    NSML = False

def train(*, env_id, num_env, hps, num_timesteps, seed):
    venv = VecFrameStack(
        make_atari_env(env_id, num_env, seed, wrapper_kwargs=dict(),
                       start_index=num_env * MPI.COMM_WORLD.Get_rank(),
                       max_episode_steps=hps.pop('max_episode_steps')),
        hps.pop('frame_stack'))
    venv.score_multiple = 1
    venv.record_obs = False
    ob_space = venv.observation_space
    ac_space = venv.action_space
    gamma = hps.pop('gamma')
    policy = {'rnn': CnnGruPolicy,
              'cnn': CnnPolicy}[hps.pop('policy')]

    agent = PpoAgent(
        scope='ppo',
        ob_space=ob_space,
        ac_space=ac_space,
        stochpol_fn=functools.partial(
            policy,
                scope='pol',
                ob_space=ob_space,
                ac_space=ac_space,
                update_ob_stats_independently_per_gpu=hps.pop('update_ob_stats_independently_per_gpu'),
                proportion_of_exp_used_for_predictor_update=hps.pop('proportion_of_exp_used_for_predictor_update'),
                exploration_type = hps.pop("exploration_type"),
                beta = hps.pop("beta"),
            ),
        gamma=gamma,
        gamma_ext=hps.pop('gamma_ext'),
        lam=hps.pop('lam'),
        nepochs=hps.pop('nepochs'),
        nminibatches=hps.pop('nminibatches'),
        lr=hps.pop('lr'),
        cliprange=0.1,
        nsteps=128,
        ent_coef=0.001,
        max_grad_norm=hps.pop('max_grad_norm'),
        use_news=hps.pop("use_news"),
        comm=MPI.COMM_WORLD if MPI.COMM_WORLD.Get_size() > 1 else None,
        update_ob_stats_every_step=hps.pop('update_ob_stats_every_step'),
        int_coeff=hps.pop('int_coeff'),
        ext_coeff=hps.pop('ext_coeff'),
        noise_type=hps.pop('noise_type'),
        noise_p=hps.pop('noise_p'),
        use_sched=hps.pop('use_sched'),
        num_env=num_env,
        exp_name=hps.pop('exp_name'),
    )
    agent.start_interaction([venv])
    if hps.pop('update_ob_stats_from_random_agent'):
        agent.collect_random_statistics(num_timesteps=128*50)
    assert len(hps) == 0, "Unused hyperparameters: %s" % list(hps.keys())

    counter = 0
    while True:
        info = agent.step()
        n_updates = 0

        if info['update']:
            logger.logkvs(info['update'])
            logger.dumpkvs()

            if NSML:
                n_updates = int(info['update']['n_updates'])
                nsml_dict = {k: np.float64(v) for k, v in info['update'].items() if isinstance(v, Number)}
                nsml.report(step=n_updates, **nsml_dict)

            counter += 1
        if n_updates >= 40*1000: # 40K updates
            break

    agent.stop_interaction()


def main():
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', default='FrostbiteNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--max_episode_steps', type=int, default=4500)

    parser.add_argument('--num-timesteps', type=int, default=int(1e12))
    parser.add_argument('--num_env', type=int, default=64)
    parser.add_argument('--use_news', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--update_ob_stats_every_step', type=int, default=0)
    parser.add_argument('--update_ob_stats_independently_per_gpu', type=int, default=0)
    parser.add_argument('--update_ob_stats_from_random_agent', type=int, default=1)
    parser.add_argument('--proportion_of_exp_used_for_predictor_update', type=float, default=1.)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--policy', type=str, default='cnn', choices=['cnn', 'rnn'])
    parser.add_argument('--int_coeff', type=float, default=1.)
    parser.add_argument('--ext_coeff', type=float, default=2.)
    parser.add_argument('--beta', type=float, default=1e-3)
    parser.add_argument('--exploration_type', type=str, default='bottleneck')
    parser.add_argument('--noise_type', type=str, default='box', choices=['none', 'box'])
    parser.add_argument('--noise_p', type=float, default=0.1)
    parser.add_argument('--use_sched', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='none')

    args = parser.parse_args()
    if args.policy == 'rnn':
        args.gamma_ext = 0.999
    else:
        args.gamma_ext = 0.99

    logger.configure(dir=logger.get_dir(), format_strs=['stdout', 'log', 'csv'] if MPI.COMM_WORLD.Get_rank() == 0 else [])
    if MPI.COMM_WORLD.Get_rank() == 0:
        with open(os.path.join(logger.get_dir(), 'experiment_tag.txt'), 'w') as f:
            f.write(args.tag)

    seed = 10000 * args.seed + MPI.COMM_WORLD.Get_rank()
    set_global_seeds(seed)

    hps = dict(
        frame_stack=4,
        nminibatches=4,
        nepochs=4,
        lr=0.0001,
        max_grad_norm=0.0,
        use_news=args.use_news,
        gamma=args.gamma,
        gamma_ext=args.gamma_ext,
        max_episode_steps=args.max_episode_steps,
        lam=args.lam,
        update_ob_stats_every_step=args.update_ob_stats_every_step,
        update_ob_stats_independently_per_gpu=args.update_ob_stats_independently_per_gpu,
        update_ob_stats_from_random_agent=args.update_ob_stats_from_random_agent,
        proportion_of_exp_used_for_predictor_update=args.proportion_of_exp_used_for_predictor_update,
        policy=args.policy,
        int_coeff=args.int_coeff,
        ext_coeff=args.ext_coeff,
        exploration_type = args.exploration_type,
        beta = args.beta,
        noise_type = args.noise_type,
        noise_p = args.noise_p,
        use_sched = args.use_sched,
        exp_name=args.exp_name,
    )

    tf_util.make_session(make_default=True)
    train(env_id=args.env, num_env=args.num_env, seed=seed,
        num_timesteps=args.num_timesteps, hps=hps)


if __name__ == '__main__':
    main()
