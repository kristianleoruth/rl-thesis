import argparse
import gymnasium as gym
from gymnasium import ObservationWrapper
import stable_baselines3 as sb3
import sb3_contrib as sb3c
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
import ale_py
import random
import torch
import numpy as np
import multiprocessing as mp
import os
from typing import List, Optional, Union
os.environ["ALE_DISABLE_LOG"] = "1"

ASTEROIDS_ENVID = "ALE/Asteroids-v5"
# gym.register(ASTEROIDS_ENVID, )
# gym.register_envs(ale_py)

def linear_schedule(initial_lr: float, final_lr: float = 1e-5):
    def func(progress_remaining: float) -> float:
        return final_lr + (initial_lr - final_lr) * progress_remaining
    return func


def parse_mdl_args(argstr=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True, 
                        help="Algorithm (ppo, rppo, trpo, a2c)")
    parser.add_argument("--fc1", type=int, default=256, required=False,
                        help="Size of first fully connected layer")
    parser.add_argument("--fc2", type=int, default=128, required=False,
                        help="Size of second fully connected layer")
    parser.add_argument("--cnn", action="store_true", required=False,
                        help="Use convolutional (Cnn) policy")
    parser.add_argument("--lrsched", action="store_true", required=False,
                        help="Use scheduled learning rate")
    parser.add_argument("--lr", type=float, default=3e-4, required=False,
                        help="Define learning rate")
    parser.add_argument("--lrstart", "--lrs", type=float, default=3e-4, required=False,
                        help="Starting LR (LR schedule)")
    parser.add_argument("--lrend", "--lre", type=float, default=3e-4, required=False,
                        help="Ending LR (LR schedule)")
    parser.add_argument("--entcoef", "--entr", type=float, default=0.0, required=False,
                        help="Entropy coefficienct (used if available for model)")
    parser.add_argument("--kl", "--usekl", action="store_true", required=False,
                        help="Use KL target (PPO, RPPO)")
    parser.add_argument("--kltarg", "--klt", type=float, default=0.02, required=False,
                        help="KL target/constraint (PPO, RPPO, TRPO)")
    parser.add_argument("--clip", "--cliprange", type=float, default=0.2, required=False,
                        help="Clip range (PPO, RPPO)")
    parser.add_argument("--vfcoef", type=float, default=0.5, required=False,
                        help="Value function loss coefficient")
    parser.add_argument("--gae", type=float, default=0.95, required=False,
                        help="GAE lambda")
    parser.add_argument("--gamma", type=float, default=0.99, required=False,
                        help="Gamma value")
    parser.add_argument("--n_steps", type=int, default=4096, required=False,
                        help="n_steps argument for model")
    parser.add_argument("--batch_size", type=int, default=4096, required=False,
                        help="batch size argument for model")
    parser.add_argument("--n_envs", type=int, default=16, required=False,
                        help="N envs per rollout")
    parser.add_argument("--n_epochs", type=int, default=10, required=False,
                        help="n_epochs argument to model")
    parser.add_argument("--logdir", type=str, default="",
                        help="Tensorboard log directory")
    return parser.parse_args(argstr.split() if argstr is not None else None)


class ScaleObsTo01(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=env.observation_space.shape, dtype=np.float32)
    
    def observation(self, obs):
        obs = np.clip(obs.astype(np.float32) / 255.0, 0.0, 1.0)
        return np.transpose(obs, (2, 0, 1))


class FixedSeedEnv(gym.Wrapper):
    """
        First reset according to specified seed,
        further resets are stochastic
    """
    def __init__(self, env, seed):
        super().__init__(env)
        self._seed = seed
        self._first_rst = True

    def reset(self, **kwargs):
        if self._first_rst:
            if "seed" not in kwargs:
                kwargs["seed"] = self._seed
            self._first_rst = False
        return self.env.reset(**kwargs)


def get_callable_env(env_id: str, seed: Optional[int], wrap_atari=False):
    def _func():
        import gymnasium as gym  # re-import in subprocess
        env = gym.make(env_id)   # this triggers ALE registration internally
        if wrap_atari:
            from stable_baselines3.common.atari_wrappers import AtariWrapper
            env = AtariWrapper(env, terminal_on_life_loss=False)
        else:
            from gymnasium.wrappers import ClipReward
            env = ClipReward(env, -1.0, 1.0)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = FixedSeedEnv(env, seed)
        return env
    return _func


def get_env(env_id: str, n_envs: int=1, seed: int=None, n_stack: int=1,
            wrap_atari=False):
    """
        Get AtariWrapper, VecEnv, (optional) VecFrameStack of env_name
        Arguments:
            env_id: corresponds to env_id in gym.make
            n_envs: number of environments in parallel (for vec_env)
            seed: None -> random, or int seed
            n_stack: Argument to VecFrameStack i.e. how many frames to stack in obs
        Returns:
            tuple: (atari wrapped vec env, seed used)
    """
    if seed is None:
        seed = random.randint(0, 0xefffffff)
    env_fns = [get_callable_env(env_id, seed=seed+i, wrap_atari=wrap_atari) for i in range(n_envs)]
    env = SubprocVecEnv(env_fns)
    # env = VecNormalize(env, norm_obs=norm_obs, norm_reward=norm_reward)
    if n_stack > 1:
        env = VecFrameStack(env, n_stack=n_stack)
    print("Obs shape:", env.observation_space.shape)
    return env, seed


def get_cnn_env(env_id, n_envs, seed=None):
    """
        Calls get_env with n_stack=4
    """
    return get_env(env_id, n_envs, seed, n_stack=4, wrap_atari=True)


def get_mlp_env(env_id, n_envs, seed=None):
    return get_env(env_id, n_envs, seed, n_stack=4, wrap_atari=False)



def _get_policy(args):
    policy = "CnnPolicy" if args.cnn else "MlpPolicy"
    if args.algo == "rppo":
        policy = "CnnLstmPolicy" if args.cnn else "MlpLstmPolicy"
    return policy


def _get_dev(args):
    return "cuda" if args.cnn and torch.cuda.is_available() else "cpu"


def fix_mp_macos():
    mp.set_start_method("fork", force=True)


def get_model(argstr, env, seed=None):
    """
        Get PPO, TRPO, A2C models based on argstr, pass -h for list of args

        Returns (model, seed)
    """
    args = parse_mdl_args(argstr)
    args.algo = args.algo.lower()
    if seed is None:
        seed = random.randint(0, 0xefffffff)
    mdl = None
    _dict = None
    match args.algo:
        case "ppo":
            _dict = _get_ppo(args, env, seed)
            mdl = sb3.PPO(**_dict)
        case "trpo":
            _dict = _get_trpo(args, env, seed)
            mdl = sb3c.TRPO(**_dict)
    print(_dict)
    return mdl, seed
    

def _get_ppo(args, env, seed):
    lr = linear_schedule(args.lrstart, args.lrend) if args.lrsched else args.lr
    policy_kwargs = dict(
        net_arch=dict(
            pi=[args.fc1, args.fc2],
            vf=[args.fc1, args.fc2]
        ),
        normalize_images=args.cnn,
    )

    mdl_dict = dict(
        policy=_get_policy(args),
        device=_get_dev(args),
        env=env,
        learning_rate=lr,
        seed=seed,
        policy_kwargs=policy_kwargs,
        verbose=1,
        gamma=args.gamma,
        gae_lambda=args.gae,
        normalize_advantage=True,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        ent_coef=args.entcoef,
        clip_range=args.clip,
        vf_coef=args.vfcoef,
    )

    if args.kl:
        mdl_dict["target_kl"] = args.kltarg

    if args.logdir != "":
        mdl_dict["tensorboard_log"] = args.logdir
    return mdl_dict


def _get_trpo(args, env, seed):
    lr = linear_schedule(args.lrstart, args.lrend) if args.lrsched else args.lr
    policy_kwargs = dict(
        net_arch=[args.fc1, args.fc2],
        normalize_images=args.cnn,
    )

    mdl_dict = dict(
        policy=_get_policy(args),
        device=_get_dev(args),
        env=env,
        seed=seed,
        learning_rate=lr,
        policy_kwargs=policy_kwargs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae,
        normalize_advantage=True,
        target_kl=args.kltarg,
        verbose=1,
    )

    if args.logdir != "":
        mdl_dict["tensorboard_log"] = args.logdir

    return mdl_dict


if __name__ == "__main__":
    env, _ = get_cnn_env(ASTEROIDS_ENVID, 4, 1234)
    mdl, _ = get_model("--algo ppo --cnn --n_steps 1024 --batch_size 1024", env, 1234)
    mdl.learn(1e6)
