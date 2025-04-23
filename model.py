import argparse
import gymnasium as gym
import stable_baselines3 as sb3
import sb3_contrib as sb3c
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import ale_py
import random
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import torch.nn.functional as F
import multiprocessing as mp
import os
from typing import List, Optional, Union
import math
os.environ["ALE_DISABLE_LOG"] = "1"

ASTEROIDS_ENVID = "ALE/Asteroids-v5"
ASTEROIDS_RAM_ENVID = "ALE/Asteroids-ram-v5"
# gym.register(ASTEROIDS_ENVID, )
USE_CUSTOM_CNN = False

class ResNetFeatureExtractor(BaseFeaturesExtractor):
    """
    Pretrained ResNet-18 feature extractor (512 dim) using torchvision.models.ResNet18_Weights.DEFAULT
    """
    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space=observation_space, features_dim=512)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.normalize = T.Normalize(
            mean=models.ResNet18_Weights.DEFAULT.meta["mean"],
            std=models.ResNet18_Weights.DEFAULT.meta["std"]
        )

        for param in self.resnet.parameters():
            param.requires_grad = False
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observation is (3, H, W)
        x = observations.mean(dim=1)

        # Repeat to 3 channels: (B, 3, 84, 84)
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)

        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        x = torch.clamp(x, 0.0, 1.0)
        x = self.normalize(x)
        x = self.resnet(x)
        return x.view(x.size(0), -1)


def cosine_schedule(initial_value, min_lr=1e-5):
    def func(progress_remaining):
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (1 - progress_remaining)))
        return min_lr + (initial_value - min_lr) * cosine_decay
    return func


def linear_schedule(initial_lr: float, final_lr: float = 1e-5):
    def func(progress_remaining: float) -> float:
        return final_lr + (initial_lr - final_lr) * progress_remaining
    return func


def parse_mdl_args(argstr=None, return_parser=False, add_help=True):
    """
        If return_parser=True, returns the parser
        If argstr=None, parses command line args
        If argstr=str, parses argstr
    """
    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument("--algo", type=str, required=True, 
                        help="Algorithm (ppo, rppo, trpo, a2c)")
    parser.add_argument("--fc1", type=int, default=256, required=False,
                        help="Size of first fully connected layer")
    parser.add_argument("--fc2", type=int, default=128, required=False,
                        help="Size of second fully connected layer")
    parser.add_argument("--cnn", action="store_true", required=False,
                        help="Use convolutional (Cnn) policy")
    parser.add_argument("--resnet", action="store_true", required=False,
                        help="Use pretrained feature extractor (ResNet-18)")
    parser.add_argument("--lrsched", action="store_true", required=False,
                        help="Use scheduled learning rate (linear)")
    parser.add_argument("--lrcos", action="store_true", required=False,
                        help="Use scheduled learning rate (cosine)")
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
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help="max_grad_norm arg for a2c, trpo")
    parser.add_argument("--cg_max", type=int, default=15,
                        help="TRPO cg_max_steps")
    parser.add_argument("--cg_damp", type=float, default=0.1,
                        help="TRPO cg_damping")
    parser.add_argument("--ls_max_iter", type=int, default=10,
                        help="TRPO line_search_max_iter")
    parser.add_argument("--n_critic_updates", type=int, default=10,
                        help="TRPO n_critic_updates")
    if return_parser:
        return parser
    return parser.parse_args(argstr.split() if argstr is not None else None)


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


class ScaledReward(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.prev_obs = None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        reward /= 100.0

        # Reward shaping heuristic: pixel difference between observations
        # if self.prev_obs is not None:
        #     movement_bonus = float((obs != self.prev_obs).sum()) / obs.size
        #     reward += 0.01 * movement_bonus
        # self.prev_obs = obs
        return obs, reward, terminated, truncated, info


class SkipNFrames(gym.Wrapper):
    """
    Repeat action for N frames, return last observation and summed reward
    """
    def __init__(self, env, limit: int):
        super().__init__(env)
        self.limit = limit
    
    def step(self, action):
        tot_reward = 0
        for _ in range(self.limit - 1):
            obs, reward, terminated, truncated, info = self.env.step(action)
            tot_reward += reward
            if terminated or truncated:
                return obs, tot_reward, terminated, truncated, info
        obs, reward, terminated, truncated, info = self.env.step(action)
        tot_reward += reward
        return obs, tot_reward, terminated, truncated, info


def get_callable_env(env_id: str, seed: Optional[int], wrap_atari=False,
                     atari_frame_skip=4, clip_reward=True):
    def _func():
        import gymnasium as gym  # re-import in subprocess
        if wrap_atari:
            env = gym.make(env_id, render_mode="rgb_array")
            from stable_baselines3.common.atari_wrappers import AtariWrapper
            env = AtariWrapper(env, terminal_on_life_loss=False,
                               frame_skip=atari_frame_skip,
                               clip_reward=clip_reward)
        else:
            env = gym.make(env_id)
            if clip_reward:
                from gymnasium.wrappers import ClipReward
                env = ClipReward(env, -1.0, 1.0)
            env = SkipNFrames(env, atari_frame_skip)
            
        from stable_baselines3.common.monitor import Monitor
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        env = FixedSeedEnv(env, seed)
        env = Monitor(env)
        return env
    return _func


def get_env(env_id: str, n_envs: int = 1, seed: int = None, n_stack: int = 1,
            wrap_atari=False, disable_vec_env=False, clip_reward=True):
    """
        Get AtariWrapper, VecEnv, (optional) VecFrameStack
            of env_id (if image obs)
        Get ClipReward, SubprocVecEnv, (optional) VecFrameStack of env_id (if mlp obs)
        Arguments:
            env_id: corresponds to env_id in gym.make
            n_envs: number of environments in parallel (for vec_env)
            seed: None -> random, or int seed
            n_stack: Argument to VecFrameStack i.e. how many frames to stack in obs
            wrap_atari: whether to wrap with AtariWrapper (only for image obs)
            disable_vec_env: Do not wrap with SubprocVecEnv (overrides n_envs)
        Returns:
            tuple: (atari wrapped vec env, seed used)
    """
    if seed is None:
        seed = random.randint(0, 0xefffffff)
    if disable_vec_env:
        env = get_callable_env(env_id, seed=seed,
                               wrap_atari=wrap_atari, atari_frame_skip=4)()
        if n_stack > 1:
            env = gym.wrappers.FrameStackObservation(env, stack_size=n_stack)
        return env, seed

    env_fns = [get_callable_env(env_id, seed=seed+i,
                                wrap_atari=wrap_atari, clip_reward=clip_reward,
                                atari_frame_skip=4)
               for i in range(n_envs)]
    env = SubprocVecEnv(env_fns)
    # env = VecNormalize(env, norm_obs=norm_obs, norm_reward=norm_reward)
    if n_stack > 1:
        env = VecFrameStack(env, n_stack=n_stack)
    return env, seed


def get_cnn_env(env_id, n_envs, seed=None, n_stack=4, clip_reward=True):
    """
        Calls get_env with AtariWrapper env
    """
    return get_env(env_id, n_envs, seed, n_stack=n_stack, wrap_atari=True, clip_reward=clip_reward)


def get_mlp_env(env_id, n_envs, seed=None, n_stack=4, clip_reward=True):
    """
        Returns get_env
        No AtariWrapper, only clipped rwd
    """
    return get_env(env_id, n_envs, seed, n_stack=n_stack, wrap_atari=False, clip_reward=clip_reward)


def get_video_env(env_id, seed=None, n_stack=4,
                  save_to="./saved_models/videos/", filename="unnamed"):
    os.makedirs(save_to, exist_ok=True)
    env, seed = get_env(env_id, seed=seed, n_stack=n_stack, wrap_atari=True, disable_vec_env=True)
    env = gym.wrappers.RecordVideo(env, video_folder=save_to, name_prefix=filename)
    return env, seed


def run_episode_with_model(env, algo: str, model_path: str,
                           deterministic: bool = True):
    """
    Loads a model and runs one full episode in the given env.

    Args:
        env: Gym environment (should be already wrapped with RecordVideo)
        algo: Algorithm name (ppo, rppo, a2c, trpo)
        model_path: Path to the saved model .zip file
        deterministic: Whether to use deterministic actions (default True)
    """
    algo = algo.lower()

    if algo == "ppo":
        model = sb3.PPO.load(model_path)
    elif algo == "a2c":
        model = sb3.A2C.load(model_path)
    elif algo == "trpo":
        model = sb3c.TRPO.load(model_path)
    elif algo == "rppo":
        model = sb3c.RecurrentPPO.load(model_path)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    obs, _ = env.reset()
    obs = obs.squeeze(-1).transpose(1, 2, 0)
    state = None
    done = False

    while not done:
        if algo == "rppo":
            action, state = model.predict(obs, state=state, deterministic=deterministic)
        else:
            action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, _ = env.step(action)
        obs = obs.squeeze(-1).transpose(1, 2, 0)
        done = terminated or truncated
    env.close()


def _get_policy(args):
    policy = "CnnPolicy" if args.cnn else "MlpPolicy"
    if args.algo == "rppo":
        policy = "CnnLstmPolicy" if args.cnn else "MlpLstmPolicy"
    return policy


def _get_dev(args):
    return "cuda" if args.cnn and torch.cuda.is_available() else "cpu"


def fix_mp_macos():
    mp.set_start_method("fork", force=True)


def get_model(args, env, seed=None):
    """
        Get PPO, TRPO, A2C models based on args,
        pass -h for list of args, or pass argparse.Namespace obj

        Get parsed args from parse_mdl_args()

        Returns (model, seed)
    """
    if isinstance(args, str):
        args = parse_mdl_args(args)
    elif not isinstance(args, argparse.Namespace):
        raise TypeError("Expected string or argparse.Namespace for args")

    args.algo = args.algo.lower()
    if seed is None:
        seed = random.randint(0, 0xefffffff)

    if os.path.exists(args.save_to + ".zip"):
        print(f"Model file found; Loading model at {args.save_to}")
        match args.algo:
            case "ppo":
                return sb3.PPO.load(args.save_to), seed
            case "trpo":
                return sb3c.TRPO.load(args.save_to), seed
            case "rppo":
                return sb3c.RecurrentPPO.load(args.save_to), seed

    mdl = None
    _dict = None
    match args.algo:
        case "ppo":
            _dict = _get_ppo(args, env, seed)
            mdl = sb3.PPO(**_dict)
        case "trpo":
            _dict = _get_trpo(args, env, seed)
            mdl = sb3c.TRPO(**_dict)
        case "a2c":
            _dict = _get_a2c(args, env, seed)
            mdl = sb3.A2C(**_dict)
        case "rppo":
            _dict = _get_rppo(args, env, seed)
            mdl = sb3c.RecurrentPPO(**_dict)
    print(_dict)
    return mdl, seed


def _get_lr(args):
    if args.lrcos:
        return cosine_schedule(args.lr)
    if args.lrsched:
        return linear_schedule(args.lrstart, args.lrend)
    return args.lr


def _get_ppo(args, env, seed):
    lr = _get_lr(args)
    policy_kwargs = dict(
        net_arch=dict(
            pi=[args.fc1, args.fc2],
            vf=[args.fc1, args.fc2]
        ),
        normalize_images=args.cnn,
    )

    if args.cnn and args.resnet:
        policy_kwargs["features_extractor_class"] = ResNetFeatureExtractor

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
    lr = _get_lr(args)
    policy_kwargs = dict(
        net_arch=[args.fc1, args.fc2],
        normalize_images=args.cnn,
    )

    if args.cnn and args.resnet:
        policy_kwargs["features_extractor_class"] = ResNetFeatureExtractor

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
        cg_max_steps=args.cg_max,
        cg_damping=args.cg_damp,
        line_search_max_iter=args.ls_max_iter,
        n_critic_updates=args.n_critic_updates,
    )

    if args.logdir != "":
        mdl_dict["tensorboard_log"] = args.logdir

    return mdl_dict


def _get_rppo(args, env, seed):
    mdl_dict = _get_ppo(args, env, seed)
    # can add specific RPPO args
    return mdl_dict


def _get_a2c(args, env, seed):
    lr = _get_lr(args)
    policy_kwargs = dict(
        net_arch=dict(
            pi=[args.fc1, args.fc2],
            vf=[args.fc1, args.fc2]
        ),
        normalize_images=args.cnn,
        # Stabilizes according to SB3 A2C docs
        optimizer_class=RMSpropTFLike,
        optimizer_kwargs=dict(eps=1e-5)
    )

    if args.cnn and args.resnet:
        policy_kwargs["features_extractor_class"] = ResNetFeatureExtractor

    mdl_dict = dict(
        policy=_get_policy(args),
        device=_get_dev(args),
        env=env,
        seed=seed,
        learning_rate=lr,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=args.n_steps,
        normalize_advantage=True,
        gamma=args.gamma,
        gae_lambda=args.gae,
        ent_coef=args.entcoef,
        vf_coef=args.vfcoef,
        max_grad_norm=args.max_grad_norm,
    )

    if args.logdir != "":
        mdl_dict["tensorboard_log"] = args.logdir
    return mdl_dict


if __name__ == "__main__":
    env, _ = get_cnn_env(ASTEROIDS_ENVID, 4, 1234)
    mdl, _ = get_model("--algo ppo --cnn --n_steps 1024 --batch_size 1024", env, 1234)
    mdl.learn(1e6)
