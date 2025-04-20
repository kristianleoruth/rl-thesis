import model
import optuna
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import argparse
import gc
import cProfile
import pstats
import io
import torch
from torch.distributions import Categorical as TorchCategorical
import traceback

class FastCategorical(TorchCategorical):
    def __init__(self, *args, **kwargs):
        kwargs['validate_args'] = False
        super().__init__(*args, **kwargs)

torch.distributions.Categorical = FastCategorical

class TrialEvalCallback(BaseCallback):
    def __init__(self, trial, eval_env, prune_after=100_000, eval_freq=15000,
                 n_eval_episodes=25, verbose=0, n_envs=1):
        super().__init__(verbose)
        self.trial = trial
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.step = 0
        self.prune_after = prune_after
        self.n_envs = n_envs

    def _on_step(self):
        self.step += self.n_envs
        if self.step % self.eval_freq != 0:
            return True

        mean_reward, _ = evaluate_policy(
            self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, return_episode_rewards=False
        )

        self.trial.report(mean_reward, self.step)

        if self.trial.should_prune() and self.prune_after <= self.step:
            raise optuna.exceptions.TrialPruned()

        return True


def build_argstr(trial: optuna.Trial, n_envs: int, algo: str, **kwargs):
    """
        kwargs:
            cnn=True, passes --cnn flag
            for ppo, can pass kl=True to tune kl target
    """
    cmd = f"--algo {algo} --fc1 512 --fc2 512"
    cnn = False
    if "cnn" in kwargs.keys() and kwargs["cnn"]:
        cnn = True
        cmd += " --cnn"
        if "resnet" in kwargs.keys() and kwargs["resnet"]:
            cmd += " --resnet"
        
    match algo:
        case "ppo":
            # lr = trial.suggest_float("lr", 5e-5 if cnn else 5e-4, 3e-4 if cnn else 3e-3,
            #                      step=5e-5 if cnn else 5e-4)
            # gae_lambda = trial.suggest_float("gae_lambda", 0.93, 0.99, step=0.01)
            # gamma = trial.suggest_float("gamma", 0.95, 0.99, step=0.01)
            # vf_coef = trial.suggest_float("vf_coef", 0.2, 0.7, step=0.1)
            # ent_coef = trial.suggest_float("ent_coef", 0.005, 0.02, step=0.005)
            # n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
            # batch_size = trial.suggest_categorical("batch_size",
            #                                        [512, 1024, 2048, 4096])
            # if batch_size > n_steps * n_envs:
            #     raise optuna.exceptions.TrialPruned()
            # n_epochs = trial.suggest_int("n_epochs", 5, 25, step=5)
            lr = trial.suggest_categorical(
                "lr",
                [5e-5, 7e-5, 1e-4] if cnn
                else [1e-4, 5e-4, 1e-3])
            gamma = trial.suggest_categorical("gamma", [0.99, 0.995])
            gae_lambda = trial.suggest_categorical("gae_lambda", [0.95, 0.97])
            vf_coef = trial.suggest_categorical("vf_coef", [0.25, 0.5, 0.75])
            ent_coef = trial.suggest_categorical("ent_coef", [0.0, 0.01, 0.02])
            n_steps = trial.suggest_categorical("n_steps", [1024, 2048])
            batch_size = trial.suggest_categorical("batch_size",
                                                   [1024, 2048])
            if batch_size > n_steps * n_envs:
                raise optuna.exceptions.TrialPruned()
            n_epochs = trial.suggest_categorical("n_epochs", [5, 15])
            cmd += f" --lr {lr} --gae {gae_lambda} --gamma {gamma}"
            cmd += f" --vfcoef {vf_coef} --entcoef {ent_coef} --n_steps {n_steps}"
            cmd += f" --batch_size {batch_size} --n_epochs {n_epochs}"
            if "kl" in kwargs.keys() and kwargs["kl"]:
                kl_target = trial.suggest_categorical("kl_target", [0.01, 0.02, 0.03])
                cmd += f" --kl --kltarg {kl_target}"
            else:
                clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
                cmd += f" --clip {clip_range}"
            trial.set_user_attr("cfg", {
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "n_steps": n_steps,
                "batch_size": batch_size,
            })
        case "rppo":
            lr = trial.suggest_categorical(
                "lr",
                [5e-5, 7e-5, 1e-4] if cnn
                else [1e-4, 5e-4, 1e-3]
            )

            gamma = 0.99
            gae_lambda = 0.95
            # gae_lambda = trial.suggest_float("gae_lambda", 0.95, 0.97, step=0.01)
            # gamma = trial.suggest_float("gamma", 0.97, 0.99, step=0.01)
            vf_coef = trial.suggest_categorical("vf_coef", [0.25, 0.5, 0.75])
            ent_coef = trial.suggest_categorical("ent_coef", [0.0, 0.01, 0.02])
            n_steps = trial.suggest_categorical("n_steps", [256, 512])
            batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
            if batch_size > n_envs * n_steps:
                raise optuna.exceptions.TrialPruned()

            n_epochs = trial.suggest_categorical("n_epochs", [4, 6])

            cmd += f" --lr {lr} --gae {gae_lambda} --gamma {gamma}"
            cmd += f" --vfcoef {vf_coef} --entcoef {ent_coef} --n_steps {n_steps}"
            cmd += f" --batch_size {batch_size} --n_epochs {n_epochs}"

            if "kl" in kwargs.keys() and kwargs["kl"]:
                kl_target = trial.suggest_categorical("kl_target", [0.01, 0.02, 0.03])
                cmd += f" --kl --kltarg {kl_target}"
            else:
                clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
                cmd += f" --clip {clip_range}"
        case "a2c":
            lr = trial.suggest_categorical(
                "lr",
                [5e-5, 7e-5, 1e-4] if cnn
                else [1e-4, 5e-4, 7e-4, 1e-3]
            )
            gamma = trial.suggest_float("gamma", 0.97, 0.99, step=0.01)
            gae_lambda = trial.suggest_float("gae_lambda", 0.95, 0.98, step=0.01)
            vf_coef = trial.suggest_float("vf_coef", 0.2, 0.7, step=0.1)
            ent_coef = trial.suggest_float("ent_coef", 0.005, 0.02, step=0.005)
            n_steps = trial.suggest_categorical("n_steps", [5, 7, 10])
            max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.25, 0.5, 0.75])
            
            cmd += f" --lr {lr} --gae {gae_lambda} --gamma {gamma} --max_grad_norm {max_grad_norm}"
            cmd += f" --vfcoef {vf_coef} --entcoef {ent_coef} --n_steps {n_steps}"
        case "trpo":
            lr = trial.suggest_categorical(
                "lr",
                [5e-5, 7e-5, 1e-4] if cnn
                else [3e-4, 7e-4, 1e-3]
            )
            kl_target = trial.suggest_categorical("kl_target", [0.01, 0.02, 0.03])

            gamma = 0.99
            gae_lambda = 0.95
            n_steps = 2048
            batch_size = n_envs * n_steps
            cg_max_steps = trial.suggest_categorical("cg_max_steps", [10, 15, 20])
            cg_damping = trial.suggest_categorical("cg_damping", [0.05, 0.1, 0.2])
            line_search_max_iter = trial.suggest_categorical("line_search_max_iter", [5, 10, 15])
            n_critic_updates = trial.suggest_categorical("n_critic_updates", [5, 10, 15])
            trial.set_user_attr("cfg", {
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "n_steps": n_steps,
                "batch_size": batch_size,
            })
            cmd += f" --lr {lr} --gae {gae_lambda} --gamma {gamma} --n_critic_updates {n_critic_updates}"
            cmd += f" --batch_size {batch_size} --n_steps {n_steps} --kltarg {kl_target}"
            cmd += f" --cg_max {cg_max_steps} --cg_damp {cg_damping} --ls_max_iter {line_search_max_iter}"

    trial.set_user_attr("info", {
        "cmd_str": cmd,
    })
    return cmd


def objective(trial: optuna.Trial, algo: str, env_id: str,
              n_envs: int = 1, seed: int = None, **kwargs):
    """
        args:
            algo=ppo, trpo, a2c
            seed=seed to pass to model creator, env creator
            n_envs=number of parallelized envs to create
            env_id=env id
        kwargs:
            cnn=True, passes --cnn flag
            for ppo, can pass kl=True to tune kl target
            learn_ts=int, learning timesteps
            prune_after=int, start pruning after N steps
    """
    model_str = build_argstr(trial, n_envs, algo, **kwargs)
    env = None
    eval_env = None
    if "cnn" in kwargs.keys() and kwargs["cnn"]:
        env, _ = model.get_cnn_env(env_id, n_envs, seed, clip_reward=True)
        eval_env, _ = model.get_cnn_env(env_id, 1, seed, clip_reward=False)
    else:
        env, _ = model.get_mlp_env(env_id, n_envs, seed, clip_reward=True)
        eval_env, _ = model.get_mlp_env(env_id, 1, seed, clip_reward=False)

    if "learn_ts" not in kwargs.keys():
        kwargs["learn_ts"] = 1_000_000
    if "prune_after" not in kwargs.keys():
        kwargs["prune_after"] = int(kwargs["learn_ts"] / 10)

    mdl, _ = model.get_model(model_str, env, seed)

    callback = TrialEvalCallback(
        trial,
        eval_env,
        prune_after=kwargs["prune_after"],
        eval_freq=50_000, n_eval_episodes=15, verbose=0,
        n_envs=n_envs
    )
    try:
        mdl.learn(kwargs["learn_ts"], callback=callback)
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        trial.set_user_attr("fail_reason", str(e))
        raise

    mean_ret, std_ret = evaluate_policy(mdl, eval_env, n_eval_episodes=150,
                                        deterministic=False)
    print(f"\n[ObjectiveEvaluation] Mean reward: {mean_ret} | Std reward: {std_ret}\n")
    del mdl, eval_env, env
    gc.collect()
    return mean_ret


def opt_hyperparams(algo: str, env_id: str, n_envs: int = 1, seed: int = None,
                    study_name="tuning", store_dir: str = "hyperparam_studies/",
                    n_trials=50, **kwargs):
    """
        args:
            algo=ppo, trpo, a2c
            seed=seed to pass to model creator, env creator, TPEsampler
            n_envs=number of parallelized envs to create
            env_id=env id
            prune_after
        kwargs:
            cnn=True, passes --cnn flag
            for ppo, can pass kl=True to tune kl target
            learn_ts=int, learning timesteps
            prune_after=int, start pruning after N steps
            resnet=False, custom features extractor ResNet-18
    """
    study = optuna.create_study(direction="maximize", storage=f"sqlite:///{store_dir}.db",
                                load_if_exists=True, study_name=study_name,
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
                                sampler=optuna.samplers.TPESampler(multivariate=True, seed=seed))
    study.set_user_attr("cfg", {
        "study_name": study_name,
        "algo": algo,
        "seed": seed,
        "n_envs": n_envs,
        "n_trials": n_trials,
    })
    study.optimize(lambda trial: objective(trial, algo, env_id, n_envs, seed, **kwargs),
                   n_trials=n_trials)
    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True,
                        help="Algorithm to tune parameters for")
    parser.add_argument("--env_id", "--env", type=str, default=model.ASTEROIDS_ENVID,
                        help="Env id to load")
    parser.add_argument("--n_envs", type=int, default=8,
                        help="Number of environments to train on")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--cnn", action="store_true",
                        help="Use convolutional policy")
    parser.add_argument("--kl", action="store_true",
                        help="Tune KL target instead of clip range (PPO)")
    parser.add_argument("--learn_ts", type=int, default=int(1e6),
                        help="Timesteps to learn per trial")
    parser.add_argument("--study_name", "--name", type=str, default="tuning")
    parser.add_argument("--store_dir", "--dir", type=str, default="hyperparam_studies/",
                        help="(Relative path, do not use ./) directory to store study db, must end with /")
    parser.add_argument("--seed", type=int, default=1, help="Seed (passed to model creator, env creator)")
    parser.add_argument("--prune_after", type=int, default=50_000,
                        help="Start checking pruning after N steps")
    parser.add_argument("--resnet", action="store_true", help="Use custom feature extractor ResNet-18")
    args = parser.parse_args()

    ENABLE_PROFILING = False

    if ENABLE_PROFILING:
        prf = cProfile.Profile()
        try:
            prf.enable()
            opt_hyperparams(args.algo.lower(), args.env_id, args.n_envs, args.seed, args.study_name, args.store_dir,
                            args.n_trials, cnn=args.cnn, kl=args.kl, learn_ts=args.learn_ts, resnet=args.resnet)
        except KeyboardInterrupt:
            prf.disable()
            s = io.StringIO()
            stat = pstats.Stats(prf, stream=s).sort_stats("cumtime")
            stat.print_stats(20)
            print(s.getvalue())
            s.close()
            del s
    else:
        opt_hyperparams(args.algo.lower(), args.env_id, args.n_envs, args.seed, args.study_name, args.store_dir,
                        args.n_trials, cnn=args.cnn, kl=args.kl, learn_ts=args.learn_ts)
