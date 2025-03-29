import model
import argparse
import warnings
from stable_baselines3.common.callbacks import BaseCallback
import os


class EvalAndSaveCallback(BaseCallback):
    def __init__(self, check_freq=150_000, name="tmp", save_dir="./saved_models", verbose: int = 1,
                 start_saving_after_ts=5_000_000):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.name = name
        self.best_mean_reward = 0
        self.last_saved_path = None
        self.save_dir = os.path.join(save_dir, "tmp")
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq != 0:
            return True
        reward = self.logger.name_to_value.get("rollout/ep_rew_mean")
        if reward is None:
            return True

        if reward > self.best_mean_reward:
            self.best_mean_reward = reward
            save_path = os.path.join(
                self.save_dir,
                f"{self.name}_{self.num_timesteps}_rew{reward:.1f}.zip"
            )
            self.model.save(save_path)
            if self.last_saved_path and os.path.exists(self.last_saved_path):
                os.remove(self.last_saved_path)
                if self.verbose:
                    print(f"Deleted previous checkpoint: {self.last_saved_path}")
            self.last_saved_path = save_path
            if self.verbose:
                print(f"New best mean reward: {reward:.2f} — model saved to {save_path}")

        return True


if __name__ == "__main__":
    mdl_parser = model.parse_mdl_args(return_parser=True, add_help=False)

    train_parser = argparse.ArgumentParser(add_help=False)
    train_parser.add_argument("--tb_log_name", type=str, default="",
                              help="Tensorboard log name")
    train_parser.add_argument("--env_id", type=str,
                              default=model.ASTEROIDS_ENVID,
                              help="Env id for training")
    train_parser.add_argument("--seed", type=int, default=1234,
                              help="Seed for model, env creation")
    train_parser.add_argument("--timesteps", type=int, default=int(10e6),
                              help="Training timesteps")
    train_parser.add_argument("--save_to", type=str,
                              default="./saved_models/model",
                              help="Where to save model (def ./saved_models/model)")
    parser = argparse.ArgumentParser(parents=[mdl_parser, train_parser])
    args = parser.parse_args()
    env = None
    env_seed = None

    if args.logdir == "" and args.tb_log_name != "":
        warnings.warn("tb_log_name ignored — make sure to pass --logdir as well.\n")
    if args.logdir != "" and args.tb_log_name == "":
        warnings.warn("logdir is set, but tb_log_name is empty — logs will be written directly to the logdir.\n")

    n_stack = 4
    if args.algo.lower() == "rppo":
        n_stack = 1
    if args.cnn:
        env, env_seed = model.get_cnn_env(args.env_id,
                                          args.n_envs,
                                          args.seed,
                                          n_stack,
                                          clip_reward=False)
    else:
        env, env_seed = model.get_mlp_env(args.env_id,
                                          args.n_envs,
                                          args.seed,
                                          n_stack,
                                          clip_reward=False)

    mdl, mdl_seed = model.get_model(args, env, args.seed)

    print(f"Seeds used:\nModel: {mdl_seed}\nEnv: {env_seed}")

    try:
        callback = EvalAndSaveCallback(name=args.tb_log_name or "unnamed", save_dir=os.path.dirname(args.save_to))
        mdl.learn(args.timesteps, tb_log_name=args.tb_log_name, callback=callback)
        mdl.save(args.save_to)
        print(f"Model saved {args.save_to}\n\n")
    except KeyboardInterrupt:
        key = input("Save model? [y/N]: ")
        if key.lower() == "y":
            mdl.save(args.save_to)
            print(f"Model saved {args.save_to}\n\n")
        raise