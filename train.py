import model
import argparse
import warnings


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
                                          args.n_envs, args.seed, n_stack)
    else:
        env, env_seed = model.get_mlp_env(args.env_id,
                                          args.n_envs, args.seed, n_stack)

    mdl, mdl_seed = model.get_model(args, env, args.seed)

    print(f"Seeds used:\nModel: {mdl_seed}\nEnv: {env_seed}")

    try:
        mdl.learn(args.timesteps, tb_log_name=args.tb_log_name)
        mdl.save(args.save_to)
        print(f"Model saved {args.save_to}\n\n")
    except KeyboardInterrupt:
        key = input("Save model? [y/N]: ")
        if key.lower() == "y":
            mdl.save(args.save_to)
            print(f"Model saved {args.save_to}\n\n")
        raise