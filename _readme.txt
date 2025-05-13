[section]
[header][label=Hyperparameter Optimization and Training of PPO, LSTM-PPO, and TRPO in Atari Asteroids]
This research was conducted as part of my degree project for the program Information and Communication Technology at KTH Royal Institute of Technology. The research focuses on the question \bold{how do PPO, LSTM-PPO, and TRPO perform in complex environments with delayed rewards?} The project entailed writing two scripts, \textcode{tune.py} and \textcode{train.py}, which allow for hyperparameter optimization and model training, respectively.

[subheader][label=How to use the scripts]
The arguments \textcode{-h, --help} may be used to view the options associated with each file. 
[code][
 % python tune.py -h

usage: tune.py [-h] --algo ALGO [--env_id ENV_ID] [--n_envs N_ENVS] [--n_trials N_TRIALS] [--cnn] [--kl] [--learn_ts LEARN_TS] [--study_name STUDY_NAME] [--store_dir STORE_DIR] [--seed SEED]
               [--prune_after PRUNE_AFTER]

options:
  -h, --help            show this help message and exit
  --algo ALGO           Algorithm to tune parameters for
  --env_id ENV_ID, --env ENV_ID
                        Env id to load
  --n_envs N_ENVS       Number of environments to train on
  --n_trials N_TRIALS
  --cnn                 Use convolutional policy
  --kl                  Tune KL target instead of clip range (PPO)
  --learn_ts LEARN_TS   Timesteps to learn per trial
  --study_name STUDY_NAME, --name STUDY_NAME
  --store_dir STORE_DIR, --dir STORE_DIR
                        (Relative path, do not use ./) directory to store study db, must end with /
  --seed SEED           Seed (passed to model creator, env creator)
  --prune_after PRUNE_AFTER
                        Start checking pruning after N steps
]

For example, to run a PPO study of 50 trials on Atari Asteroids, with a convolutional policy, and logging to Optuna Dashboard:

[code][
python tune.py --algo ppo --env ALE/Asteroids-v5 --n_trials 50 --cnn --study_name ppo_study --store_dir studies/
]