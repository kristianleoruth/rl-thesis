[section]
[header][label=How do PPO, LSTM-PPO, and TRPO perform in Atari Asteroids?]
This project was part of my degree project for the program Information and Communication Technology (tcomk) at the KTH Royal Institute of Technology. The research focuses on the titular research question: \bold{How do PPO, LSTM-PPO, and TRPO perform in Atari Asteroids?} This page documents the performance of the three reinforcement learning (RL) algorithms in the Atari game Asteroids (1979), which is a benchmark title for RL and poses several challenges, most importantly delayed rewards, navigation, and long-term planning.

\tableofcontents

[header][label=Background, uid=background]
[subheader][label=Atari Asteroids, uid=env_expl]
Asteroids (1979) is a game featuring the player controlling a ship in a field of asteroids. The asteroids are slow moving projectiles which, upon coming into contact with the player, cause the loss of a life. The player can accelerate forward, turn clockwise and anti-clockwise, as well as shoot a fast moving projectile which can destroy asteroids. Asteroids, upon being shot by a projectile, split into smaller pieces which can be further split or destroyed if small enough. These actions gain scores of 20, 50, and 100 in order of decreasing size.

[section][notopmarg=True, align=center]
[img][
    src=assets/atari-asteroids.jpg,
    maxwidth=30%,
    caption=In-game screenshot, running using the Arcade Learning Environment.
]
[section][notopmarg=True]

[section][align=justify]
[subheader][label=Algorithms, uid=algos]
[subsubheader][label=Markov Decision Processes, uid=mdp_expl]
Markov Decision Processes (MDPs) describe interactions with an environment using a set of states S, actions A, reward function R(s,a) which maps a state and action to the real numbers. MDPs must satisfy the Markov property, that the next state is only dependent on the previous state and action, thus state transitions are independent of each other. Return \textcode{G_t} is the sum of rewards from t to T (T = final time step). This quantifies the performance of the model.

[subsubheader][label=Value Functions and Advantage Functions]
Value functions measure the quality of states and actions under a policy π (which gives the probability of an action given a state). We get two value functions, \bold{state-value} and \bold{action-value functions}, 
[code][
v_pi(s) = Sum over actions a (pi(a|s) * q_pi(s, a)),

q_pi(s, a) = Sum over next states s', rewards r (p(s', r | s, a)(r + gamma * v_pi(s'))).
]

Advantage functions differ from value functions by calculating the value of actions relative to the value of the state,

[code][
A_pi(s, a) = q_pi(s, a) - v_pi(s).
]

[subsubheader][label=Trust Region Policy Optimization, uid=trpo_expl]
Trust Region Policy Optimization (TRPO) is a policy gradient method which takes constrained updates on the policy based on the KL divergence of the policy \textcode{pi_theta} (with parameters theta) in relation to the old policy, \textcode{pi_theta_k}. TRPO is a stochastic on-policy method, which samples actions from a probability distribution given by the policy. The algorithm also makes use of an objective function, which uses the previously defined advantage function to relate the performance of the new policy with the old:

[code][
L(theta_k, theta) = Expected value ((pi_theta(a|s) / pi_theta_k(a|s)) * A_pi_theta_k(s,a)) | s, a ~ pi_theta_k,
]

which gives the policy update

[code][
theta_(k+1) = argmax_theta L(theta_k, theta), s.t. D_KL(theta||theta_k) <= delta.
]

Here \textcode{theta_(k+1)} are the new parameters, taken from the parameters with the highest expected value calculated earlier. In this way, the step is guaranteed to improve the policy, as the argmax maximizes the probability of actions which have greatest advantage over others in the given state s. TRPO approximates both equations to simplify calculations. The error introduced by approximation is combated by means of a line search that determines the step size (such that the KL divergence constraint is satisfied). Further reading: \link{https://spinningup.openai.com/en/latest/algorithms/trpo.html}{OpenAI's Spinning Up: TRPO}, \link{https://arxiv.org/abs/1502.05477}{Trust Region Policy Optimization paper}.

[subsubheader][label=Proximal Policy Optimization, uid=ppo_expl]
Proximal Policy Optimization (PPO) was developed as a simpler and more efficient alternative to TRPO. Two versions exist which differ in their usage of the KL-constraint used in TRPO. PPO traditionally clips the policy update with a constant constraint (around 0.1 − 0.3), a simple and fast approximation of the KL-constraint. Alternatively, instead of subjecting the policy update to a constraint, a KL penalty is added to the objective function. This discourages large updates to the policy. In this project, the KL penalty method was not used, so all testing is done with clip range as the update constraint. This also applies to LSTM-PPO. Further reading: \link{https://spinningup.openai.com/en/latest/algorithms/ppo.html}{OpenAI's Spinning Up: PPO}, \link{https://arxiv.org/abs/1707.06347}{Proximal Policy Optimization algorithms paper}.

[subsubheader][label=Long Short-Term Memory, uid=rppo_expl]
Recurrent Neural Networks (RNNs) work on input data in a time series format; therefore they are characterized by their output being dependent on previous data in the time series. This is done by maintaining a hidden internal state at each time step. The architecture is similar to standard 3-layer MLPs, but hidden perceptrons can either have self-feedback or additional context cells, in addition to the input data. Long short-term memory (LSTM) is a solution to the vanishing error problem by enforcing constant error throughout backpropagataion. Typically, reinforcement learning assumes independence between time steps. As a result, delayed rewards can be misattributed to actions taken during the reward frame, not the actions which led to the reward originally.

[section]
[header][label=Method, uid=method_expl]
[subheader][label=Human Normalized Score, uid=hns_expl]
In \link{https://www.nature.com/articles/nature14236}{Human-level control through deep reinforcement learning}, we find a groundbreaking RL study on a general DQN agent for Atari environments. This paper sets the precedent for Atari benchmarks in RL, and from it we find that the DQN struggled to match human expert gameplay. From this paper and testing, the scores that were achieved by a random agent, DQN agent, and human expert were collected.

[subsubheader][label=Benchmark Scores, uid=benchmarks]
[list][
-) Random: 455 (+-239)
-) Human: 13157
-) DQN: 1629 (+-542) | 9.4% HNS
]

HNS denotes the Human Normalized Score (specified in the Nature paper), which is calculated as the quotient of the agent and human scores with the random baseline subtracted from both:
[code][class=dummy][
HNS = (score - random_score) / (human_score - random_score)
]

This represents how much of the gap between random and human scores has been covered. So the DQN agent has trouble closing 10% of that difference, which is clearly not a good achievemnt, especially compared with the performance achieved in other games (Pinball: 2539%, Breakout: 1327%, etc.). This research aims to find whether PPO, TRPO, or LSTM-PPO can achieve better scores and to investigate their learning behavior. 

[subheader][label=Libraries and Frameworks (Implementation), uid=libs_fws]
All programming was done in Python 3.11.11. The models were created using \link{https://stable-baselines3.readthedocs.io/en/master/}{Stable Baselines 3} (sb3) and \link{https://sb3-contrib.readthedocs.io/en/master/}{SB3-Contrib} (sb3c). These platforms allow for quick and easy model creation and modification, with direct access to hyperparameters, network architecture, as well as support for TensorBoard logging. The hyperparameter optimization was done with \link{https://optuna-dashboard.readthedocs.io/en/latest/}{Optuna}, a library for creating optimization studies. This allowed me to customize the hyperparameter search space, sampling method, pruning strategy, as well as logged everything to Optuna Dashboard, a tool for visualizing trial performance, overall study, and evaluation of hyperparameter importance.

[subheader][label=Environment Preprocessing, uid=preproc_method]
[section][notopmarg=True]
[column][notopmarg=True]
[subsubheader][label=Convolutional Policies]
[list][align=left][
*)Dimensionality reduction: 210x160 RGB image is converted to 84x84 grayscale observation according to Nature paper (Mnih et al. 2015) and sb3 docs
*)Frame skipping and maxing last two frames: common Atari preprocessing
*)Observation stacking: last four observations are returned rather than just most recent
*)Reward clipping (sign based)
*)Sticky actions: introduce stochasticity and emulate input latency
]
[column][notopmarg=True]
[subsubheader][label=MLP Policies]
[list][align=left][
*)No dimensionality reduction (input is memory state of program)
*)Frame skipping: 4 frames
*)Observation stacking: last four observations are returned
*)Reward clipping (sign based)
*)Sticky actions: enabled
]
[section]
[subheader][label=Hyperparameter Optimization, uid=hypopt_method]
As explained, an Optuna study was created for each algorithm. These would run trials optimizing the score from evaluation. This evaluation is done using sb3's \textcode{evaluate_policy} method. Each study was conducted using the seed 1234, which was used to create the model and environment. The parameter space reduced dramatically through using discrete subset in reasonable hyperparameter value ranges (e.g. learning rate options for PPO—5e-5, 7e-5, 1e-4). 100 trials were performed per study, limited by time constraints; however, this is sufficient for a coarse search of the limited parameter space. Each trial lasted 1M time steps, though LSTM-PPO was trained on 300K because of slow performance. Pruning was done by using Optuna's \textcode{MedianPruner}. Both value and policy networks use architecture of two fully connected hidden layers, 512 neurons each. 

The procedure for running every study was as follows:
[list][align=left][
#) Create training and evaluation environments using given \textcode{seed_i}
#) Set study objective to maximize score returned from evaluation method (from 15 episodes)
#) Every trial:
##) Sample hyperparameters using Optuna's \textcode{TPESampler}, with \textcode{multivariate=True}—assuming non-independence between hyperparameters
##) Create and initialize model with \textcode{seed_i} and sampled hyperparameteres
##) Train for N time steps, on interval running evaluations and reporting to the \textcode{MedianPruner}
##) Upon trial completion, run final evaluation on 150 episodes
#) Store top trial hyperparameters
]

Tuning was done using the script \textcode{tune.py}, allowing customization of trial length, amount, as well as pruning aggression and logging of study progress. As a note to each trial, the argument string used to construct the model (using \textcode{model.py}) is stored. This allows for quick progression to training, as the same argument string can be used with the training script \textcode{train.py}.

Below is an example of using \textcode{tune.py}:
[code][
python tune.py --algo ppo --env ALE/Asteroids-v5 --n_trials 100 --cnn --learn_ts 1000000 --study_name ppo_cnn_clip_seed5678 --store_dir hyperparam_studies/ppo --n_envs 24 --prune_after 50000 --seed 5678
]

[subsubheader][label=Parameter search spaces (PPO; LSTM-PPO; TRPO), uid=tuning_vals]
Parameter search spaces were reduced significantly by choosing values within normal ranges to test. This avoids the trouble of continuous search spaces, and limits sampling to discrete categories. As a result, these parameter values amount to about 2K-5K combinations, which can be efficiently (but coarsely) searched within 100 trials. Below are listed the parameters and their possible values.

\bold{PPO}
[list][
#) Learning rate: 5e-5, 7e-5, 1e-4 (CNN), or 1e-4, 5e-4, 1e-3 (MLP)
#) Gamma: 0.99, 0.995
#) GAE lambda: 0.95, 0.97
#) Value function coefficient: 0.25, 0.5, 0.75
#) Entropy coefficient: 0.0, 0.01, 0.02
#) \textcode{N_steps}: 1024, 2048
#) Batch size: 1024, 2048
#) \textcode{N_epochs}: 5, 15
#) Clip range: 0.1, 0.2, 0.3
#) \textcode{N_envs}: 24, 4
]

\bold{LSTM-PPO}
[list][
#) Learning rate: 5e-5, 7e-5, 1e-4 (CNN), or 1e-4, 5e-4, 1e-3 (MLP)
#) Gamma: 0.99
#) GAE lambda: 0.95
#) Value function coefficienct: 0.25, 0.5, 0.75
#) Entropy coefficient: 0.0, 0.01, 0.02
#) \textcode{N_steps}: 1024, 2048
#) Batch size: 512, 1024
#) \textcode{N_epochs}: 4, 6
#) Clip range: 0.1, 0.2, 0.3
#) \textcode{N_envs}: 32, 4
]

\bold{TRPO}
[list][
#) Learning rate: 5e-5, 7e-5, 1e-4 (CNN), or 3e-4, 7e-4, 1e-3 (MLP)
#) KL target: 0.01, 0.02, 0.03
#) Gamma: 0.99, 0.995
#) GAE lambda: 0.95, 0.97
#) \textcode{N_steps}: 512, 2048
#) Batch size: \textcode{N_steps}, \textcode{N_envs*N_steps}
#) CG max steps: 10, 15, 20
#) CG damping: 0.05, 0.1, 0.2
#) Line search max iterations: 5, 10, 15
#) \textcode{N_critic_updates}: 5, 10, 15
#) \textcode{N_envs}: 1 (CUDA compatibility errors; forced), 4
]
[subheader][label=Training Procedure, uid=training_method]
The training script \textcode{train.py} allows the choice of model, corresponding hyperparameters, as well as customization of training through number of environments, changing environment, logging, etc. This is all done by passing command-line arguments to the program, as shown below:
[code][
python train.py --algo ppo --fc1 512 --fc2 512 --cnn --lr 0.0001 --gae 0.97 --gamma 0.995 --vfcoef 0.5 --entcoef 0.0 --n_steps 1024 --batch_size 2048 --n_epochs 15 --clip 0.3 --n_envs 24 --timesteps 100000000 --lrcos --env_id ALE/Asteroids-v5 --logdir ./tblogs --tb_log_name ppo_cnn_clip_1234 --save_to ./saved_models/ppo_cnn_clip_1234 --seed 1234
]
[section]
[header][label=Results: Hyperparameter Optimization, uid=results]
[subheader][label=PPO Tuning, uid=ppo_tuning]

[subsubheader][label=Best PPO parameters per policy (MLP; CNN), uid=ppo_params]
[list][
-)Learning rate: 1e-4 (MLP), 1e-4 (CNN)
-)Gamma: 0.995 (Same for both)
-)GAE lambda: 0.97
-)Value coeff: 0.25, 0.5
-)Entropy coeff: 0.0
-)\textcode{N_steps}: 1024
-)Batch size: 1024, 2048
-)\textcode{N_epochs}: 15
-)Clip range: 0.2, 0.3
-)SCORE: \bold{1068}, 859
]

The above list shows the PPO hyperparameter values used by each policy, as well as the mean score of the best trial from 150 episodes. The same evaluation setup is used in all trials. As seen, the MLP policy outperformed the CNN counterpart by 24.4%. They achieved mean scores of 1068 and 859, respectively. However this is not necessarily a problem for CNN policies as it is a given that the learning progress is slower; Evaluated over only 1M steps of learning does not entirely reflect the quality of the final, trained models.

[section][align=center]
[img][
    src=./assets/ppo/overview.png,
    caption=Convolutional and MLP policy studies overview (PPO); the points are trial values and lines are the maximum trial value achieved in the study.,
    maxwidth=70%
]

[img][
    src=./assets/ppo/edf.png,
    caption=Empirical Distribution Function (EDF) of the two studies; depicts the probability distribution of the trial values,
    maxwidth=70%
]

[section][notopmarg=True]
MLP policies clearly outperformed CNN policies, and the above figure shows this. The lines record the best trial value, and the points are individual trial values, corresponding to their x-axis value. It is also apparent from the figure that MLP trials were more distributed in value, while the CNN trials are split, with few trials scoring in the 300-450 range. This could also be exacerbated by the pruning, which is enabled after 50k steps. The empirical distribution functions (EDFs) of MLP and CNN policies are plotted above. They confirm that CNN policies are generally separated in two groups at both extremes of performance, while MLP policies spanned the range of scores up to 1000 evenly. Furthermore, the CNN trials are shifted to the left in relation to the MLP trials, which shows their consistently worse performance.

[subsubheader][label=PPO CNN trials, uid=ppo_cnn_trials]
[section][notopmarg=True]
[column][notopmarg=True]
[img][
    src=./assets/ppo/intermediates-cnn.png,
    caption=Intermediate trial values (CNN) taken at intervals by evaluation,
    maxwidth=100%
]
[column][notopmarg=True]
[img][
    src=./assets/ppo/ped-anova-cnn.png,
    caption=PED-ANOVA hyperparameter importance; higher values are more important relative to others,
    maxwidth=100%
]
[section]
[subsubheader][label=PPO MLP trials, uid=ppo_mlp_trials]
[section][notopmarg=True]
[column][notopmarg=True]
[img][
    src=./assets/ppo/intermediates-mlp.png,
    caption=Intermediate trial values (MLP) taken at intervals by evaluation,
    maxwidth=100%
]
[column][notopmarg=True]
[img][
    src=./assets/ppo/ped-anova-mlp.png,
    caption=PED-ANOVA hyperparameter importance; higher values are more important relative to others,
    maxwidth=100%
]

[section]
[subheader][label=LSTM-PPO Tuning, uid=rppo_tuning]

[subsubheader][label=Best LSTM-PPO parameters per policy (MLP; CNN), uid=rppo_params]
[list][
-)Learning rate: 5e-4 , 7e-45
-)Gamma: 0.99
-)GAE lambda: 0.95
-)Value coeff: 0.25, 0.75
-)Entropy coeff: 0.0
-)\textcode{N_steps}: 1024
-)Batch size: 1024, 512
-)\textcode{N_epochs}: 6
-)Clip range: 0.2, 0.3
-)SCORE: \bold{622}, 549
]

Above we find the best LSTM-PPO hyperparameters (identical to PPO). As with PPO, the best trial's mean episode score achieved in evaluation is higher for the MLP policy (622) versus the CNN policy (549), which is 13.3% better. It is also clear that LSTM-PPO did not have enough time to properly develop policies within the 300K steps. As a result the scores are noticeably poorer than PPO. 

[section][align=center]
[img][
    src=./assets/rppo/overview.png,
    caption=Convolutional and MLP policy studies overview (LSTM-PPO); the points are trial values and lines are the maximum trial value achieved in the study.,
    maxwidth=70%
]

[img][
    src=./assets/rppo/edf.png,
    caption=Empirical Distribution Function (EDF) of the two studies; depicts the probability distribution of the trial values,
    maxwidth=70%
]

[section][notopmarg=True]
The study overview shows individual trial values plotted against the trial number. As opposed to the PPO case, MLP and CNN policies were distributed similarly, except that MLP outperformed CNN across the trials. The EDFs of the two studies show a similar distribution between MLP and CNN, and that the CNN curve is shifted up and to the right compared to the MLP curve. Both have large gaps in trial performance, with each curve having a flat region (ranges \[180, 400\] for CNN, \[300, 450\] for MLP).

[subsubheader][label=LSTM-PPO CNN trials, uid=rppo_cnn_trials]
[section][notopmarg=True]
[column][notopmarg=True]
[img][
    src=./assets/rppo/fanova-cnn.png,
    caption=fANOVA hyperparameter importance (Intermediate values missing due to error),
    maxwidth=100%
]
[column][notopmarg=True]
[img][
    src=./assets/rppo/ped-anova-cnn.png,
    caption=PED-ANOVA hyperparameter importance,
    maxwidth=100%
]

[section]
[subsubheader][label=LSTM-PPO MLP trials, uid=rppo_mlp_trials]
[section][notopmarg=True]
[column][notopmarg=True]
[img][
    src=./assets/rppo/intermediates-mlp.png,
    caption=Intermediate trial values (MLP) taken at intervals by evaluation,
    maxwidth=100%
]
[column][notopmarg=True]
[img][
    src=./assets/rppo/ped-anova-mlp.png,
    caption=PED-ANOVA hyperparameter importance,
    maxwidth=100%
]
[section]
[subheader][label=TRPO Tuning, uid=trpo_tuning]

[subsubheader][label=Best TRPO parameters per policy (MLP; CNN), uid=trpo_params]
[list][
-)KL target: 0.03, 0.01
-)Learning rate: 1e-4, 1e-4
-)Gamma: 0.99
-)GAE lambda: 0.95
-)\textcode{N_steps}: 2048, 512
-)Batch size: 8192, 512
-)CG max steps: 15
-)CG damping: 0.05, 0.1
-)Line search max iterations: 10
-)\textcode{N_critic_updates}: 10, 15
-)SCORE: \bold{732}, 582
]

Again, the MLP policy outperformed its convolutional counterpart, this time by 25.7% with a score of 732 (CNN achieved 582). As with PPO, each trial was conducted over 1M time steps, which also seems to be too little to judge TRPO's final policy from full training. Perhaps the lower dimensionality of the MLP observations (just 512 bytes) allows patterns to be more quickly identified, resulting in faster early learning. Meanwhile, the convolutional policies must train the feature extraction network as well as the MLP network, and thus take longer to converge.

[section][align=center]
[img][
    src=./assets/trpo/overview.png,
    caption=Convolutional and MLP policy studies overview (TRPO); the points are trial values and lines are the maximum trial value achieved in the study.,
    maxwidth=70%
]

[img][
    src=./assets/trpo/edf.png,
    caption=Empirical Distribution Function (EDF) of the two studies; depicts the probability distribution of the trial values,
    maxwidth=70%
]

[section][notopmarg=True]
In the above figures, the MLP trials were generally distributed higher by about 200-400 points, which suggests a more consistent parameter space. The EDFs also show that CNN policies are consistenly worse performers, while MLP policies are distributed over 350-730, with a tighter and steeper EDF. A significantly higher amount of CNN trials appear to have been pruned, as they achieve scores worse than the random agent. This shows a greater instability in the parameter space, that there are fewer regions of high performance compared to MLP parameter space.

[subsubheader][label=TRPO CNN trials, uid=trpo_cnn_trials]
[section][notopmarg=True]
[column][notopmarg=True]
[img][
    src=./assets/trpo/intermediates-cnn.png,
    caption=Intermediate trial values (CNN) taken at intervals by evaluation,
    maxwidth=100%
]
[column][notopmarg=True]
[img][
    src=./assets/trpo/ped-anova-cnn.png,
    caption=PED-ANOVA hyperparameter importance; higher values are more important relative to others,
    maxwidth=100%
]
[section]
[subsubheader][label=TRPO MLP trials, uid=trpo_mlp_trials]
[section][notopmarg=True]
[column][notopmarg=True]
[img][
    src=./assets/trpo/intermediates-mlp.png,
    caption=Intermediate trial values (MLP) taken at intervals by evaluation,
    maxwidth=100%
]
[column][notopmarg=True]
[img][
    src=./assets/trpo/ped-anova-mlp.png,
    caption=PED-ANOVA hyperparameter importance; higher values are more important relative to others,
    maxwidth=100%
]

[section]
[header][label=Results: Training, uid=training_results]
[subheader][label=PPO Training, uid=ppo_training]
[section][notopmarg=True]
[column][notopmarg=True]
[img][
    src=./assets/ppo/eval-score-cnn.jpg,
    caption=Convolutional PPO agent scores in evaluations performed during training,
    maxwidth=100%
]
[img][
    src=./assets/ppo/eval-eplen-cnn.jpg,
    caption=Mean episode length of convolutional PPO agent in evaluations,
    maxwidth=100%
]
[img][
    src=./assets/ppo/returns-cnn.jpg,
    caption=Mean episode returns during training (agent receives these rewards which are clipped to the sign of the score for stability),
    maxwidth=100%
]
[column][notopmarg=True]
[img][
    src=./assets/ppo/eval-score-mlp.jpg,
    caption=MLP PPO agent scores in evaluations performed during training,
    maxwidth=100%
]
[img][
    src=./assets/ppo/eval-eplen-mlp.jpg,
    caption=Mean episode length of MLP PPO agent in evaluations,
    maxwidth=100%
]
[img][
    src=./assets/ppo/returns-mlp.jpg,
    caption=Mean episode returns during training,
    maxwidth=100%
]
[section]
From the figures, it seems clear that convolutional training is more susceptible to variation in seeds. As can be seen, one of the runs was consistently underperforming for most of the training, and did not succeed as the others. The training is also less stable, with significant dips in both returns and evaluated scores. This could be due to the high clip range (0.3), which may have been favored in trials for rapid early development, but not as suited for longer training runs; however this is not corroborated by the MLP case, which ended trials with a clip range of 0.2. The convolutional agent achieved a 9.4% HNS, which closely matches the DQN performance. In the Nature paper, the agent is only trained for 50M time steps, whereas these agents are trained on 100M.

[section][notopmarg=True, align=center]
[img][
    src=./assets/ppo/loss-mlp.jpg,
    caption=MLP policy training losses; loss vs. step,
    maxwidth=50%
]

[section]
The MLP agent has stable training, mostly monotonic with a few dips. All training runs are closely correlated, suggesting that seed choice is not an important factor. Possibly the hyperparameters chosen are more generalizable, or the RAM observations are more consistent/easy to understand. The MLP agent converges to a worse policy (in the range of 200-400 points lower than its convolutional counterpart). This could be due to 50M steps limiting the training, as the returns and scores curves suggest that the policy is not completely converged. The losses increase, which could be due to the greater magnitude of rewards seen as training progresses, which tracks with the rewards curve. Loss functions such as Mean Squared Error are sensitive to reward magnitudes. The agent achieves a 7.6% HNS, which is worse than the benchmark DQN agent (9.4%), and the convolutional agent (9.4%).

Both agents see very little increase in episode length, suggesting poor asteroid avoidance. Thus the score increases are probably the agent optimizing within a low-reward policy space, because it seems evident that longer episodes with worse 'aggression' or score-gathering policies would have more opportunities to score and would therefore achieve better results. A 250 frame episode is equivalent to 4.17 seconds of gameplay, which is not good.

[subheader][label=LSTM-PPO Training, uid=rppo_training]
[section][notopmarg=True]
[column][notopmarg=True]
[img][
    src=./assets/rppo/eval-score-cnn.jpg,
    caption=Convolutional LSTM-PPO agent scores in evaluations performed during training,
    maxwidth=100%
]
[img][
    src=./assets/rppo/eval-eplen-cnn.jpg,
    caption=Mean episode length of convolutional LSTM-PPO agent in evaluations,
    maxwidth=100%
]
[img][
    src=./assets/rppo/returns-cnn.jpg,
    caption=Mean episode returns during training (agent receives these rewards which are clipped to the sign of the score for stability),
    maxwidth=100%
]
[column][notopmarg=True]
[code][
[img][
    src=./assets/rppo/eval-score-mlp.jpg,
    caption=MLP LSTM-PPO agent scores in evaluations performed during training,
    maxwidth=100%
]
[img][
    src=./assets/rppo/eval-eplen-mlp.jpg,
    caption=Mean episode length of LSTM-MLP PPO agent in evaluations,
    maxwidth=100%
]
[img][
    src=./assets/rppo/returns-mlp.jpg,
    caption=Mean episode returns during training,
    maxwidth=100%
]
]