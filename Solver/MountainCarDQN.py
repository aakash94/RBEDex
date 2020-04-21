# inspired from stable baseline's example which is in turn inspired from OpenAI's implementation

import itertools

import gym
import numpy as np
import tensorflow as tf

from sacred import Experiment
from tqdm import trange

import stable_baselines.common.tf_util as tf_utils
from stable_baselines import  deepq
from stable_baselines.common.buffers import ReplayBuffer
from stable_baselines.deepq.policies import FeedForwardPolicy

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           layers=[64],
                                           feature_extraction="mlp")



class MountainCarDQN:

    def __init__(self,
                 env_string = 'MountainCar-v0',
                 ep_start = 1.0,
                 mode_rbed = True,
                 ep_min = 0,
                 episode_count = 5000,
                 steps_to_move_in = -1,
                 target_increment = 1,
                 schedule_timesteps = 5000,
                 sacred_ex = None):

        self.env = gym.make(env_string)
        self.action_space = self.env.action_space
        self.ex = sacred_ex
        self.mode_rbed = mode_rbed
        self.episode_count = episode_count
        self.env_target = -110
        # assert isinstance(action_space, gym.spaces.discrete.Discrete), 'unsupported action space for now.'

        # for sacred logging
        self.VAL_REWARD = "REWARD"
        self.VAL_EPSILON = "EPSILON"
        self.VAL_AVG100 = "AVG100"
        self.VAL_SOLVEDAT = "SOLVEDAT"

        # options
        self.epsilon = ep_start  # probability of choosing a random action
        self.epsilon_min = ep_min

        # RBED exclusive
        self.reward_threshold = -200  # Keep track of reward target for the agent
        self.epsilon_max = ep_start

        # Set to target value.
        if steps_to_move_in > 0 :
            self.steps_to_move_in = steps_to_move_in
        else:
            self.steps_to_move_in = self.env_target - self.reward_threshold


        self.quanta = (self.epsilon_max - self.epsilon_min) / self.steps_to_move_in  # what value to move epsilon by
        self.target_increment = target_increment  # Howmuch to increment the target every time the agent meets it.

        self.schedule_timesteps = schedule_timesteps
        self.initial_p = ep_start


    def linear_decay(self, step):
        fraction = min(float(step) / self.schedule_timesteps, 1.0)
        self.epsilon = self.initial_p + fraction * (self.epsilon_min - self.initial_p)


    def rb_decay_epsilon(self, current_reward = 0):
        if current_reward > self.reward_threshold and self.epsilon > self.epsilon_min:
            self.reward_threshold += self.target_increment
            self.epsilon -= self.quanta
            if self.epsilon < 0:
                self.epsilon = 0


    def epsilon_decay(self, step = 0, current_reward = 0):
        if self.mode_rbed:
            self.rb_decay_epsilon(current_reward=current_reward)
        else:
            self.linear_decay(step=step)


    def exec(self):
        """
        Train a DQN agent on cartpole env
        :param args: (Parsed Arguments) the input arguments
        """
        with tf_utils.make_session(8) as sess:
            # Create the environment
            env = self.env
            # Create all the functions necessary to train the model
            act, train, update_target, _ = deepq.build_train(
                q_func=CustomPolicy,
                ob_space=env.observation_space,
                ac_space=env.action_space,
                optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
                sess=sess,
                double_q = False,
            )
            # Create the replay buffer
            replay_buffer = ReplayBuffer(50000)
            # Create the schedule for exploration starting from 1 (every action is random) down to
            # 0.02 (98% of actions are selected according to values predicted by the model).
            solved_yet = False
            is_solved = False
            steps_so_far = 0
            # Initialize the parameters and copy them to the target network.
            tf_utils.initialize()
            update_target()

            episode_rewards = [0.0]
            obs = env.reset()
            for i in trange(self.episode_count):

                step = 0
                done = False
                while not done:
                    step += 1
                    steps_so_far += 1

                    if not self.mode_rbed:
                        self.linear_decay(step=steps_so_far)
                    # Take action and update exploration to the newest value
                    action = act(obs[None], update_eps=self.epsilon)[0]
                    new_obs, rew, done, _ = env.step(action)
                    # Store transition in the replay buffer.
                    replay_buffer.add(obs, action, rew, new_obs, float(done))
                    obs = new_obs

                    episode_rewards[-1] += rew

                    if done:
                        obs = env.reset()

                        last_reward = episode_rewards[-1]

                        if self.mode_rbed:
                            self.rb_decay_epsilon(current_reward=last_reward)

                        if len(episode_rewards[-101:-1]) == 0:
                            mean_100ep_reward = sum(episode_rewards)/100
                        else:
                            mean_100ep_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                        # is_solved = step > 100 and mean_100ep_reward >= self.env_target

                        # log epsilon
                        self.ex.log_scalar(self.VAL_EPSILON, self.epsilon)

                        # log reward
                        self.ex.log_scalar(self.VAL_REWARD, last_reward)

                        # log average reward
                        self.ex.log_scalar(self.VAL_AVG100, mean_100ep_reward)

                        # log solved at
                        if mean_100ep_reward >= self.env_target and (not solved_yet):
                            solved_yet = True
                            self.ex.log_scalar(self.VAL_SOLVEDAT, i)

                        # For next run
                        episode_rewards.append(0)

                    # Do not train further once solved. Keeping consistent with the original scheme
                    if not solved_yet:
                        # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                        if steps_so_far > 1000:
                            obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                            train(
                                obses_t,
                                actions,
                                rewards,
                                obses_tp1,
                                dones,
                                np.ones_like(rewards))
                        # Update target network periodically.
                        if steps_so_far % 1000 == 0:
                            update_target()