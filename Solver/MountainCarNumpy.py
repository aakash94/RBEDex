# taken from
# https://gist.github.com/gkhayes/3d154e0505e31d6367be22ed3da2e955#file-mountain_car-py-L64
import numpy as np
import gym
from tqdm import trange


class MountainCarNumpy():

    def __init__(self,
                 env_string='MountainCar-v0',
                 ep_start=1.0,
                 ep_decay_rate=0.98,
                 mode_rbed=True,
                 ep_min=0,
                 episode_count=50000,
                 steps_to_move_in=-1,
                 target_increment=1,
                 discount=0.9,
                 sacred_ex=None):

        # Import and initialize Mountain Car Environment
        self.env = gym.make(env_string)
        self.env.reset()
        self.ex = sacred_ex
        self.mode_rbed = mode_rbed
        self.episode_count = episode_count
        self.env_target = -110
        self.steps_to_move_in = steps_to_move_in
        self.discount = discount

        self.quanta = 0
        self.target_increment = 0

        # for sacred logging
        self.VAL_REWARD = "REWARD"
        self.VAL_EPSILON = "EPSILON"
        self.VAL_AVG100 = "AVG100"
        self.VAL_SOLVEDAT = "SOLVEDAT"

        # options
        self.epsilon = ep_start  # probability of choosing a random action
        self.epsilon_decay = ep_decay_rate  # decay of epsilon per episode
        self.epsilon_min = ep_min

        if mode_rbed :

            # RBED exclusive
            self.reward_threshold = -200  # Keep track of reward target for the agent
            self.epsilon_max = ep_start

            # Set to target value.
            if steps_to_move_in > 0:
                self.steps_to_move_in = steps_to_move_in
            else:
                self.steps_to_move_in = self.env_target

            self.quanta = (self.epsilon_max - self.epsilon_min) / self.steps_to_move_in  # what value to move epsilon by
            self.target_increment = target_increment  # Howmuch to increment the target every time the agent meets it.
        else:
            if steps_to_move_in == -1:
                self.steps_to_move_in = self.episode_count
            else:
                self.steps_to_move_in = steps_to_move_in

            self.quanta = ( ep_start - ep_min ) / self.steps_to_move_in


    def decay_epsilon(self, current_reward):
        if self.mode_rbed:
            self.rb_decay_epsilon(current_reward=current_reward)
        else:
            self.lin_decay_epsilon()

    def lin_decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.quanta

    def exp_decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def rb_decay_epsilon(self, current_reward = -200):
        if current_reward > self.reward_threshold and self.epsilon > self.epsilon_min:
            self.reward_threshold += self.target_increment
            self.epsilon -= self.quanta
            if self.epsilon < 0:
                self.epsilon = 0


    # Define Q-learning function
    def exec(self):

        learning = 0.2 # from the hyperparameter in the original code.
        # Determine size of discretized state space
        num_states = (self.env.observation_space.high - self.env.observation_space.low) * \
                     np.array([10, 100])
        num_states = np.round(num_states, 0).astype(int) + 1

        # Initialize Q table
        Q = np.random.uniform(low=-1, high=1,
                              size=(num_states[0], num_states[1],
                                    self.env.action_space.n))

        # Initialize variables to track rewards
        episode_rewards = []
        solved_yet = False

        # Run Q learning algorithm
        for i in trange(self.episode_count):
            # Initialize parameters
            done = False
            tot_reward, reward = 0, 0
            state = self.env.reset()

            # Discretize state
            state_adj = (state - self.env.observation_space.low) * np.array([10, 100])
            state_adj = np.round(state_adj, 0).astype(int)

            while not done:
                # No rendering required.
                # Render environment for last five episodes
                # if i >= (episodes - 20):
                #     self.env.render()

                # Determine next action - epsilon greedy strategy
                if np.random.random() < 1 - self.epsilon:
                    action = np.argmax(Q[state_adj[0], state_adj[1]])
                else:
                    action = np.random.randint(0, self.env.action_space.n)

                # Get next state and reward
                state2, reward, done, info = self.env.step(action)

                # Discretize state2
                state2_adj = (state2 - self.env.observation_space.low) * np.array([10, 100])
                state2_adj = np.round(state2_adj, 0).astype(int)

                # Allow for terminal states
                if done and state2[0] >= 0.5:
                    Q[state_adj[0], state_adj[1], action] = reward

                # Adjust Q value for current state
                else:
                    delta = learning * (reward +
                                        self.discount * np.max(Q[state2_adj[0],
                                                            state2_adj[1]]) -
                                        Q[state_adj[0], state_adj[1], action])
                    Q[state_adj[0], state_adj[1], action] += delta

                # Update variables
                tot_reward += reward
                state_adj = state2_adj


            episode_rewards.append(tot_reward)
            # Logging and other metrics.
            if len(episode_rewards[-101:-1]) == 0:
                mean_100ep_reward = -200
            else:
                mean_100ep_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

            # Decay epsilon
            self.decay_epsilon(current_reward=tot_reward)

            # Sacred Logging
            self.ex.log_scalar(self.VAL_EPSILON, self.epsilon)

            # log reward
            self.ex.log_scalar(self.VAL_REWARD, tot_reward)

            # log average reward
            self.ex.log_scalar(self.VAL_AVG100, mean_100ep_reward)

            # log solved at
            if mean_100ep_reward >= self.env_target and (not solved_yet):
                solved_yet = True
                self.ex.log_scalar(self.VAL_SOLVEDAT, i)

        self.env.close()
