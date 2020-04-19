""" inspired from https://gym.openai.com/evaluations/eval_lEi8I8v2QLqEgzBxcvRIaA/ """

import numpy as np
from sacred import Experiment
from tqdm import trange

import gym
from collections import deque

class EpisodicAgent():
    """
    Episodic agent is a simple nearest-neighbor based agent:
    - At training time it remembers all tuples of (state, action, reward).
    - After each episode it computes the empirical value function based
        on the recorded rewards in the episode.
    - At test time it looks up k-nearest neighbors in the state space
        and takes the action that most often leads to highest average value.
    """

    def __init__(self,
                 env_string = 'CartPole-v1',
                 ep_start = 1.0,
                 ep_decay_rate = 0.98,
                 mode_rbed = True,
                 ep_min = 0,
                 episode_count = 500,
                 steps_to_move_in = -1,
                 target_increment = 1,
                 sacred_ex = None):

        self.env = gym.make(env_string)
        self.action_space = self.env.action_space
        self.ex = sacred_ex
        self.mode_rbed = mode_rbed
        self.episode_count = episode_count
        self.env_target = 195
        #assert isinstance(action_space, gym.spaces.discrete.Discrete), 'unsupported action space for now.'

        # for sacred logging
        self.VAL_REWARD = "REWARD"
        self.VAL_EPSILON = "EPSILON"
        self.VAL_AVG100 = "AVG100"
        self.VAL_SOLVEDAT = "SOLVEDAT"

        # options
        self.epsilon = ep_start  # probability of choosing a random action
        self.epsilon_decay = ep_decay_rate  # decay of epsilon per episode
        self.epsilon_min = ep_min

        # RBED exclusive
        self.reward_threshold = 0  # Keep track of reward target for the agent
        self.epsilon_max = ep_start

        # Set to target value.
        if steps_to_move_in > 0:
            self.steps_to_move_in = steps_to_move_in
        else:
            self.steps_to_move_in = self.env_target

        self.quanta = (self.epsilon_max - self.epsilon_min) / self.steps_to_move_in  # what value to move epsilon by
        self.target_increment = target_increment # Howmuch to increment the target every time the agent meets it.


        # Hyper Params from original version of code #NoChangeRequired
        self.nnfind = 500  # how many nearest neighbors to consider in the policy?
        self.mem_needed = 500  # amount of data to have before we can start exploiting
        self.mem_size = 50000  # maximum size of memory
        self.gamma = 0.95  # discount factor

        # internal vars
        self.iter = 0
        self.mem_pointer = 0  # memory pointer
        self.max_pointer = 0
        self.db = None  # large array of states seen
        self.dba = {}  # actions taken
        self.dbr = {}  # rewards obtained at all steps
        self.dbv = {}  # value function at all steps, computed retrospectively
        self.ep_start_pointer = 0


    def exec(self):
        last_ep_reward = 0
        scores100 = deque(maxlen=100)  # because solution depends on last 100 solution.
        scores100.append(int(0))
        solved_yet = False
        last_ep_reward = 0

        for i in trange(self.episode_count):

            sum_reward = 0
            done = False
            reward = 0
            time_step =0
            ob = self.env.reset()
            while not done:
                action = self.act(observation=ob, reward=reward, done=done, last_total_reward=last_ep_reward)
                ob, reward, done, _ = self.env.step(action)
                time_step +=1
                sum_reward += reward
                if done:
                    last_ep_reward = sum_reward
                    # print("Run ", i, "\tsteps elapsed ", self.env._elapsed_steps)
                    self.act(observation=ob, reward=reward, done=done, last_total_reward=last_ep_reward)

            # log epsilon
            self.ex.log_scalar(self.VAL_EPSILON, self.epsilon)

            # log reward
            self.ex.log_scalar(self.VAL_REWARD, last_ep_reward)

            # log average reward
            scores100.append(int(last_ep_reward))

            # check if solved
            current_score = sum(scores100) / 100
            self.ex.log_scalar(self.VAL_AVG100, current_score)

            # log solved at
            if current_score >= self.env_target and (not solved_yet):
                solved_yet = True
                self.ex.log_scalar(self.VAL_SOLVEDAT, i)


    def exp_decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def rb_decay_epsilon(self, current_reward = 0):
        if current_reward > self.reward_threshold and self.epsilon > self.epsilon_min:
            self.reward_threshold += self.target_increment
            self.epsilon -= self.quanta
            if self.epsilon < 0:
                self.epsilon = 0
        #     print("Updated \t", self.epsilon)
        # else:
        #     print("current reward ",current_reward, "\treward thresh", self.reward_threshold, "\tepsilon", self.epsilon,"\tmin", self.epsilon_min);

    def act(self, observation, reward, done, last_total_reward=0):
        assert isinstance(observation, np.ndarray) and observation.ndim == 1, 'unsupported observation type for now.'

        if self.db is None:
            # lazy initialization of memory
            self.db = np.zeros((self.mem_size, observation.size))
            self.mem_pointer = 0
            self.ep_start_pointer = 0

        # we have enough data, we want to explore, and we have seen at least one episode already (so values were computed)
        if self.iter > self.mem_needed and np.random.rand() > self.epsilon and self.dbv:
            # exploit: find the few closest states and pick the action that led to highest rewards
            # 1. find k nearest neighbors
            ds = np.sum((self.db[:self.max_pointer] - observation) ** 2, axis=1)  # L2 distance
            ix = np.argsort(ds)  # sorts ascending by distance
            ix = ix[:min(len(ix), self.nnfind)]  # crop to only some number of nearest neighbors

            # find the action that leads to most success. do a vote among actions
            adict = {}
            ndict = {}
            for i in ix:
                vv = self.dbv[i]
                aa = self.dba[i]
                vnew = adict.get(aa, 0) + vv
                adict[aa] = vnew
                ndict[aa] = ndict.get(aa, 0) + 1

            for a in adict:  # normalize by counts
                adict[a] = adict[a] / ndict[a]

            its = [(y, x) for x, y in adict.items()]
            its.sort(reverse=True)  # descending
            a = its[0][1]

        else:
            # explore: do something random
            a = self.action_space.sample()

        # record move to database
        if self.mem_pointer < self.mem_size:
            self.db[self.mem_pointer] = observation  # save the state
            self.dba[self.mem_pointer] = a  # and the action we took
            self.dbr[self.mem_pointer - 1] = reward  # and the reward we obtained last time step
            self.dbv[self.mem_pointer - 1] = 0
        self.mem_pointer += 1
        self.iter += 1

        if done:  # episode Ended;

            # compute the estimate of the value function based on this rollout
            v = 0
            for t in reversed(range(self.ep_start_pointer, self.mem_pointer)):
                v = self.gamma * v + self.dbr.get(t, 0)
                self.dbv[t] = v

            self.ep_start_pointer = self.mem_pointer
            self.max_pointer = min(max(self.max_pointer, self.mem_pointer), self.mem_size)


            if self.mode_rbed :
                self.rb_decay_epsilon(current_reward=last_total_reward)
            else :
                self.exp_decay_epsilon()

        return a