from sacred import Experiment
import random

class TestDummy:

    def __init__(self, rate = -1, n_runs = 1, start_val = 1, sacred_ex = None):
        self.rate = rate
        self.num_runs = n_runs
        self.start_val = start_val
        self.run_count = 0
        self.ex = sacred_ex
        print("TD initialized with \trate=", rate,"\tnum runs=",n_runs,"\tstart val=",start_val )

    def decay(self, value, rate, min_val = 0.0001):
        return max(value*rate, min_val)

    def exec(self):
        final_val = self.start_val
        total_val = 0
        for i in range(self.num_runs):
            final_val = self.decay(final_val,self.rate)
            random_reward = random.randint(0,10)*random.random()
            # For Sacred Logging
            self.ex.log_scalar("EPSILON" , final_val, self.run_count)
            self.ex.log_scalar("REWARD" , random_reward, self.run_count)
            total_val += random_reward
            self.run_count+=1

        total_val = random.random()*total_val
        self.ex.log_scalar("SOLVEDAT", total_val)