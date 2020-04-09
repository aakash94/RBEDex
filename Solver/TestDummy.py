
class TestDummy:

    def __init__(self, rate = -1, n_runs = 1, start_val = 1):
        self.rate = rate
        self.num_runs = n_runs
        self.start_val = start_val
        print("TD initialized with \trate=", rate,"\tnum runs=",n_runs,"\tstart val=",start_val )

    def decay(self, value, rate):
        return value*rate

    def run_test(self):
        final_val = self.start_val
        for i in range(self.num_runs):
            final_val = self.decay(final_val,self.rate)
            print(i,"\t",final_val)
