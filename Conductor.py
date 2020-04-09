from sacred import Experiment
from Solver.TestDummy import TestDummy

ex = Experiment()

class Conductor:

    def __init__(self, decay_rate=1, n_runs=1, start_val=1):

        # take in all the params and initialize them here
        #self.param = param
        self.td = TestDummy(rate = decay_rate, n_runs=n_runs, start_val=start_val)

    def go_run(self):

        # placeholder probably call something to run here.
        # print("Hello Sacred World", self.param)
        self.td.run_test()


@ex.main
def main(decay_rate, n_runs, start_val):
    # the values initialized are initial placeholder values
    # actual values will come from command line fom sacred
    param_val = 0  # read this from config or whatever

    cndctr = Conductor(decay_rate=decay_rate, n_runs=n_runs, start_val=start_val)
    cndctr.go_run()

@ex.config
def configure():
    # this needs to be a separate method
    # else main gets called teic
    decay_rate = -1
    n_runs = 1
    start_val = 1


if __name__ == '__main__':
    ex.run_commandline()
    # main()
