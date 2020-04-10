from sacred import Experiment
from sacred.observers import FileStorageObserver
from Solver.TestDummy import TestDummy

ex = Experiment("Test1")

class Conductor:

    def __init__(self, decay_rate=1, n_runs=1, start_val=1, sacred_ex = None):

        # take in all the params and initialize them here
        #self.param = param
        self.td = TestDummy(rate = decay_rate, n_runs=n_runs, start_val=start_val, sacred_ex=sacred_ex)

    def go_run(self):

        # placeholder probably call something to run here.
        # print("Hello Sacred World", self.param)
        self.td.run_test()


@ex.main
def main(decay_rate, n_runs, start_val):
    # the values initialized are initial placeholder values
    # actual values will come from command line fom sacred
    param_val = 0  # read this from config or whatever

    cndctr = Conductor(decay_rate=decay_rate, n_runs=n_runs, start_val=start_val, sacred_ex=ex)
    cndctr.go_run()


### Can also use VARIANTS for different agents/experiments
@ex.config
def configure():
    # this needs to be a separate method
    # else main gets called twice
    decay_rate = -1
    n_runs = 1
    start_val = 1
    fso_folder = "Runs"
    ex.observers.append(FileStorageObserver(fso_folder))


if __name__ == '__main__':
    ex.run_commandline()
    # main()
