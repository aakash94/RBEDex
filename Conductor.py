from sacred import Experiment
from sacred.observers import FileStorageObserver
from Solver.TestDummy import TestDummy
from Solver.CartPoleNEC import EpisodicAgent
from Solver.StableDQN import CarpoleDQN

ex = Experiment("RBEDex1")


class Conductor:

    def __init__(self,
                 agent="CartPoleNEC",
                 env="CartPole-v1",
                 mode_rbed=True,
                 ep_start=1.0,
                 ep_min=0,
                 ep_decay_rate=0.98,
                 episode_count=500,
                 target_increment=1,
                 schedule_timesteps=5000,
                 sacred_ex=None
                 ):

        if agent == 'CartPoleNEC':
            self.solver = EpisodicAgent(env_string=env,
                                        ep_start=ep_start,
                                        ep_decay_rate=ep_decay_rate,
                                        mode_rbed=mode_rbed,
                                        ep_min=ep_min,
                                        episode_count=episode_count,
                                        target_increment=target_increment,
                                        sacred_ex=sacred_ex)

        elif agent == 'CartPoleDQN':
            self.solver = CarpoleDQN(env_string=env,
                                        ep_start=ep_start,
                                        mode_rbed=mode_rbed,
                                        ep_min=ep_min,
                                        episode_count=episode_count,
                                        target_increment=target_increment,
                                        schedule_timesteps=schedule_timesteps,
                                        sacred_ex=sacred_ex)

        else:
            self.solver = TestDummy(rate=ep_decay_rate, n_runs=episode_count, start_val=ep_start, sacred_ex=sacred_ex)

    def go_run(self):
        self.solver.exec()


@ex.main
def main(agent="CartPoleNEC",
         env="CartPole-v1",
         mode_rbed=True,
         ep_start=1.0,
         ep_min=0,
         ep_decay_rate=0.98,
         episode_count=500,
         schedule_timesteps = 5000,
         target_increment=1):

    '''
    print(agent)
    print(mode_rbed)
    print(env)
    print(ep_start)
    print(ep_min)
    print(ep_decay_rate)
    print(episode_count)
    print(target_increment)
    if mode_rbed:
        print("Got true correct")
    '''
    cndctr = Conductor(agent=agent,
                       env=env,
                       mode_rbed=mode_rbed,
                       ep_start=ep_start,
                       ep_min=ep_min,
                       ep_decay_rate=ep_decay_rate,
                       episode_count=episode_count,
                       target_increment=target_increment,
                       schedule_timesteps=schedule_timesteps,
                       sacred_ex=ex)
    cndctr.go_run()


### Can also use VARIANTS for different agents/experiments
@ex.config
def configure():
    # this needs to be a separate method
    # else main gets called twice
    agent="CartPole-NEC"
    env="CartPole-v1"
    mode_rbed=True
    ep_start=1.0
    ep_min=0
    ep_decay_rate=0.98
    episode_count=500
    target_increment=1
    fso_folder = "Runs"
    schedule_timesteps = 5000
    ex.observers.append(FileStorageObserver(fso_folder))

if __name__ == '__main__':
    ex.run_commandline()
