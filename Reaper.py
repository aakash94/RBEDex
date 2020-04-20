## This file when presented with a folder will parse all the run files for logs in it.

import argparse
import numpy as np
from collections import defaultdict
import os
import json
import shutil
from matplotlib import pyplot as plt


class Reaper:

    def create_folder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
            else:
                shutil.rmtree(directory)
                os.makedirs(directory)
        except:
            print('Error: unable to create' + directory)

    def get_list_folders(self, path):
        count = 0
        folders = []
        for i in os.listdir(path):
            full_path = path + i
            if os.path.isdir(full_path):
                folders.append(full_path)
        return folders

    def get_relevant_folder_count(self, path):
        count = 0
        for i in os.listdir(path):
            full_path = path + i
            if os.path.isdir(full_path) and i.isdigit():
                count += 1
        return count

    def get_list_of_json(self, path):
        json_paths = []
        file_path = '/metrics.json'
        for i in range(1, self.get_relevant_folder_count(path) + 1):
            full_path = path + str(i) + file_path
            json_paths.append(full_path)
        return json_paths

    def __init__(self,path="Runs/"):
        self.path = path
        self.REWARD_VAL = "REWARD"
        self.EPSILON_VAL = "EPSILON"
        self.SOLVEDAT_VAL = "SOLVEDAT"
        self.AVG100_VAL = "AVG100"
        self.VALUE_KEY = "values"

    def reap(self,path=""):
        # will create output folders
        if path == "":
            path = self.path

        for folder in self.get_list_folders(path):
            # each folder is an experiment
            # average values so that each experiment can contribute only 1 value
            folder += "/"
            destination = folder + "results/"

            destination_epsilon = destination + "average_apsilon.csv"
            destination_reward = destination + "average_reward.csv"
            destination_avg100 = destination + "average100.csv"
            destination_solvedat = destination + "solvedat.json"

            self.create_folder(destination)

            files = self.get_list_of_json(folder)
            num_run = len(files)

            file = files[0]
            # assuming number of ep is the max number of episodes per run
            num_ep = len(json.load(open(file))[self.EPSILON_VAL][self.VALUE_KEY])

            np_reward = np.zeros((num_run, num_ep))
            np_epsilon = np.zeros((num_run, num_ep))
            np_avg100 = np.zeros((num_run, num_ep))
            solved_counter = defaultdict(int)

            for i, file in enumerate(files):
                data = json.load(open(file))

                list_epsilon = data[self.EPSILON_VAL][self.VALUE_KEY]
                list_reward = data[self.REWARD_VAL][self.VALUE_KEY]
                list_avg100 = data[self.AVG100_VAL][self.VALUE_KEY]
                solved_at = data[self.SOLVEDAT_VAL][self.VALUE_KEY][0]

                solved_counter[solved_at] += 1

                np_reward[i] = list_reward
                np_epsilon[i] = list_epsilon
                np_avg100 = list_avg100

            np_reward = np.mean(np_reward, axis=0)
            np_epsilon = np.mean(np_epsilon, axis=0)
            np_avg100 = np.mean(np_avg100,axis=0)

            np.savetxt(destination_epsilon, np_epsilon, delimiter=',')
            np.savetxt(destination_reward, np_reward, delimiter=',')
            np.savetxt(destination_avg100,np_avg100,delimiter=',')

            json_w = json.dumps(solved_counter)
            f = open(destination_solvedat, "w")
            f.write(json_w)
            f.close()

    def metric_plot(self,path=""):

        if path == "":
            path = self.path

        for folder in self.get_list_folders(path):
            # each folder is an experiment
            # average values so that each experiment can contribute only 1 value
            folder += "/"
            files = self.get_list_of_json(folder)
            num_run = len(files)

            file = files[0]
            # assuming number of ep is the max number of episodes per run
            num_ep = len(json.load(open(file))[self.EPSILON_VAL][self.VALUE_KEY])

            for i, file in enumerate(files):
                folder_num = str(i + 1)
                current_folder = folder + "" + folder_num + "/graphs/"
                data = json.load(open(file))

                list_epsilon = data[self.EPSILON_VAL][self.VALUE_KEY]
                list_reward = data[self.REWARD_VAL][self.VALUE_KEY]
                list_avg100 = data[self.AVG100_VAL][self.VALUE_KEY]
                solved_at = data[self.SOLVEDAT_VAL][self.VALUE_KEY][0]

                plt.clf()
                plt.plot(list_epsilon)
                plt.xlabel("Episode")
                plt.ylabel("Epsilon")
                save_path = current_folder + "epsilon.png"
                plt.savefig(save_path)

                plt.clf()
                plt.plot(list_reward)
                plt.xlabel("Episode")
                plt.ylabel("Reward")
                save_path = current_folder + "reward.png"
                plt.savefig(save_path)

                plt.clf()
                plt.plot(list_avg100)
                plt.xlabel("Episode")
                plt.ylabel("Avg100")
                save_path = current_folder + "avg100.png"
                plt.savefig(save_path)

if __name__ == '__main__':
    print("Itdoesntevenmatter")
