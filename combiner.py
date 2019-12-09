import os
import pickle
import numpy as np
import cv2
import csv
from concurrent.futures import ThreadPoolExecutor


class Reader:
    def __init__(self, log_file):
        self._log_file = open(log_file, 'rb')

    def read(self):
        end = False
        observations = []
        actions = []
        while not end:
            try:
                log = pickle.load(self._log_file)
                for entry in log:
                    step = entry['step']
                    actions.append(step[1])
                    observations.append(step[0])
            except EOFError:
                end = True

        return observations, actions

    def close(self):
        self._log_file.close()


reader1 = Reader('1.log')
reader2 = Reader('2.log')
obs1, log1 = reader1.read()
obs2, log2 = reader2.read()


class Logger:
    def __init__(self, log_file):
        self._log_file = open(log_file, 'wb')
        # we log the data in a multithreaded fashion
        self._multithreaded_recording = ThreadPoolExecutor(8)
        self.recording = []

    def log(self, observation, action):
        self.recording.append({
            'step': [
                observation,
                action,
            ]
        })

    def on_episode_done(self):
        print('Quick write!')
        self._multithreaded_recording.submit(self.commit)

    def commit(self):
        # we use pickle to store our data
        pickle.dump(self.recording, self._log_file)
        self._log_file.flush()
        del self.recording[:]
        self.recording.clear()

    def close(self):
        self._multithreaded_recording.shutdown()
        self._log_file.close()
        # make file read-only after finishing
        os.chmod(self._log_file.name, 0o444)


newLog = Logger(log_file='Combined.log')


class Combiner:
    def __init__(self, log1_obs, log1_action, log2_obs, log2_action):
        self.log1_obs = log1_obs
        self.log1_action = log1_action
        self.log2_obs = log2_obs
        self.log2_action = log2_action

    def combine(self):

        for i in range(len(self.log1_obs)):
            newLog.log(self.log1_obs[i], self.log1_action[i])
            newLog.commit()
            print(i)
        for i in range(len(self.log2_obs)):
            newLog.log(self.log2_obs[i], self.log2_action[i])
            newLog.commit()
            print(i)


combiner = Combiner(obs1, log1, obs2, log2)
combiner.combine()
