from concurrent.futures import ThreadPoolExecutor

import os
import pickle
import numpy as np

class Logger:
    def __init__(self, env, log_file):
        self.env = env

        self._log_file = open(log_file, 'wb')
        # we log the data in a multithreaded fashion
        self._multithreaded_recording = ThreadPoolExecutor(4)
        self.recording = []

    def log(self, observation, action, reward, done, info):
        x, y, z = self.env.cur_pos
        self.recording.append({
            'step': [
                observation,
                action,
            ],
            # this is metadata, you may not use it at all, but it may be helpful for debugging purposes
            'metadata': [
                (x, y, z, self.env.cur_angle),  # we store the pose, just in case we need it
                reward,
                done,
                info
            ]
        })

    def on_episode_done(self):
        print('Quick write!')
        self._multithreaded_recording.submit(self._commit)

    def _commit(self):
        # we use pickle to store our data
        pickle.dump(self.recording, self._log_file)
        self._log_file.flush()
        del self.recording[:]
        self.recording.clear()

    def close(self):
        self._multithreaded_recording.shutdown()
        self._log_file.close()
        os.chmod(self._log_file.name, 0o444)  # make file read-only after finishing
        