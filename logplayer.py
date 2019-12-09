import os
import pickle
import numpy as np
import cv2
import csv
from concurrent.futures import ThreadPoolExecutor

SIMPLE = True


class SteeringToWheelVelWrapper:
    """ Converts policy that was trained with [velocity|heading] actions to
    [wheelvel_left|wheelvel_right] to comply with AIDO evaluation format
    """

    def __init__(self, gain=1.0, trim=0.0, radius=0.0318, k=27.0, limit=1.0, wheel_dist=0.102):
        # Should be adjusted so that the effective speed of the robot is 0.2 m/s
        self.gain = gain

        # Directional trim adjustment
        self.trim = trim

        # Wheel radius
        self.radius = radius

        # Motor constant
        self.k = k

        # Wheel velocity limit
        self.limit = limit

        # Distance between wheels
        self.wheel_dist = wheel_dist

    def convert(self, vel, angle):

        # Distance between the wheels
        baseline = self.wheel_dist

        # assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # adjusting k by gain and trim
        k_r_inv = (self.gain + self.trim) / k_r
        k_l_inv = (self.gain - self.trim) / k_l

        omega_r = (vel + 0.5 * angle * baseline) / self.radius
        omega_l = (vel - 0.5 * angle * baseline) / self.radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, self.limit), -self.limit)
        u_l_limited = max(min(u_l, self.limit), -self.limit)

        vels = np.array([u_l_limited, u_r_limited])
        return vels


ik = SteeringToWheelVelWrapper()


class Logger:
    def __init__(self, log_file):
        self._log_file = open(log_file, 'wb')
        # we log the data in a multithreaded fashion
        self._multithreaded_recording = ThreadPoolExecutor(8)
        self.recording = []

    def log(self, observation, action, reward, done, info):
        self.recording.append({
            'step': [
                observation,
                action,
            ],
            # this is metadata, you may not use it at all, but it may be helpful for debugging purposes
            'metadata': [
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
        # make file read-only after finishing
        os.chmod(self._log_file.name, 0o444)


newLog = Logger(log_file='processed.log')


class Reader:
    def __init__(self, log_file):
        self._log_file = open(log_file, 'rb')

    def read(self, simple=SIMPLE):
        end = False
        observations = []
        actions = []
        pwm_left = []
        pwm_right = []
        if not simple:
            reward = []
        while not end:
            try:
                log = pickle.load(self._log_file)
                for entry in log:
                    step = entry['step']
                    actions.append(step[1])
                    observations.append(step[0])
                    pwm_left_local, pwm_right_local = ik.convert(
                        step[1][0], step[1][1])
                    pwm_left.append(pwm_left_local)
                    pwm_right.append(pwm_right_local)
                    if not simple:
                        meta = entry['metadata']
                        reward.append(meta[1])
                    else:
                        reward = None

            except EOFError:

                end = True

        return observations, actions, pwm_left, pwm_right, reward

    def close(self):
        self._log_file.close()


reader1 = Reader('flipped.log')
if not SIMPLE:
    reader2 = Reader('Combined_raw.log')
    raw, action, pwm_left, pwm_right, reward = reader2.read()
else:
    raw = None
    
observation, action, pwm_left, pwm_right, reward = reader1.read()

# print('Length Check: ', len(raw), len(action),
#       len(pwm_left), len(pwm_right), len(reward))


class Illustrator:
    def __init__(self, observation, action, pwm_left, pwm_right, reward, raw):
        self.observation = observation
        self.action = action
        self.pwm_left = pwm_left
        self.pwm_right = pwm_right
        self.reward = reward
        self.raw = raw
        cv2.namedWindow('Raw_log', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Training_log', cv2.WINDOW_NORMAL)
        return

    def run_log_parsers(self, excel=True, show=True, post_process=True,increase=False):
        for i in range(len(self.observation)):
            print('Current Frame: ',i)
            if show:
                self.show_log(i)

            if excel:
                self.write_to_excel(i)

            if post_process:
                self.process_good_reward(i)
            
            if increase:
                self.increase_data(i)
        return

    def show_log(self, index):
        training_frame = self.observation[index]
        linear = self.action[index][0]
        angular = self.action[index][1]
        local_pwm_left = self.pwm_left[index]
        local_pwm_right = self.pwm_right[index]
        if not SIMPLE:
            local_reward = self.reward[index]
            raw_frame = self.raw[index]
        if not SIMPLE:
            canvas = cv2.resize(raw_frame, (640, 480))
            #! Speed bar indicator
            cv2.rectangle(canvas, (20, 240), (50, int(240-220*linear)),
                      (76, 84, 255), cv2.FILLED)
            cv2.rectangle(canvas, (320, 430), (int(320-150*angular), 460),
                      (76, 84, 255), cv2.FILLED)
            cv2.imshow('Raw_log', canvas)
        cv2.imshow('Training_log', training_frame)
        cv2.waitKey(1)

    def write_to_excel(self, index):
        linear = self.action[index][0]
        angular = self.action[index][1]
        local_pwm_left = self.pwm_left[index]
        local_pwm_right = self.pwm_right[index]
        if not SIMPLE:
            local_reward = self.reward[index]
        else:
            local_reward = 0

        with open('distribution.csv', 'a') as newFile:
            newFileWriter = csv.writer(newFile)
            newFileWriter.writerow(
                [linear, angular, local_pwm_left, local_pwm_right, local_reward])
        return

    def increase_data(self,index):
        current_frame = self.observation[index]
        current_action = self.action[index]
        new_frame = cv2.flip(current_frame,1)
        new_actions = self.action[index]*-1
        rewards = None
        done = None
        info = None        
        newLog.log(current_frame,current_action,rewards,done,info)
        newLog.log(new_frame,new_actions,rewards,done,info)
        newLog.on_episode_done()
        return

    def process_good_reward(self, index):
        training_frame = self.training[index]
        actions = self.action[index]
        rewards = self.reward[index]
        done = None
        info = None

        if rewards > 0.4:
            newLog.log(training_frame, actions, rewards, done, info)
            newLog.on_episode_done()
        return


runner = Illustrator(observation, action, pwm_left,
                     pwm_right, reward, raw)
runner.run_log_parsers()
