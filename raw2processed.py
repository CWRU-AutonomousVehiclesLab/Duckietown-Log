import os
import pickle
import numpy as np
import cv2
import csv
from concurrent.futures import ThreadPoolExecutor
import carnivalmirror as cm
import itertools
class Distortion(object):
    def __init__(self, camera_rand=False):
        # Image size
        self.H = 480
        self.W = 640
        # K - Intrinsic camera matrix for the raw (distorted) images.
        camera_matrix = [
            305.5718893575089, 0, 303.0797142544728,
            0, 308.8338858195428, 231.8845403702499,
            0, 0, 1,
        ]
        self.camera_matrix = np.reshape(camera_matrix, (3, 3))

        # distortion parameters - (k1, k2, t1, t2, k3)
        distortion_coefs = [
            -0.2, 0.0305,
            0.0005859930422629722, -0.0006697840226199427, 0
        ]

        self.distortion_coefs = np.reshape(distortion_coefs, (1, 5))

        # R - Rectification matrix - stereo cameras only, so identity
        self.rectification_matrix = np.eye(3)

        # Initialize mappings

        # Used for rectification
        self.mapx = None
        self.mapy = None

        # Used for distortion
        self.rmapx = None
        self.rmapy = None
        if camera_rand:
            self.camera_matrix, self.distortion_coefs = self.randomize_camera()

        # New camera matrix - specifies the intrinsic (camera) matrix
        #  of the processed (rectified) image
        self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(cameraMatrix=self.camera_matrix,
                                                                  distCoeffs=self.distortion_coefs,
                                                                  imageSize=(self.W, self.H),
                                                                  alpha=0)

    def randomize_camera(self):
        """Randomizes parameters of the camera according to a specified range"""
        K = self.camera_matrix
        D = self.distortion_coefs

        # Define ranges for the parameters:
        # TODO move this to config file
        ranges = {'fx': (0.95 * K[0, 0], 1.05 * K[0, 0]),
                  'fy': (0.95 * K[1, 1], 1.05 * K[1, 1]),
                  'cx': (0.95 * K[0, 2], 1.05 * K[0, 2]),
                  'cy': (0.95 * K[1, 2], 1.05 * K[1, 2]),
                  'k1': (0.95 * D[0, 0], 1.05 * D[0, 0]),
                  'k2': (0.95 * D[0, 1], 1.05 * D[0, 1]),
                  'p1': (0.95 * D[0, 2], 1.05 * D[0, 2]),
                  'p2': (0.95 * D[0, 3], 1.05 * D[0, 3]),
                  'k3': (0.95 * D[0, 4], 1.05 * D[0, 4])}

        # Create a ParameterSampler:
        sampler = cm.ParameterSampler(ranges=ranges, cal_width=self.W, cal_height=self.H)

        # Get a calibration from sampler
        calibration = sampler.next()

        return calibration.get_K(self.H), calibration.get_D()

    def distort(self, observation):
        """
        Distort observation using parameters in constructor
        """

        if self.mapx is None:
            # Not initialized - initialize all the transformations we'll need
            self.mapx = np.zeros(observation.shape)
            self.mapy = np.zeros(observation.shape)

            H, W, _ = observation.shape

            # Initialize self.mapx and self.mapy (updated)
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(cameraMatrix=self.camera_matrix,
                                                               distCoeffs=self.distortion_coefs,
                                                               R=self.rectification_matrix,
                                                               newCameraMatrix=self.new_camera_matrix,
                                                               size=(W, H),
                                                               m1type=cv2.CV_32FC1)

            # Invert the transformations for the distortion
            self.rmapx, self.rmapy = self._invert_map(self.mapx, self.mapy)

        return cv2.remap(observation, self.rmapx, self.rmapy, interpolation=cv2.INTER_NEAREST)

    def _undistort(self, observation):
        """
        Undistorts a distorted image using camera parameters
        """

        # If mapx is None, then distort was never called
        assert self.mapx is not None, "You cannot call undistort on a rectified image"

        return cv2.remap(observation, self.mapx, self.mapy, cv2.INTER_NEAREST)

    def _invert_map(self, mapx, mapy):
        """
        Utility function for simulating distortion
        Source: https://github.com/duckietown/Software/blob/master18/catkin_ws
        ... /src/10-lane-control/ground_projection/include/ground_projection/
        ... ground_projection_geometry.py
        """

        H, W = mapx.shape[0:2]
        rmapx = np.empty_like(mapx)
        rmapx.fill(np.nan)
        rmapy = np.empty_like(mapx)
        rmapy.fill(np.nan)

        for y, x in itertools.product(range(H), range(W)):
            tx = mapx[y, x]
            ty = mapy[y, x]

            tx = int(np.round(tx))
            ty = int(np.round(ty))

            if (0 <= tx < W) and (0 <= ty < H):
                rmapx[ty, tx] = x
                rmapy[ty, tx] = y

        self._fill_holes(rmapx, rmapy)
        return rmapx, rmapy

    def _fill_holes(self, rmapx, rmapy):
        """
        Utility function for simulating distortion
        Source: https://github.com/duckietown/Software/blob/master18/catkin_ws
        ... /src/10-lane-control/ground_projection/include/ground_projection/
        ... ground_projection_geometry.py
        """
        H, W = rmapx.shape[0:2]

        R = 2
        F = R * 2 + 1

        def norm(_):
            return np.hypot(_[0], _[1])

        deltas0 = [(i - R - 1, j - R - 1) for i, j in itertools.product(range(F), range(F))]
        deltas0 = [x for x in deltas0 if norm(x) <= R]
        deltas0.sort(key=norm)

        def get_deltas():
            return deltas0

        holes = set()

        for i, j in itertools.product(range(H), range(W)):
            if np.isnan(rmapx[i, j]):
                holes.add((i, j))

        while holes:
            nholes = len(holes)
            nholes_filled = 0

            for i, j in list(holes):
                # there is nan
                nholes += 1
                for di, dj in get_deltas():
                    u = i + di
                    v = j + dj
                    if (0 <= u < H) and (0 <= v < W):
                        if not np.isnan(rmapx[u, v]):
                            rmapx[i, j] = rmapx[u, v]
                            rmapy[i, j] = rmapy[u, v]
                            nholes_filled += 1
                            holes.remove((i, j))
                            break

            if nholes_filled == 0:
                break
distorter = Distortion()


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

    def read(self):
        end = False
        observations = []
        actions = []
        reward = []
        pwm_left = []
        pwm_right = []

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
                    meta = entry['metadata']
                    reward.append(meta[1])

            except EOFError:

                end = True

        return observations, actions, pwm_left, pwm_right, reward

    def close(self):
        self._log_file.close()


reader1 = Reader('raw_log_2.log')
observation, action, pwm_left, pwm_right, reward = reader1.read()

print('Length Check: ', len(observation), len(action),
      len(pwm_left), len(pwm_right), len(reward))


class Illustrator:
    def __init__(self, observation, action, pwm_left, pwm_right, reward):
        self.observation = observation
        self.action = action
        self.pwm_left = pwm_left
        self.pwm_right = pwm_right
        self.reward = reward
        cv2.namedWindow('Training_log', cv2.WINDOW_NORMAL)
        return

    def convert2distortion(self,index):
        raw_frame = self.observation[index]
        linear = self.action[index][0]
        angular = self.action[index][1]
        reward = self.reward[index]

        obs_distorted = distorter.distort(raw_frame)
        #! resize to Nvidia standard:
        def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
            # initialize the dimensions of the image to be resized and
            # grab the image size
            dim = None
            (h, w) = image.shape[:2]

            # if both the width and height are None, then return the
            # original image
            if width is None and height is None:
                return image

            # check to see if the width is None
            if width is None:
                # calculate the ratio of the height and construct the
                # dimensions
                r = height / float(h)
                dim = (int(w * r), height)

            # otherwise, the height is None
            else:
                # calculate the ratio of the width and construct the
                # dimensions
                r = width / float(w)
                dim = (width, int(h * r))

            # resize the image
            resized = cv2.resize(image, dim, interpolation=inter)

            # return the resized image
            return resized

        obs_distorted_DS = image_resize(obs_distorted, width=200)

        #! ADD IMAGE-PREPROCESSING HERE!!!!!
        # height, width = obs_distorted_DS.shape[:2]
        # print('Distorted return image Height: ', height,' Width: ',width)
        cropped = obs_distorted_DS[0:150, 0:200]

        # NOTICE: OpenCV changes the order of the channels !!!
        cropped_final = cv2.cvtColor(cropped, cv2.COLOR_BGR2YUV)

        done = None
        info = None
        newLog.log(cropped_final, self.action[index], self.reward[index],done,info)
        newLog.on_episode_done()

    def run_log_parsers(self, excel=False, show=True, post_process=False, increase=False,raw2train=True    ):
        for i in range(len(self.observation)):
            print('Current Frame: ', i)
            if raw2train:
                self.convert2distortion(i)            
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
        local_reward = self.reward[index]
        canvas = cv2.resize(training_frame, (640, 480))
        #! Speed bar indicator
        # cv2.rectangle(canvas, (20, 240), (50, int(240-220*linear)),
        #                  (76, 84, 255), cv2.FILLED)
        # cv2.rectangle(canvas, (320, 430), (int(320-150*angular), 460),
        #                 (76, 84, 255), cv2.FILLED)
        cv2.imshow('Training_log', training_frame)
        cv2.waitKey(1)

    def write_to_excel(self, index):
        linear = self.action[index][0]
        angular = self.action[index][1]
        local_pwm_left = self.pwm_left[index]
        local_pwm_right = self.pwm_right[index]
        local_reward = self.reward[index]

        with open('distribution.csv', 'a') as newFile:
            newFileWriter = csv.writer(newFile)
            newFileWriter.writerow(
                [linear, angular, local_pwm_left, local_pwm_right, local_reward])
        return

    # def increase_data(self, index):
    #     current_frame = self.observation[index]
    #     current_action = self.action[index]
    #     new_frame = cv2.flip(current_frame, 1)
    #     new_actions = self.action[index]*-1
    #     rewards = None
    #     done = None
    #     info = None
    #     newLog.log(current_frame, current_action, rewards, done, info)
    #     newLog.log(new_frame, new_actions, rewards, done, info)
    #     newLog.on_episode_done()
    #     return

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


runner = Illustrator(observation, action, pwm_left,pwm_right, reward)
runner.run_log_parsers()
