#!/usr/bin/env python3
import os
import sys
sys.path.append('.')
sys.path.append('./env')
sys.path.append('./inference')
import rospy 

from scipy.spatial.transform import Rotation as R
import numpy as np
import timeit

import tensorflow as tf

import gflags
from common_flags import FLAGS
from config import *
from utilities import bcolors

from enum import Enum

## environment type
if RUN_IN_SIM:
    if SIM_USE_FLIGHTMARE:
        from env.flightmare_wrapper import FlightmareWrappers
    else:
        from env.rotors_wrappers import RotorsWrappers
else:
    from realtime_ros_wrapper_infogain_ardupilot import RealtimeRosWrapperInfoGainArdupilot

## inference type
if COLLISION_USE_TENSORRT:
    if PLANNING_TYPE == 1: # seVAE-ORACLE
        from network_inference_tensorrt import seVAENetworkInferenceTensorRTV2
    else:
        from network_inference_tensorrt import NetworkInferenceTensorRTV2
else:
    if PLANNING_TYPE == 1: # seVAE-ORACLE
        from network_inference_tensorflow import seVAENetworkInferenceTensorflow
    else:
        from network_inference_tensorflow import NetworkInferenceTensorflowV2

if INFOGAIN_USE_TENSORRT:
    from network_inference_tensorrt import InfoNetworkInferenceTensorRT
else:    
    from network_inference_tensorflow import InfoNetworkInferenceTensorflow

# import cv2

if __name__ == "__main__":
    # Limiting GPU memory growth: https://www.tensorflow.org/guide/gpu
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # tf.config.experimental.set_virtual_device_configuration(
            #     gpus[0],
            #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])            
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(bcolors.OKBLUE, len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs", bcolors.ENDC)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(bcolors.FAIL + "GPU error" + bcolors.ENDC)
            print(e)

class ActionResult(Enum):
    FAIL = 0
    RUNNING = 1
    SUCCESS = 2
    NONE = 3 # initial value

class ORACLEPlanner:
    def __init__(self, waypoint_path = None):
        self.run_planner = False
        self.step_cnt = 0

        # create fixed action library
        FOV_rad = np.deg2rad(PLANNING_HORIZONTAL_FOV)
        self.cmd_velocity_x_lib = np.array([CMD_VELOCITY], dtype=np.float32) # only support one value of vel_x for now
        self.cmd_vel_x = np.repeat(self.cmd_velocity_x_lib, NUM_VEL_Z * NUM_YAW)
        self.cmd_vel_x = np.reshape(self.cmd_vel_x, (NUM_SEQUENCE_TO_EVALUATE, 1))
        
        # calculate action_seq at hover state (for visualization)
        angle_z_limit = np.array([-np.deg2rad(CAM_PITCH) - np.deg2rad(PLANNING_VERTICAL_FOV/2 ), 
                                -np.deg2rad(CAM_PITCH) + np.deg2rad(PLANNING_VERTICAL_FOV/2)])
        angle_z_limit = np.clip(angle_z_limit, np.deg2rad(-80.0), np.deg2rad(80.0))
        cmd_angle_z_lib = np.linspace(angle_z_limit[0], angle_z_limit[1], NUM_VEL_Z, dtype=np.float32)
        replace_idx = int(NUM_VEL_Z / 2) - 1
        cmd_angle_z_lib[replace_idx] = 0.0 # make sure we always have action sequence having v_z = 0.0
        cmd_velocity_z_tmp = np.tan(cmd_angle_z_lib)
        cmd_velocity_z_lib = np.array([], dtype=np.float32)
        for i in range(NUM_VEL_X):
            cmd_velocity_z_lib = np.append(cmd_velocity_z_lib, self.cmd_velocity_x_lib[i] * cmd_velocity_z_tmp)
        self.cmd_vel_z = np.repeat(cmd_velocity_z_lib, NUM_YAW)
        # cmd_vel_z = np.tile(cmd_vel_z, NUM_VEL_X)
        self.cmd_vel_z = np.reshape(self.cmd_vel_z, (NUM_SEQUENCE_TO_EVALUATE, 1))

        cmd_yaw_relative_lib = np.linspace(-0.45*FOV_rad, 0.45*FOV_rad, NUM_YAW, dtype=np.float32)
        self.cmd_yaw_relative = np.tile(cmd_yaw_relative_lib, NUM_VEL_X * NUM_VEL_Z)
        self.cmd_yaw_relative = np.reshape(self.cmd_yaw_relative, (NUM_SEQUENCE_TO_EVALUATE, 1))
        
        self.action_seq = np.concatenate((self.cmd_vel_x, self.cmd_vel_z, self.cmd_yaw_relative), axis=1) # shape (NUM_SEQUENCE_TO_EVALUATE, ACTION_SHAPE_EVALUATE)
        self.action_seq = np.reshape(self.action_seq, (NUM_SEQUENCE_TO_EVALUATE, 1, ACTION_SHAPE_EVALUATE))
        self.action_seq = np.repeat(self.action_seq, ACTION_HORIZON, axis=1) # shape (NUM_SEQUENCE_TO_EVALUATE, ACTION_HORIZON, ACTION_SHAPE_EVALUATE)
        # print('self.action_seq:', self.action_seq)
        self.action_seq = np.ascontiguousarray(self.action_seq, dtype=np.float32)
        print('self.action_seq shape:', np.shape(self.action_seq))

        # create estimated trajectories (only for visualization purpose)
        self.trajectory_lib = np.zeros((NUM_SEQUENCE_TO_EVALUATE, ACTION_HORIZON, 3)) # position in vehicle frame
        self.trajectory_lib_ensemble = np.zeros((N_E, NUM_SEQUENCE_TO_EVALUATE, ACTION_HORIZON, 3))
        for i in range(NUM_SEQUENCE_TO_EVALUATE):
            # assumed fixed initial values (only for visualization)
            # we can update the robot' initial states every time we receive a new observation
            # but it will cost time just for visualization! 
            psi_tmp = 0.0
            vel_x_tmp = CMD_VELOCITY
            vel_y_tmp = 0.0
            vel_z_tmp = 0.0
            pos_tmp = np.zeros(3)
            for j in range(ACTION_HORIZON):
                cmd_vel_x = self.action_seq[i,j,0]
                cmd_vel_y = 0.0
                cmd_vel_z = self.action_seq[i,j,1]
                cmd_relative_yaw = self.action_seq[i,j,2]
                for k in range(10): # do multiple integration in smaller time steps (1/10th) of original time step)
                    vel_x_tmp = ALPHA_VX * vel_x_tmp + (1 - ALPHA_VX) * cmd_vel_x
                    vel_y_tmp = ALPHA_VY * vel_y_tmp + (1 - ALPHA_VY) * cmd_vel_y
                    vel_z_tmp = ALPHA_VZ * vel_z_tmp + (1 - ALPHA_VZ) * cmd_vel_z
                    psi_tmp = ALPHA_PSI * psi_tmp + (1 - ALPHA_PSI) * cmd_relative_yaw
                    v_t = np.array([vel_x_tmp * np.cos(psi_tmp) - vel_y_tmp * np.sin(psi_tmp), 
                                    vel_x_tmp * np.sin(psi_tmp) + vel_y_tmp * np.cos(psi_tmp), vel_z_tmp])
                    pos_tmp = pos_tmp + v_t * DEPTH_TS * SKIP_STEP_GENERATE / 10
                self.trajectory_lib[i,j] = pos_tmp
        for k in range(N_E):
            for i in range(NUM_SEQUENCE_TO_EVALUATE):
                for j in range(ACTION_HORIZON):
                    self.trajectory_lib_ensemble[k,i,j] = (1 - 0.1*k) * self.trajectory_lib[i,j]


        # weight matrix
        self.time_weight = np.zeros((1, ACTION_HORIZON))
        for i in range(ACTION_HORIZON):
            self.time_weight[0,i] = np.exp(-TIME_WEIGHT_FACTOR * i)

        # read waypoint list
        if (waypoint_path != None):
            self.waypoints = np.loadtxt(waypoint_path, ndmin=2)
        else:
            self.waypoints = np.array([])
        print(bcolors.OKGREEN, 'waypoints:', self.waypoints, bcolors.ENDC)
        self.wp_idx = 0
        self.num_wp = self.waypoints.shape[0]
        if self.num_wp != 0:
            self.current_wp = self.waypoints[self.wp_idx]

        # UT params (http://ais.informatik.uni-freiburg.de/teaching/ws12/mapping/pdf/slam05-ukf.pdf)
        self.delta_vx = np.sqrt((L + LAMBDA) * P_vx, dtype=np.float32)
        self.delta_vy = np.sqrt((L + LAMBDA) * P_vy, dtype=np.float32)
        self.delta_vz = np.sqrt((L + LAMBDA) * P_vz, dtype=np.float32)
        W_m_0 = LAMBDA / (L + LAMBDA)
        W_m_i = 1 / (2 * (L + LAMBDA))
        self.W_m = np.array([W_m_0, W_m_i, W_m_i, W_m_i, W_m_i, W_m_i, W_m_i], dtype=np.float32)
        W_c_0 = W_m_0 + 1 - ALPHA**2 + BETA
        self.W_c = np.array([W_c_0, W_m_i, W_m_i, W_m_i, W_m_i, W_m_i, W_m_i], dtype=np.float32)
        print('LAMBDA:', LAMBDA)
        print('L + LAMBDA:', L + LAMBDA)
        print('self.W_m:', self.W_m)
        print('self.W_c:', self.W_c)

        self.action_seq_expand = np.repeat(self.action_seq, N_SIGMA, axis=0)
        self.action_seq_expand = np.ascontiguousarray(self.action_seq_expand, dtype=np.float32)
        print('action_seq_expand shape:', np.shape(self.action_seq_expand)) 

        self.turn_angle = 0.0
        self.receive_goal_yaw_service = False
        ## env type
        if RUN_IN_SIM:
            if SIM_USE_FLIGHTMARE:
                self.env = FlightmareWrappers()
            else:
                self.env = RotorsWrappers()
            rospy.sleep(1.0)
            self.env.reset()
            if PLANNING_TYPE == 3: # with voxblox
                self.env.clear_map()
        else:
            self.env = RealtimeRosWrapperInfoGainArdupilot() #RealtimeRosWrapper()

        # for collecting metrics
        self.collision_episode_cnt = 0
        self.itr = 0

        self.env.register_start_cb(self.start_planner)
        self.env.register_stop_cb(self.stop_planner)
        self.env.register_goal_cb(self.goal_cb)        

    def start_planner(self):
        self.run_planner = True
        if self.num_wp > 0:
            self.wp_time_start[self.wp_idx] = rospy.Time.now()

    def stop_planner(self):
        self.run_planner = False
        self.step_cnt = 0
        self.env.stop_robot()

    def goal_cb(self, turn_angle):
        # in planning mode and doesn't follow waypoints?
        if (self.run_planner == True) and (self.num_wp == 0) and (self.receive_goal_yaw_service == False):
            self.receive_goal_yaw_service = True
            self.goal_yaw = (self.robot_yaw + turn_angle + np.pi) % (2 * np.pi) - np.pi
            print('RECEIVED set_goal_dir service:', np.rad2deg(turn_angle), ' deg')
            success = True
        else:
            success = False
        return success

    def deadend_action(self, min_col_score):
        if (self.deadend_step == 0) and (min_col_score < DEADEND_COL_SCORE_THRESHOLD_HIGH):
            return ActionResult.SUCCESS
        elif (self.deadend_step == 1) and (min_col_score < DEADEND_COL_SCORE_THRESHOLD_LOW):
            self.deadend_step = 0
            return ActionResult.SUCCESS 
        else:
            print(bcolors.FAIL + 'DEADEND:' + str(min_col_score) + bcolors.ENDC)
            if self.deadend_step == 0:
                self.env.stop_robot()
                self.deadend_step = 1
                self.has_deadend_in_segment = True
                return ActionResult.RUNNING
            elif self.deadend_step == 1:
                self.env.yaw_in_spot(self.robot_yaw + np.deg2rad(15.0))
                return ActionResult.RUNNING        
        return ActionResult.RUNNING

    def calculate_goal_dir(self):
        if (self.num_wp == 0):
            goal_yaw = self.robot_yaw # try to maintain the current robot's yaw
            goal_el = 0.0
        else:
            delta_vec = self.current_wp - self.robot_pos
            goal_yaw = np.arctan2(delta_vec[1], delta_vec[0])
            goal_el = np.arctan2(delta_vec[2], np.sqrt(delta_vec[0]*delta_vec[0] + delta_vec[1]*delta_vec[1]))
        return goal_yaw, goal_el

    def update_waypoint(self):
        self.wp_idx = self.wp_idx + 1
        self.current_wp = self.waypoints[self.wp_idx]
        self.goal_yaw, self.goal_el = self.calculate_goal_dir()
        self.has_deadend_in_segment = False
        
        # variables for cheking timeout event
        self.wp_time_start[self.wp_idx] = rospy.Time.now()
        self.total_time_from_prev_wp = 0.0 # for timeout type 2

        print(bcolors.OKGREEN, 'REACH WP ', self.wp_idx, ': ', self.waypoints[self.wp_idx-1], bcolors.ENDC)
        print(bcolors.OKGREEN, 'Current pos:', self.robot_pos, bcolors.ENDC)
        print(bcolors.OKGREEN, 'NEXT WP:', self.current_wp, bcolors.ENDC)

    def reach_goal_action(self):
        if self.num_wp == 0:
            return ActionResult.SUCCESS
        if np.linalg.norm(self.robot_pos[0:2] - self.current_wp[0:2]) > WAYPOINT_DISTANCE_THRESHOLD: # no z comparison here
            return ActionResult.SUCCESS
        if (self.wp_idx >= self.num_wp - 1): # reach final waypoint
            self.env.stop_robot()
            return ActionResult.FAIL
        if ALLOW_YAW_AT_WAYPOINT == False:
            self.update_waypoint()
            return ActionResult.SUCCESS
        else: # yaw_in_spot to align with the next waypoint (if the next waypoint is not too close to the robot)
            while np.linalg.norm(self.robot_pos[0:2] - self.waypoints[self.wp_idx + 1, 0:2]) <= WAYPOINT_DISTANCE_THRESHOLD: # no z comparison here
                self.update_waypoint()
                if (self.wp_idx >= self.num_wp - 1): # reach final waypoint
                    self.env.stop_robot()
                    return ActionResult.FAIL
                pass
            delta_vec = self.waypoints[self.wp_idx + 1] - self.robot_pos
            goal_yaw = np.arctan2(delta_vec[1], delta_vec[0])
            delta_yaw = (self.robot_yaw - goal_yaw + np.pi) % (2 * np.pi) - np.pi
            if np.abs(delta_yaw) <= WAYPOINT_YAW_THRESHOLD:
                # print(bcolors.OKGREEN, 'Exit WAYPOINT YAWING mode', bcolors.ENDC)
                self.update_waypoint()
                return ActionResult.SUCCESS
            else:
                # print(bcolors.OKGREEN, 'In WAYPOINT YAWING mode', bcolors.ENDC)
                if delta_yaw > 0:
                    self.env.yaw_in_spot(self.robot_yaw - np.deg2rad(15.0))
                else:
                    self.env.yaw_in_spot(self.robot_yaw + np.deg2rad(15.0))
                return ActionResult.RUNNING

    def direction_service_action(self):
        if self.receive_goal_yaw_service == False:
            return ActionResult.SUCCESS
        delta_yaw = (self.robot_yaw - self.goal_yaw + np.pi) % (2 * np.pi) - np.pi
        if (np.abs(delta_yaw) <= WAYPOINT_YAW_THRESHOLD):
            self.receive_goal_yaw_service = False
            return ActionResult.SUCCESS
        else:
            self.env.yaw_in_spot(self.goal_yaw)
            return ActionResult.RUNNING

    def is_timeout(self): # TODO: this doesn't account for stop_planner period
        if self.num_wp == 0:
            return False
        if TIMEOUT_TYPE == 0:
            time_passed = np.abs((rospy.Time.now() - self.wp_time_start[self.wp_idx]).to_sec())
            if (time_passed > self.time_allowed[self.wp_idx]):
                print(bcolors.OKGREEN, 'TIMEOUT', bcolors.ENDC)
                return True
        elif TIMEOUT_TYPE == 1:
            time_passed = np.abs((rospy.Time.now() - self.wp_time_start[self.wp_idx]).to_sec())
            approx_remaining_time = np.linalg.norm(self.robot_pos[0:2] - self.current_wp[0:2]) / CMD_VELOCITY
            if (time_passed + approx_remaining_time > self.time_allowed[self.wp_idx]):
                print(bcolors.OKGREEN, 'TIMEOUT', bcolors.ENDC)
                return True                
        elif TIMEOUT_TYPE == 2:
            approx_remaining_time = np.linalg.norm(self.robot_pos[0:2] - self.current_wp[0:2]) / CMD_VELOCITY
            self.total_time_from_prev_wp = self.total_time_from_prev_wp + np.linalg.norm(self.robot_pos[0:2] - self.prev_pos[0:2]) / CMD_VELOCITY
            if (self.total_time_from_prev_wp + approx_remaining_time > self.time_allowed[self.wp_idx]):
                print(bcolors.OKGREEN, 'TIMEOUT', bcolors.ENDC)
                return True
        return False

    def timeout_action(self):
        if (PLANNING_TYPE < 2): # ORACLE or seVAE-ORACLE
            return ActionResult.SUCCESS
        if self.is_timeout() == False:
            return ActionResult.SUCCESS
        if self.has_deadend_in_segment == True: # prevent fluctuating btw deadend and timeout yaw mode
            return ActionResult.FAIL
        delta_vec = self.current_wp - self.robot_pos
        self.goal_yaw = np.arctan2(delta_vec[1], delta_vec[0])
        delta_yaw = (self.robot_yaw - self.goal_yaw + np.pi) % (2 * np.pi) - np.pi
        if (np.abs(delta_yaw) <= WAYPOINT_YAW_THRESHOLD):
            return ActionResult.FAIL
        else:
            if delta_yaw > 0:    
                self.env.yaw_in_spot(self.robot_yaw - np.deg2rad(15.0))
            else:
                self.env.yaw_in_spot(self.robot_yaw + np.deg2rad(15.0))
            return ActionResult.RUNNING

    def run(self):
        if COLLISION_USE_TENSORRT:
            if PLANNING_TYPE == 1: # seVAE-ORACLE
                network_inference = seVAENetworkInferenceTensorRTV2()
            else:
                network_inference = NetworkInferenceTensorRTV2() #NetworkInferenceTensorRT()
        else:    
            if PLANNING_TYPE == 1: # seVAE-ORACLE
                network_inference = seVAENetworkInferenceTensorflow()
            else:
                network_inference = NetworkInferenceTensorflowV2()

        if PLANNING_TYPE == 2: # A-ORACLE
            if INFOGAIN_USE_TENSORRT:
                info_network_inference = InfoNetworkInferenceTensorRT()
            else:
                info_network_inference = InfoNetworkInferenceTensorflow()
        
        print(bcolors.OKGREEN, "START planner", bcolors.ENDC)
        
        # current executed step in the action sequence
        self.step_cnt = 0

        # running time
        time_cnn_full = 0.0
        time_rnn_full = 0.0
        time_combiner_full = 0.0
        time_planner_full = 0.0
        time_cnn_info_full = 0.0
        time_rnn_info_full = 0.0
        time_planner_info_full = 0.0
        planning_itr = 0
        planning_itr_with_infogain = 0
        time_cnn_info = 0.0
        time_rnn_info = 0.0

        # metrics
        first_planning_step = True # for calculating traveled distance
        traveled_distance = 0.0
        traveled_time = 0.0
        average_acc = 0.0
        average_jerk = 0.0
        metric_cnt = 0

        self.info = {'status':'none'}
        self.done = False

        # calculate allowed time
        robot_state, current_di, current_mask, valid_obs = self.env.get_new_obs()
        while valid_obs == False: # wait until new sensor data is available
            if rospy.is_shutdown():
                print("ROSPY DEAD. EXITING.")
                sys.exit(0)
            try:
                rospy.sleep(0.0001)
                robot_state, current_di, current_mask, valid_obs = self.env.get_new_obs()
            except Exception as e:
                print(e)
                pass

        self.prev_pos = np.array([])
        self.total_time_from_prev_wp = 0.0
        if self.num_wp > 0:
            self.time_allowed = np.zeros(self.num_wp)
            self.wp_time_start = [None] * self.num_wp # save ROS timestamp
            for i in range(self.num_wp):
                if i == 0:
                    self.time_allowed[0] = np.linalg.norm(self.waypoints[i, 0:2] - robot_state[0:2]) / CMD_VELOCITY
                else:
                    self.time_allowed[i] = np.linalg.norm(self.waypoints[i, 0:2] - self.waypoints[i-1, 0:2]) / CMD_VELOCITY
            self.time_allowed = self.time_allowed * TIME_ALLOWED
        else:
            self.time_allowed = np.array([])

        if RUN_IN_SIM:
            traveled_distances = []
            traveled_times = []
            average_acc_list = []
            average_jerk_list = []
            collision_labels = []
        
        ##### a naive 'behavior tree' implementation #####
        self.tick = 1
        self.deadend_res = ActionResult.NONE
        self.reach_goal_res = ActionResult.NONE
        self.direction_service_res = ActionResult.NONE
        self.timeout_res = ActionResult.NONE

        self.deadend_step = 0
        self.has_deadend_in_segment = True

        ##### get the new observation #####
        while (RUN_IN_SIM == False) or (self.itr < NUM_EPISODES_EVALUATE):
            rospy.sleep(0.0001)
            robot_state, current_di, current_mask, valid_obs = self.env.get_new_obs()
            # wait until new sensor data is available
            while valid_obs == False:
                if rospy.is_shutdown():
                    print("ROSPY DEAD. EXITING.")
                    sys.exit(0)
                if (RUN_IN_SIM == True) and (SIM_USE_FLIGHTMARE):
                    # add the below checks for flightmare integration since depth image is not published when the end of traj is reached
                    # hence we need to reset the env without relying on the tick by get_new_obs()
                    if (self.env.reach_end_of_traj()): # reach final waypoint?
                        self.info = {'status':'reach final WP'}
                        self.done = True # reset env         
                        print(bcolors.OKGREEN, 'REACH FINAL WP', bcolors.ENDC)
                        print(bcolors.BLUE, 'Traveled distance [m]:', traveled_distance, bcolors.ENDC)
                        self.env.stop_robot()
                        self.run_planner = False
                        break
                    if self.env.is_robot_collide():
                        self.info = {'status':'collide'}
                        self.done = True # reset env
                        self.run_planner = False
                    if self.env.is_robot_outside():
                        self.info = {'status':'outside'}
                        self.done = True # reset env
                        self.run_planner = False
                try:
                    rospy.sleep(0.0001)
                    robot_state, current_di, current_mask, valid_obs = self.env.get_new_obs()
                except Exception as e:
                    print(e)
                    pass

            # debugging
            # print('evaluate_planner_info_gain')
            # cv2.imshow('threshold image', current_mask*255)
            # cv2.waitKey(1)

            if self.run_planner:
                if (self.tick == 1):
                    # calculate the metrics
                    if (first_planning_step):
                        first_planning_step = False
                        self.prev_pos = robot_state[0:3]
                        self.robot_pos = robot_state[0:3]
                        self.prev_acc = robot_state[6:9]
                        self.robot_acc = robot_state[6:9]
                        
                        stamp_start = rospy.Time.now()
                        current_stamp = stamp_start
                        prev_stamp = stamp_start
                    else:
                        self.prev_pos = self.robot_pos
                        self.robot_pos = robot_state[0:3]
                        self.prev_acc = self.robot_acc
                        self.robot_acc = robot_state[6:9]
                        traveled_distance = traveled_distance + np.linalg.norm(self.robot_pos - self.prev_pos)
                        
                        prev_stamp = current_stamp
                        current_stamp = rospy.Time.now()
                    
                    metric_cnt = metric_cnt + 1
                    average_acc = average_acc + np.linalg.norm(self.robot_acc)
                    average_jerk = average_jerk + np.linalg.norm(self.robot_acc - self.prev_acc) / ((current_stamp - prev_stamp).to_sec() + 1e-6)

                    ##### prepare input data for CPN and IPN #####
                    quaternion = robot_state[9:13]
                    angular_rate_z = robot_state[15]
    
                    if PLANNING_TYPE != 1:
                        if RUN_IN_SIM and USE_ADDITIVE_GAUSSIAN_IMAGE_NOISE:
                            current_di = current_di + np.random.normal(0.0, IMAGE_NOISE_FACTOR, (DI_SHAPE[0], DI_SHAPE[1])) * np.square(current_di) # quadratic noise
                            np.clip(current_di, 0.0, MAX_RANGE, out=current_di)
                        current_di = np.reshape(current_di, [1, DI_SHAPE[0], DI_SHAPE[1], DI_SHAPE[2]]) * MAX_RANGE_INV # scale the image to 0->1
                    else: # seVAE
                        current_di = np.reshape(current_di, [1, DI_LATENT_SIZE])

                    r_robot = R.from_quat(quaternion)
                    robot_euler_angles = r_robot.as_euler('xyz', degrees=False)
                    self.robot_yaw = robot_euler_angles[2]

                    # create action seq lib
                    # start = timeit.default_timer()
                    angle_z_limit = np.array([-np.deg2rad(CAM_PITCH) - robot_euler_angles[1] - np.deg2rad(PLANNING_VERTICAL_FOV/2), 
                                            -np.deg2rad(CAM_PITCH) - robot_euler_angles[1] + np.deg2rad(PLANNING_VERTICAL_FOV/2)])
                    angle_z_limit = np.clip(angle_z_limit, np.deg2rad(-80.0), np.deg2rad(80.0))
                    cmd_angle_z_lib = np.linspace(angle_z_limit[0], angle_z_limit[1], NUM_VEL_Z, dtype=np.float32)
                    replace_idx = int(NUM_VEL_Z / 2) - 1
                    cmd_angle_z_lib[replace_idx] = 0.0 # make sure we always have action sequence having v_z = 0.0
                    cmd_velocity_z_tmp = np.tan(cmd_angle_z_lib)
                    cmd_velocity_z_lib = np.array([], dtype=np.float32)
                    for i in range(NUM_VEL_X):
                        cmd_velocity_z_lib = np.append(cmd_velocity_z_lib, self.cmd_velocity_x_lib[i] * cmd_velocity_z_tmp)
                    self.cmd_vel_z = np.repeat(cmd_velocity_z_lib, NUM_YAW)
                    # cmd_vel_z = np.tile(cmd_vel_z, NUM_VEL_X)
                    self.cmd_vel_z = np.reshape(self.cmd_vel_z, (NUM_SEQUENCE_TO_EVALUATE, 1))
                    self.action_seq = np.concatenate((self.cmd_vel_x, self.cmd_vel_z, self.cmd_yaw_relative), axis=1) # shape (NUM_SEQUENCE_TO_EVALUATE, ACTION_SHAPE_EVALUATE)
                    self.action_seq = np.reshape(self.action_seq, (NUM_SEQUENCE_TO_EVALUATE, 1, ACTION_SHAPE_EVALUATE))
                    self.action_seq = np.repeat(self.action_seq, ACTION_HORIZON, axis=1) # shape (NUM_SEQUENCE_TO_EVALUATE, ACTION_HORIZON, ACTION_SHAPE_EVALUATE)
                    self.action_seq = np.ascontiguousarray(self.action_seq, dtype=np.float32)
                    self.action_seq_expand = np.repeat(self.action_seq, N_SIGMA, axis=0)
                    self.action_seq_expand = np.ascontiguousarray(self.action_seq_expand, dtype=np.float32)
                    # stop = timeit.default_timer()
                    # time_action_seq = stop - start
                    # print('TIME (ms): action_seq ', time_action_seq*1000)                    

                    # convert velocity fron body frame to yaw-aligned world frame
                    velocity_B = robot_state[3:6]
                    # add Gaussian noise
                    if RUN_IN_SIM and USE_ADDITIVE_GAUSSIAN_STATE_NOISE:
                        velocity_B = velocity_B + [np.random.normal(0.0, np.sqrt(P_vx), 1)[0], np.random.normal(
                            0.0, np.sqrt(P_vy), 1)[0], np.random.normal(0.0, np.sqrt(P_vz), 1)[0]]
                    velocity_W_psi = R.from_euler('xy', robot_euler_angles[0:2], degrees=False).as_matrix() @ (velocity_B)
                    if velocity_W_psi[0] < 0:
                        velocity_W_psi[0] = 0
                    
                    # sigma points from the Unscented Transform
                    if USE_UT:
                        VEL_UT_MIN = 0.0
                        if velocity_W_psi[0] - 1*self.delta_vx < VEL_UT_MIN:
                            velocity_W_psi_x_min = VEL_UT_MIN
                            velocity_W_psi_x_max = 2*velocity_W_psi[0] - VEL_UT_MIN
                        else:
                            velocity_W_psi_x_min = velocity_W_psi[0] - 1*self.delta_vx
                            velocity_W_psi_x_max = velocity_W_psi[0] + 1*self.delta_vx
                        state = np.array([[velocity_W_psi[0], velocity_W_psi[1], velocity_W_psi[2], angular_rate_z, robot_euler_angles[0], robot_euler_angles[1]],\
                                        [velocity_W_psi_x_min, velocity_W_psi[1], velocity_W_psi[2], angular_rate_z, robot_euler_angles[0], robot_euler_angles[1]],\
                                        [velocity_W_psi_x_max, velocity_W_psi[1], velocity_W_psi[2], angular_rate_z, robot_euler_angles[0], robot_euler_angles[1]],\
                                        [velocity_W_psi[0], velocity_W_psi[1] - 1*self.delta_vy, velocity_W_psi[2] , angular_rate_z, robot_euler_angles[0], robot_euler_angles[1]],\
                                        [velocity_W_psi[0], velocity_W_psi[1] + 1*self.delta_vy, velocity_W_psi[2] , angular_rate_z, robot_euler_angles[0], robot_euler_angles[1]],\
                                        [velocity_W_psi[0], velocity_W_psi[1], velocity_W_psi[2] - 1*self.delta_vz, angular_rate_z, robot_euler_angles[0], robot_euler_angles[1]],\
                                        [velocity_W_psi[0], velocity_W_psi[1], velocity_W_psi[2] + 1*self.delta_vz, angular_rate_z, robot_euler_angles[0], robot_euler_angles[1]]]) # must have (2*L + 1) sigma points
                    else:
                        state = np.array([[velocity_W_psi[0], velocity_W_psi[1], velocity_W_psi[2], angular_rate_z, robot_euler_angles[0], robot_euler_angles[1]],\
                                        [velocity_W_psi[0], velocity_W_psi[1], velocity_W_psi[2], angular_rate_z, robot_euler_angles[0], robot_euler_angles[1]],\
                                        [velocity_W_psi[0], velocity_W_psi[1], velocity_W_psi[2], angular_rate_z, robot_euler_angles[0], robot_euler_angles[1]],\
                                        [velocity_W_psi[0], velocity_W_psi[1], velocity_W_psi[2] , angular_rate_z, robot_euler_angles[0], robot_euler_angles[1]],\
                                        [velocity_W_psi[0], velocity_W_psi[1], velocity_W_psi[2] , angular_rate_z, robot_euler_angles[0], robot_euler_angles[1]],\
                                        [velocity_W_psi[0], velocity_W_psi[1], velocity_W_psi[2], angular_rate_z, robot_euler_angles[0], robot_euler_angles[1]],\
                                        [velocity_W_psi[0], velocity_W_psi[1], velocity_W_psi[2], angular_rate_z, robot_euler_angles[0], robot_euler_angles[1]]]) # must have (2*L + 1) sigma points

                    state_info_gain = np.array(
                        [[velocity_W_psi[0], velocity_W_psi[1], velocity_W_psi[2], angular_rate_z, robot_euler_angles[0], robot_euler_angles[1]]])

                    start_full = timeit.default_timer()

                    ##### run the CPN #####
                    # run the CNN part of CPN
                    di_feature_expand = []
                    if PLANNING_TYPE != 1:
                        start = timeit.default_timer()
                        di_feature = network_inference.call_cnn_only([(current_di).astype('float32')])
                        stop = timeit.default_timer()
                        time_cnn = stop - start
                        for i in range(N_E):
                            di_feature_expand.append(np.repeat(di_feature[i], N_SIGMA, axis=0))
                    else: # seVAE
                        time_cnn = 0.0
                        for i in range(N_E): # here we use the same VAE + ensemble of CPNs
                        # hence the same latent vector is pushed to the list
                        # TODO: this can be extended to ensemble of VAEs + ensemble of CPNs
                            di_feature_expand.append(np.repeat(current_di, N_SIGMA, axis=0).astype('float32'))
                    
                    # run the Combiner part of CPN
                    start = timeit.default_timer()
                    di_feature_with_state = network_inference.call_depth_state_combiner([state.astype('float32'), di_feature_expand])
                    stop = timeit.default_timer()
                    time_combiner = stop - start
                    initial_state_h = []
                    initial_state_c = []

                    for i in range(N_E):
                        di_feature_with_state_expand = np.tile(di_feature_with_state[i], [NUM_SEQUENCE_TO_EVALUATE ,1])
                        initial_state_h_single, initial_state_c_single = tf.split(di_feature_with_state_expand, 2, axis=1) # np.split returns non-contiguous arrays
                        initial_state_h.append(initial_state_h_single)
                        initial_state_c.append(initial_state_c_single)

                    # run the Prediction part of CPN
                    start = timeit.default_timer()
                    temp_collision_prob = network_inference.call_recurrent_net([initial_state_h, initial_state_c, self.action_seq_expand])
                    stop = timeit.default_timer()
                    time_rnn = stop - start
                    temp_collision_prob = np.array(temp_collision_prob)
                    # print('temp_collision_prob:', temp_collision_prob.shape) # (N_E, N_SIGMA * NUM_SEQUENCE_TO_EVALUATE, ACTION_HORIZON, 1)

                    # calculate the time-weighted sum
                    max_col_prob_sample = np.matmul(self.time_weight, temp_collision_prob)
                    # print('max_col_prob_sample:', max_col_prob_sample.shape) # (N_E, N_SIGMA * NUM_SEQUENCE_TO_EVALUATE, 1, 1)
                    # if VISUALIZATION_MODE == 1:
                    #     temp_collision_prob = np.reshape(temp_collision_prob, [NUM_SEQUENCE_TO_EVALUATE, N_SIGMA, ACTION_HORIZON, 1], 'C')
                    #     temp_collision_prob = np.average(temp_collision_prob, axis=1)
                    
                    # unscented transformation (UT)
                    max_col_prob_sample = np.reshape(max_col_prob_sample, [N_E, NUM_SEQUENCE_TO_EVALUATE, N_SIGMA], 'C')
                    # calculate mean
                    max_col_prob_UT_mean = np.dot(max_col_prob_sample, self.W_m)
                    max_col_prob_UT_mean = np.reshape(max_col_prob_UT_mean, [N_E, NUM_SEQUENCE_TO_EVALUATE, 1])
                    # calculate cov
                    max_col_prob_UT_cov = max_col_prob_sample - max_col_prob_UT_mean
                    max_col_prob_UT_cov = np.square(max_col_prob_UT_cov)
                    max_col_prob_UT_cov = np.dot(max_col_prob_UT_cov, self.W_c)
                    max_col_prob_UT_cov = np.reshape(max_col_prob_UT_cov, [N_E, NUM_SEQUENCE_TO_EVALUATE], 'C')
                    max_col_prob_UT_cov = np.average(max_col_prob_UT_cov, axis=0)
                    max_col_prob_UT_cov = np.reshape(max_col_prob_UT_cov, [1, NUM_SEQUENCE_TO_EVALUATE])

                    # combine predictions from multiple NNs
                    max_col_prob_UT_mean = np.reshape(max_col_prob_UT_mean, [N_E, NUM_SEQUENCE_TO_EVALUATE], 'C')
                    max_col_prob_MC_mean = np.average(max_col_prob_UT_mean, axis=0)
                    # print('max_col_prob_MC_mean:', max_col_prob_MC_mean)
                    max_col_prob_MC_mean = np.reshape(max_col_prob_MC_mean, [1, NUM_SEQUENCE_TO_EVALUATE])
                    
                    max_col_prob_MC_cov = max_col_prob_UT_mean - max_col_prob_MC_mean
                    max_col_prob_MC_cov = np.square(max_col_prob_MC_cov)
                    max_col_prob_MC_cov = np.average(max_col_prob_MC_cov, axis=0)
                    max_col_prob_MC_cov = np.reshape(max_col_prob_MC_cov, [1, NUM_SEQUENCE_TO_EVALUATE])
                    # print('max_col_prob_MC_cov:', max_col_prob_MC_cov.shape)

                    max_col_prob_total_std = np.sqrt(max_col_prob_UT_cov + max_col_prob_MC_cov)                              

                    # reshape to be compatible with the rest of the code
                    max_col_prob_MC_mean = np.reshape(max_col_prob_MC_mean, [NUM_SEQUENCE_TO_EVALUATE, 1])
                    max_col_prob_total_std = np.reshape(max_col_prob_total_std, [NUM_SEQUENCE_TO_EVALUATE, 1])

                    # calculate final cost (mean + alpha * std)
                    max_col_prob = max_col_prob_MC_mean + 0.05*max_col_prob_total_std                                                      

                    # find the 'safe' action sequences
                    indx_best_tmp =  np.argmin(max_col_prob)
                    indx = np.nonzero(max_col_prob < max_col_prob[indx_best_tmp] + COLLISION_THRESHOLD)[0]
                    # print('indx:', np.shape(indx)) #(N, )

                    ##### START the behavior tree (naive implementation!) #####
                    self.reach_goal_res = ActionResult.NONE
                    self.direction_service_res = ActionResult.NONE
                    self.timeout_res = ActionResult.NONE

                    self.deadend_res = self.deadend_action(max_col_prob[indx_best_tmp])
                    
                    if (self.deadend_res == ActionResult.SUCCESS):
                        self.reach_goal_res = self.reach_goal_action()
                    else:
                        self.tick = 1 # keep ticking for every observation received (high rate)

                    if (self.reach_goal_res == ActionResult.SUCCESS):
                        self.direction_service_res = self.direction_service_action()
                    else:
                        if (self.reach_goal_res == ActionResult.FAIL): # reach final waypoint
                            self.info = {'status':'reach final WP'}
                            self.done = True # reset env
                            # self.done = False # don't reset env                     
                            print(bcolors.OKGREEN, 'REACH FINAL WP', bcolors.ENDC)
                            print(bcolors.BLUE, 'Traveled distance [m]:', traveled_distance, bcolors.ENDC)
                            
                        self.tick = 1 # keep ticking for every observation received (high rate)

                    if (self.direction_service_res == ActionResult.SUCCESS):
                        self.timeout_res = self.timeout_action()
                    else:
                        self.tick = 1 # keep ticking for every observation received (high rate)

                    if (self.timeout_res == ActionResult.SUCCESS):
                        self.tick = 0 # stop ticking when receiving the next observation (replan at lower rate, defined by STEPS_TO_REPLAN)
                    elif (self.timeout_res == ActionResult.FAIL):
                        self.tick = 0
                        # don't need to evalute infogain because timeout event has happened
                    else:
                        self.tick = 1 # keep ticking for every observation received (high rate)
                        # after this, we wait for the next tick (received new observation and self.tick == 1)

                    if (self.tick == 0):
                        ##### run the information gain prediction if necessary and pick the most informative action sequence #####
                        run_infogain_prediction = False
                        # if there are more than one safe action sequence or timeout hasn't happened?
                        if (indx.size > 1) and (self.timeout_res != ActionResult.FAIL):
                            if current_mask is not None:
                                current_mask = np.reshape(current_mask, [1, DI_SHAPE[0], DI_SHAPE[1], 1])
                                current_mask[np.where(current_di >= MAX_MASK_RANGE_EVALUATE * MAX_RANGE_INV)] = 0
                                current_mask[np.where(current_di == 0)] = 0
                                # cv2.imshow('mask', current_mask[0,:,:,0])
                                # cv2.waitKey(1)
                                if (PLANNING_TYPE == 2):
                                    # run the info gain network if there are something in the mask
                                    run_infogain_prediction = np.any(current_mask > 0)
                                    if run_infogain_prediction == False:
                                        print('EMPTY mask') 
                                elif (PLANNING_TYPE == 3): # always call the service for voxblox-expert
                                    run_infogain_prediction = True

                        low_info_gain = False # if the best information gain predicted is low?
                        if run_infogain_prediction:
                            if PLANNING_TYPE == 2:
                                di_current_with_mask = np.concatenate((current_di, current_mask), axis=3)
                                start_info = timeit.default_timer()
                                info_gain_feature = info_network_inference.call_feature_extractor([di_current_with_mask.astype('float32')])
                                stop_info = timeit.default_timer()
                                time_cnn_info = stop_info - start_info

                                state_info_expand = np.tile(state_info_gain, (NUM_SEQUENCE_TO_EVALUATE_INFOGAIN, 1)).astype('float32')

                                # evaluate all the action sequences
                                start_info = timeit.default_timer()
                                if INFOGAIN_USE_TENSORRT == True:
                                    info_gains = info_network_inference.call_recurrent_net([state_info_expand, info_gain_feature, self.action_seq])
                                else:
                                    info_gains = info_network_inference.call_recurrent_net([state_info_expand, 
                                        np.tile(info_gain_feature, [NUM_SEQUENCE_TO_EVALUATE_INFOGAIN, 1, 1, 1]), self.action_seq])
                                # shape: (NUM_SEQUENCE_TO_EVALUATE_INFOGAIN, ACTION_HORIZON, 1)
                                stop_info = timeit.default_timer()
                                time_rnn_info = stop_info - start_info

                                info_gains_sum = np.sum(info_gains, axis=1)
                                interestingness = info_gains_sum[indx]
                                indx_best_in_K = np.argmax(interestingness)
                                indx_best = indx[indx_best_in_K]

                                # print(bcolors.OKGREEN + 'info_gains_sum[indx_best]:' + str(info_gains_sum[indx_best_in_K]) + bcolors.ENDC)
                                low_info_gain = (info_gains_sum[indx_best_in_K] * 100 < INFOGAIN_THRESHOLD) # the label is scaled by 0.01
                            elif PLANNING_TYPE == 3:
                                # call the service from voxblox
                                start_info = timeit.default_timer()
                                info_gains_sum = self.env.call_info_gain_baseline(np.concatenate((robot_state[0:3], robot_euler_angles), axis=0).astype('float32'))
                                stop_info = timeit.default_timer()
                                time_rnn_info = stop_info - start_info

                                info_gains_sum = np.array(info_gains_sum)
                                interestingness = info_gains_sum[indx]
                                indx_best_in_K = np.argmax(interestingness)
                                indx_best = indx[indx_best_in_K]
                                # print(bcolors.OKGREEN + 'info_gains_sum[indx_best]:' + str(info_gains_sum[indx_best]) + bcolors.ENDC)
                                low_info_gain = (info_gains_sum[indx_best] < INFOGAIN_THRESHOLD)
                            
                            if (low_info_gain == True):
                                print('LOW infogain')
                        follow_goal = (not run_infogain_prediction) or (low_info_gain == True)

                        ##### pick the best action sequence based on the collision cost only if necessary #####
                        if follow_goal:
                            # follow the goal
                            self.goal_yaw, self.goal_el = self.calculate_goal_dir()
                            yaw_cost = np.abs((self.action_seq[indx,0,2] + self.robot_yaw - self.goal_yaw + np.pi) % (2 * np.pi) - np.pi)
                            indx_yaw_best = np.where(yaw_cost == yaw_cost.min())[0]
                            indx2 = indx[indx_yaw_best]
                            el_cost = np.abs( (np.arctan2(self.action_seq[indx2, 0, 1], self.action_seq[indx2, 0, 0]) - self.goal_el + np.pi) % (2 * np.pi) - np.pi)
                            indx_best_in_K = np.argmin(el_cost)
                            indx_best = indx2[indx_best_in_K]

                        stop_full = timeit.default_timer()
                        time_full = stop_full - start_full
                # endif (self.tick == 1)

                ##### execute the chosen action sequence (in between replanning steps) #####
                if (self.tick == 0):
                    cmd_evaluate = self.action_seq[indx_best]
                    action = [[cmd_evaluate[self.step_cnt, 0], 0.0, cmd_evaluate[self.step_cnt, 1], cmd_evaluate[self.step_cnt, 2] + self.robot_yaw]]
                    self.done, self.info = self.env.step(action)

                    if (self.step_cnt == 0):
                        ##### print collision score and running time #####
                        print(bcolors.OKGREEN + 'max_col_prob[indx_best]:' + str(max_col_prob[indx_best]) + bcolors.ENDC)
                        print(bcolors.OKGREEN + 'max_col_prob_MC_mean[indx_best]:' + str(max_col_prob_MC_mean[indx_best]) + bcolors.ENDC)
                        print(bcolors.OKGREEN + 'max_col_prob_total_std[indx_best]:' + str(max_col_prob_total_std[indx_best]) + bcolors.ENDC)

                        planning_itr += 1
                        time_cnn_full += time_cnn
                        time_combiner_full += time_combiner
                        time_rnn_full += time_rnn
                        if (run_infogain_prediction):
                            planning_itr_with_infogain += 1
                            time_cnn_info_full += time_cnn_info
                            time_rnn_info_full += time_rnn_info
                            time_planner_info_full += time_full # mixed both CPN and IPN's running time
                        else:
                            time_planner_full += time_full # just CPN's running time
                        if (run_infogain_prediction and (planning_itr_with_infogain > 0)):
                            print('TIME (ms): cnn ', time_cnn_full*1000/planning_itr, ', combiner ', time_combiner_full*1000/planning_itr, ', rnn ', time_rnn_full*1000/planning_itr\
                                , ', cnn info', time_cnn_info_full*1000/planning_itr_with_infogain, ', rnn info', time_rnn_info_full*1000/planning_itr_with_infogain, ', full ', time_planner_info_full*1000/planning_itr_with_infogain)
                        else:
                            print('TIME (ms): cnn ', time_cnn_full*1000/planning_itr, ', combiner ', time_combiner_full*1000/planning_itr, ', rnn ', time_rnn_full*1000/planning_itr\
                                , ', full ', time_planner_full*1000/planning_itr)
                        time_cnn_info = 0.0
                        time_rnn_info = 0.0
                        
                        ##### visualization #####
                        if ENABLE_VISUALIZATION:
                            if (PLANNING_TYPE == 2) and (RUN_IN_SIM == False):
                                # trigger the detector (only when in real system to save computation time)
                                self.env.start_detector()
                            # print('action:', action)
                            # best_action_seq = np.reshape(cmd_evaluate, (1, ACTION_HORIZON, ACTION_SHAPE_EVALUATE))
                            # heatmap_info = info_network_inference.make_gradcam_heatmap_info(state_info_gain, di_current_with_mask, best_action_seq)    
                            # save_and_display_gradcam_info(di_current_with_mask, heatmap_info)
                            if VISUALIZATION_MODE == 0:
                                self.env.visualize_trajectory(self.trajectory_lib, max_col_prob, np.array([]), best_indx=indx_best, worst_indx=0, safe_indx=indx)
                                # self.env.visualize_trajectory(self.trajectory_lib, max_col_prob, np.array([]), best_indx=indx_best_info_gain, worst_indx=indx_best_info_gain)
                            elif VISUALIZATION_MODE == 1:
                                self.env.visualize_trajectory(self.trajectory_lib, max_col_prob, temp_collision_prob, best_indx=indx_best, worst_indx=0) # NOTE: for debugging only
                                # self.env.visualize_trajectory(self.trajectory_lib, max_col_prob, info_gains, best_indx=indx_best, worst_indx=indx_worst)
                            elif VISUALIZATION_MODE == 3:
                                # self.trajectory_lib_ensemble: [N_E, NUM_SEQUENCE_TO_EVALUATE, ACTION_HORIZON, 3]
                                # calculate the best_indx and safe_indx for all N_E nets
                                indx_best_list = []
                                safe_indx_list = []
                                for k in range(N_E):
                                    indx_best_tmp =  np.argmin(max_col_prob_UT_mean[k])
                                    indx_best_list.append(indx_best_tmp)
                                    safe_indx_tmp = np.nonzero(max_col_prob_UT_mean[k] < max_col_prob_UT_mean[k,indx_best_tmp] + COLLISION_THRESHOLD)[0]
                                    safe_indx_list.append(safe_indx_tmp)
                                # max_col_prob_UT_mean: [N_E, NUM_SEQUENCE_TO_EVALUATE]
                                self.env.visualize_trajectory(self.trajectory_lib_ensemble, max_col_prob_UT_mean, np.array([]), best_indx=indx_best_list, worst_indx=0, safe_indx=safe_indx_list)

                    self.step_cnt += 1
                    if (self.step_cnt == STEPS_TO_REPLAN): # replan after ... steps
                        self.step_cnt = 0
                        self.tick = 1 # tick to replan again
                # endif (self.tick == 0)
            # endif (self.run_planner)
            
            ##### print the metrics #####
            if RUN_IN_SIM:
                # Check if the robot collided            
                if self.info['status'] == 'collide':
                    print('STATUS collide')
                    self.collision_episode_cnt += 1 

                if self.done:
                    print('info status:', self.info['status'])
                    print(bcolors.BLUE + "End episode " + str(self.itr) + bcolors.ENDC)
                    print(bcolors.BLUE + 'Collision_episode_cnt ' + str(self.collision_episode_cnt) + bcolors.ENDC)

                    stamp_end = rospy.Time.now()
                    traveled_time = (stamp_end - stamp_start).to_sec()
                    traveled_times.append(traveled_time)
                    traveled_distances.append(traveled_distance)
                    if metric_cnt > 0:
                        average_acc_list.append(average_acc / metric_cnt)
                    else:
                        average_acc_list.append(0.0)
                    if metric_cnt > 1:
                        average_jerk_list.append(average_jerk / (metric_cnt - 1))
                    else:
                        average_jerk_list.append(0.0)
                    if self.info['status'] == 'collide':
                        collision_labels.append(1)
                    elif self.info['status'] == 'invalid best ACTION SEQUENCE':
                        collision_labels.append(2)
                    elif self.info['status'] == 'timeout':
                        collision_labels.append(3)
                    elif self.info['status'] == 'outside':
                        collision_labels.append(4)
                    else: # reach the goal
                        collision_labels.append(0)
                    # print the metrics of all flights
                    print('traveled_times:', traveled_times)
                    print('traveled_distances:', traveled_distances)
                    print('average_acc_list:', average_acc_list)
                    print('average_jerk_list:', average_jerk_list)
                    print('collision_labels:', collision_labels)

                    # reset variables
                    self.info = {'status':'none'}
                    self.done = False
                    self.itr += 1
                    self.step_cnt = 0
                    self.wp_idx = 0
                    self.num_wp = self.waypoints.shape[0]
                    if self.num_wp != 0:
                        self.current_wp = self.waypoints[self.wp_idx]
                    self.run_planner = True
                    first_planning_step = True # for calculating traveled distance
                    self.tick = 1
                    self.deadend_step = 0
                    self.has_deadend_in_segment = False
                    self.total_time_from_prev_wp = 0.0
                    traveled_distance = 0.0
                    traveled_time = 0.0
                    average_acc = 0.0
                    average_jerk = 0.0
                    metric_cnt = 0.0
                    self.env.reset()
                    if PLANNING_TYPE == 3:
                        self.env.clear_map()

        sys.exit(0)
        # self.env.pause()

if __name__ == '__main__':
    try:
        argv = FLAGS(sys.argv)  # parse flags
    except gflags.FlagsError:
        print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
        sys.exit(1)
    
    if not os.path.exists(WAYPOINT_FILE):
        print(bcolors.WARNING, 'WAYPOINT_FILE ' + WAYPOINT_FILE + ' does NOT exist', bcolors.ENDC)
        print(bcolors.WARNING, 'Use EMPTY waypoint file instead!', bcolors.ENDC)
        WAYPOINT_FILE = 'waypoints/waypoint_empty.txt'
    
    planner_obj = ORACLEPlanner(WAYPOINT_FILE)
    planner_obj.run()