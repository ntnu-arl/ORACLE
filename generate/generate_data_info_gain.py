import os
import sys
sys.path.append('.')
sys.path.append('./env')
import rospy
import numpy as np
from scipy.spatial.transform import Rotation as R
import collections
import timeit

try:
   import cPickle as pickle
except:
   import pickle

from rotors_wrappers import RotorsWrappers
from utilities import bcolors
import gflags
from common_flags import FLAGS
from config import *

def s_double_integrator(t, a_0, v_0, p_0): # [position, velocity, acc]
    position = (a_0/2)*(t**2) + v_0*t + p_0
    velocity = a_0*t          + v_0
    acceleration = a_0
    return np.concatenate((position, velocity, acceleration), axis=0)

def calculate_trajectory_with_yaw_primitives(np_random, obs):
    # sample RELATIVE reference yaw (steering angle)
    HORIZONTAL_FOV_rad = np.deg2rad(HORIZONTAL_FOV)
    VERTICAL_FOV_rad = np.deg2rad(VERTICAL_FOV)
    ref_relative_yaw = np.float32(np_random.uniform(-HORIZONTAL_FOV_rad/2, HORIZONTAL_FOV_rad/2))

    # calculate absolute reference yaw
    r_robot = R.from_quat([obs[9], obs[10], obs[11], obs[12]])
    robot_euler_angles = r_robot.as_euler('xyz', degrees=False)
    ref_yaw = ref_relative_yaw + robot_euler_angles[2]
    
    # calculate current yaw-aligned forward speed
    # velocity_B = np.array([obs[3], obs[4], obs[5]])
    # velocity_W = R.from_euler('xyz', robot_euler_angles, degrees=False).as_matrix() @ (velocity_B)

    # sample reference forward velocity in yaw-aligned world frame
    vel_x = np.float32(np_random.uniform(0.0, MAX_CMD_APPEND[0]))
    ref_tilt_angle = -np.deg2rad(CAM_PITCH) - robot_euler_angles[1] + np.float32(
        np_random.uniform(-VERTICAL_FOV_rad/2, VERTICAL_FOV_rad/2))
    ref_tilt_angle = np.clip(ref_tilt_angle, -np.pi/2, np.pi/2)
    vel_z = vel_x * np.tan(ref_tilt_angle)
    vel_z = np.clip(vel_z, MIN_CMD_APPEND[2], MAX_CMD_APPEND[2])
    
    # calculate position, velocity (simplified model for plotting, ignoring attitude dynamics)
    state = collections.deque([])
    cmd_execute = collections.deque([])
    cmd_evaluate = collections.deque([]) 
    a_0 = np.array([0.0, 0.0, 0.0])
    v_0 = np.array([vel_x, 0.0, vel_z])
    v_0 = R.from_euler('z', ref_relative_yaw, degrees=False).as_matrix() @ (v_0) # in initial yaw-aligned world frame, hence use ref_relative_yaw, not ref_yaw
    p_0 = 0
    t = 0.0
    for i in range(STEPS_TO_REGENERATE):
        t += DEPTH_TS
        state.appendleft(s_double_integrator(t, a_0, v_0, p_0))
        cmd_evaluate.appendleft(np.array([vel_x, vel_z, ref_relative_yaw])) # yaw-aligned world frame
        cmd_execute.appendleft(np.array([vel_x, 0.0, vel_z, ref_yaw]))
    return state, cmd_execute, cmd_evaluate, True

if __name__ == '__main__':
    try:
        argv = FLAGS(sys.argv)  # parse flags
    except gflags.FlagsError:
        print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
        sys.exit(1)

    save_path = FLAGS.save_path
    if not os.path.exists(save_path):
       os.mkdir(save_path) 

    # Initialize
    np_random = np.random.default_rng()
    env = RotorsWrappers()
    env.change_environment()
    obs = env.reset() # obs: [x, y, z, vx, vy, vz, ax, ay, az, qx, qy, qz, qw, wx, wy, wz]

    itr = 0
    episode = 0 
    pickle_counter = 0
    step = 0

    #Dict: episode is key 
    obs_all          = {}
    action_all       = {}
    di_all           = {}
    action_index_all = {}
    collision_all    = {}

    obs_episode           = []
    action_episode        = []
    di_episode            = []
    action_index_episode  = []
    collision_episode     = 0

    start = timeit.default_timer()
    print('Collecting data...')
    step_cnt = 0
    receive_first_key_position = False
    
    while(itr < FLAGS.NUM_EPISODES):
        #rospy.sleep(0.0001)
        robot_state, current_di, valid_obs = env.get_new_obs()
        # wait until new sensor data is available
        while valid_obs == False:
            rospy.sleep(0.0001)
            robot_state, current_di, valid_obs = env.get_new_obs()

        if (not receive_first_key_position):
            key_position = robot_state[0:3]
            receive_first_key_position = True
        
        not_valid_count = 0
        # pick a random end waypoint in depth camera FOV and calculate smooth state trajectory to reach it
        if (step_cnt == 0):
            valid_trajectory = False
            env.pause()
            while (not valid_trajectory):
                trajectory, cmd, _, valid_trajectory = calculate_trajectory_with_yaw_primitives(np_random, robot_state)
                if (not valid_trajectory):
                    if not_valid_count > FLAGS.INVALID_COUNT_LIMIT:
                        print("reset environment: cannot generate valid trajectory")
                        robot_state, current_di, valid_obs = env.reset()
                        # wait until new sensor data is available
                        while valid_obs == False:
                            rospy.sleep(0.0001)
                            robot_state, current_di, valid_obs = env.get_new_obs()
                        not_valid_count = 0
                        step_cnt = 0
                    else:
                        not_valid_count += 1
        env.unpause()
        
        # DEBUG
        r_robot = R.from_quat([robot_state[9], robot_state[10], robot_state[11], robot_state[12]])
        robot_euler_angles = r_robot.as_euler('xyz', degrees=False) # XYZ IS THE CORRECT ORDER (from W, yaw first -> pitch -> roll) IN scipy.spatial.transform Rotation!!!
        #print('current robot_euler_angles:', robot_euler_angles)
        velocity_B = np.array([robot_state[3], robot_state[4], robot_state[5]])
        #velocity_W = R.from_euler('z', robot_euler_angles[2], degrees=False).as_matrix() @ (velocity_B)
        velocity_W = R.from_euler('xyz', robot_euler_angles, degrees=False).as_matrix() @ (velocity_B)
        #print('velocity_B:', velocity_B)
        #print('velocity_W:', velocity_W)

        action = [[cmd[step_cnt-1][0], cmd[step_cnt-1][1], cmd[step_cnt-1][2], cmd[step_cnt-1][3]]]
        #print('action:', action)
        if (step_cnt % SKIP_STEP_GENERATE == 0):
            done, info = env.step(action)

        # Reshape to network format before saving
        current_di = current_di.reshape(current_di.shape[0] , current_di.shape[1], 1)

        # Check if episode collided
        if info['status'] == 'collide':
            print('STATUS collide')
            collision_episode = 1

        if (step_cnt % SKIP_STEP_GENERATE == 0):
            # save data when the robot moves ... meters or collision happens
            if collision_episode or (step_cnt == 0) or (receive_first_key_position and (np.linalg.norm(key_position - robot_state[0:3]) > THRESHOLD_DISTANCE)): 
                key_position = robot_state[0:3]
                di_episode.append(current_di) # https://towardsdatascience.com/python-lists-are-sometimes-much-faster-than-numpy-heres-a-proof-4b3dad4653ad
                action_index_episode.append(step)
            obs_episode.append(robot_state)
            action_episode.append(action)
            step += 1

        step_cnt -= 1
        if (step_cnt == -STEPS_TO_REGENERATE): # regenerate after ... steps
            step_cnt = 0

        # Add to episode if done or last

        if done:
            obs_all[episode]          = obs_episode
            di_all[episode]           = di_episode
            action_all[episode]       = action_episode  
            action_index_all[episode] = action_index_episode
            collision_all[episode]    = collision_episode

            obs_episode           = []
            action_episode        = []
            di_episode            = []
            action_index_episode  = []
            collision_episode     = 0

            episode += 1
            itr += 1

            print(bcolors.OKBLUE + "Begin episode " + str(itr) + bcolors.ENDC)
            step = 0
            step_cnt = 0
            receive_first_key_position = False

            if itr % FLAGS.RANDOMIZE_WORLD == 0: 
                print(bcolors.OKGREEN + "Changing the environment" + bcolors.ENDC)
                env.change_environment()
            ## Pickle 
            if episode == FLAGS.PICKLE_SIZE or itr == FLAGS.NUM_EPISODES:
                print("PICKELING ", str(episode), "EPISODES")
                env.pause()

                pickle.dump(obs_all, open(save_path + "/obs_dump"+str(pickle_counter)+".p","wb"))
                pickle.dump(di_all, open(save_path + "/di_dump"+str(pickle_counter)+".p","wb"))
                pickle.dump(action_all, open(save_path + "/action_dump"+str(pickle_counter)+".p","wb"))
                pickle.dump(action_index_all, open(save_path + "/action_index_dump"+str(pickle_counter)+".p","wb"))
                pickle.dump(collision_all, open(save_path + "/collision_dump"+str(pickle_counter)+".p","wb"))
                
                pickle_counter += 1
                episode = 0

                obs_all          = {}
                di_all           = {}
                action_all       = {}
                action_index_all = {}
                collision_all    = {}

                env.unpause()
            obs = env.reset()


    print('Collecting time: ', timeit.default_timer() - start)

    env.pause()