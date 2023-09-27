import sys
sys.path.append('.')
import os
from config import *
import numpy as np
import rospy
from utilities import create_point_cloud_xyz
# import cv2
from flightmare_common import place_quad_at_start, MessageHandler # replicate code from agile_autonomy
from scipy.spatial.transform import Rotation as R
# from sensor_msgs.msg import PointCloud2
from gazebo_msgs.msg import ModelState
from std_srvs.srv import EmptyRequest
from std_msgs.msg import Bool
import open3d as o3d

from sim_env_base_class import SimEnvBaseClass

class FlightmareWrappers(SimEnvBaseClass):
    def __init__(self):
        # self.pcl_publisher = rospy.Publisher("/pcl_flightmare", PointCloud2)
        # self.pcl_pub_cnt = 0
        # self.acc_W_pub = rospy.Publisher('acc_W', Float32MultiArray)
        super().__init__()
        
        self.timeout_length = EPISODE_TIMEOUT # seconds

        self.msg_handler = MessageHandler()
        rospy.sleep(1.0)
        self.msg_handler.publish_tree_spacing(SPACING)
        self.msg_handler.publish_obj_spacing(SPACING)
        
        self.outside_range = False

        # from PlannerBase.py
        self.maneuver_complete = False
        self.use_network = False
        self.net_initialized = False
        self.reference_initialized = False
        self.rollout_idx = 0
        self.odometry_used_for_inference = None
        self.time_prediction = None
        self.last_depth_received = rospy.Time.now()
        self.reference_progress = 0
        self.reference_len = 1
        
        # from PlannerLearning.py
        self.recorded_samples = 0
        self.pcd = None
        self.pcd_tree = None
        self.pc_min = None
        self.pc_max = None
        self.counter = 1000
        self.crashed = False
        self.exp_failed = False
        self.planner_succed = True
        self.data_pub = rospy.Publisher("/delta/agile_autonomy/start_flying", Bool,
                                        queue_size=1)  # Stop upon some condition
        self.fly_sub = rospy.Subscriber("/delta/agile_autonomy/start_flying", Bool,
                                        self.callback_fly, queue_size=1)  # Receive and fly
        self.planner_succed_sub = rospy.Subscriber("/test_primitive/completed_planning",
                                                  Bool, self.planner_succed_callback, queue_size=1)
        # self.success_subs = rospy.Subscriber("success_reset", Empty,
        #                                      self.callback_success_reset,
        #                                      queue_size=1)
        self.reset_metrics()
        # Check at 20Hz the collision
        self.timer_check = rospy.Timer(
            rospy.Duration(1. / 20.),
            self.check_task_progress)

    def planner_succed_callback(self, data):
        self.planner_succed = data.data

    def reset_metrics(self):
        self.metrics = {'number_crashes': 0,
                        'travelled_dist': 0,
                        'closest_distance': 1000}

    def reach_end_of_traj(self):
        return self.maneuver_complete

    def is_robot_collide(self):
        return self.collide

    def is_robot_outside(self):
        return self.outside_range

    def callback_success_reset(self):
        print("Received call to Clear Buffer and Restart Experiment")
        os.system("rosservice call /gazebo/pause_physics")
        self.rollout_idx += 1
        self.use_network = False
        self.pcd = None
        self.reference_initialized = False
        self.maneuver_complete = False
        self.counter = 1000
        self.pcd_tree = None
        self.start_time = None
        self.crashed = False
        self.exp_failed = False
        self.planner_succed = True
        # self.reset_queue() # already did
        self.reset_metrics()
        print("Resetting experiment")
        os.system("rosservice call /gazebo/unpause_physics")
        print('Done Reset')

    def publish_pcl(self, di_current, time_stamp = None):
        pixel_idx = np.indices((DI_SHAPE[0], DI_SHAPE[1]))
        z = di_current.ravel()
        x = (pixel_idx[1].ravel() - DEPTH_CX) * z / DEPTH_FX
        y = (pixel_idx[0].ravel() - DEPTH_CY) * z / DEPTH_FY
        valid_idx = np.where(z < MAX_RANGE) # only counts voxels within ... z range
        # valid_idx = np.where((z < MAX_RANGE) & (y > -0.5)) # only counts voxels within ... z range
        z_valid = z[valid_idx]
        x_valid = x[valid_idx]
        y_valid = y[valid_idx]
        p_C = np.vstack((x_valid, y_valid, z_valid))
        p_B = t_BC + R_BC @ p_C  # convert to body frame: delta/odometry_sensor1
        #p_W = R_WB_0 @ p_B  + np.expand_dims(p_WB_0, axis=1)
        pcl_msg = create_point_cloud_xyz(p_B.T, 'delta/odometry_sensor1', time_stamp)
        self.pcl_publisher.publish(pcl_msg)

    def depth_img_callback(self, data):
        # try:
        #     # https://github.com/ros-perception/vision_opencv/blob/b6842b061da413823c8a01289ff8ba353c857ada/cv_bridge/python/cv_bridge/core.py
        #     img = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough') # use np.ndarray internally
        # except Exception as e:
        #     print(e)
        #     return
        # # img = np.copy(img)
        # img = np.copy(img[5:275,:]) # get 270 rows in the middle, somehow unity crashed if we set the height to 270
        
        img = np.ndarray((data.height, data.width), '<H', data.data, 0)
        img = img[5:275,:]

        # print('np.shape(img):', np.shape(img))
        img = img.astype('float32') * 0.001 # convert pixel value to meter
        # print('np.max(img):', np.max(img))

        
        # remove nan, normalize [0,255] and cast to uint8
        np.clip(img, 0, MAX_RANGE, out=img) # this doesn't change NaN values
        img[img < 0.2] = MAX_RANGE # heuristically "remove" invalid pixels (can run fill_in_fast instead)

        # convert to uint8 image (0-255)
        # img[~np.isnan(img)] = img[~np.isnan(img)] * RANGE_SCALE_INV
        # img[np.isnan(img)] = 255
        # img = img.astype('uint8')

        # convert to float32 image (0-MAX_RANGE)
        img[np.isnan(img)] = MAX_RANGE
        img = img.astype('float32')

        arr = np.array(img)

        #collect DI_QUEUE_LEN most recent depth images
        self.di_list.appendleft(arr)
        self.di_stamp.appendleft(data.header.stamp)
        # if (len(self.di_list) > DI_QUEUE_LEN): # save the last DI_QUEUE_LEN depth image msg
        #     self.di_list.pop()
        #     self.di_stamp.pop()
        self.di_is_new = True

    def callback_fly(self, data):
        # If self.use_network is true, then trajectory is already loaded
        if data.data and (not self.use_network):
            # Load pointcloud and make kdtree out of it
            rollout_dir = os.path.join(EXPERT_FOLDER, # TODO: wtf is this
                                       sorted(os.listdir(EXPERT_FOLDER))[-1])
            pointcloud_fname = os.path.join(
                rollout_dir, "pointcloud-unity.ply")
            print("Reading pointcloud from %s" % pointcloud_fname)
            self.pcd = o3d.io.read_point_cloud(pointcloud_fname)
            self.pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)
            # get dimensions prom point cloud

            self.pc_max = self.pcd.get_max_bound()
            self.pc_min = self.pcd.get_min_bound()
            print("min max pointcloud")
            print(self.pc_max)
            print(self.pc_min)

            # Load the reference trajectory
            # if self.config.track_global_traj:
            #     traj_fname = os.path.join(rollout_dir, "ellipsoid_trajectory.csv")
            # else:
            #     traj_fname = os.path.join(rollout_dir, "reference_trajectory.csv")
            # print("Reading Trajectory from %s" % traj_fname)
            # self.load_trajectory(traj_fname)
            self.reference_initialized = True # we only fly when this flag is set (KDTree is created)
            # only enable network when KDTree and trajectory are ready

        # Might be here if you crash in less than a second.
        if self.maneuver_complete:
            return
        # If true, network should fly.
        # If false, maneuver is finished and network is off.
        # self.use_network = data.data and self.config.execute_nw_predictions
        if (not data.data):
            self.maneuver_complete = True
            # self.use_network = False

    def get_new_obs(self):
        if (len(self.robot_odom) > 0) & (len(self.imu) > 0) & self.di_is_new:
            self.di_is_new = False

            if (EVALUATE_MODE == True) and (PLANNING_TYPE == 1):
                current_di = self.latent_list[0]
            else:
                current_di = self.di_list[0]

            # self.pcl_pub_cnt = self.pcl_pub_cnt + 1
            # if self.pcl_pub_cnt == 3:
            #     self.publish_pcl(current_di, self.di_stamp[0])
            #     self.pcl_pub_cnt = 0

            if EVALUATE_MODE:
                if len(self.mask_queue) > 0:
                    current_mask = self.mask_queue[0]
                else:
                    current_mask = None

            current_odom = self.robot_odom[0]
            linear_acc = self.imu[0].linear_acceleration
            linear_acc_np_B = np.array([linear_acc.x, linear_acc.y, linear_acc.z])

            robot_pose = current_odom.pose.pose
            r_robot = R.from_quat([robot_pose.orientation.x, robot_pose.orientation.y, robot_pose.orientation.z, robot_pose.orientation.w])
            # robot_euler_angles = r_robot.as_euler('xyz', degrees=False) # should we use quat or DCM instead?
            linear_acc_np_W = r_robot.apply(linear_acc_np_B)
            # publish to check acc in W frame
            # acc_msg = Float32MultiArray()
            # acc_msg.data = linear_acc_np_W
            # self.acc_W_pub.publish(acc_msg)
            
            #print('quat:', robot_pose.orientation.x, robot_pose.orientation.y, robot_pose.orientation.z, robot_pose.orientation.w)
            #print('robot_euler_angles:', robot_euler_angles)
            #print('DCM:', r_robot.as_matrix())
            #print('linear_acc_np:', -linear_acc_np[0], -linear_acc_np[1], linear_acc_np[2] - 9.81)

            new_obs = np.array([
            robot_pose.position.x,
            robot_pose.position.y,
            robot_pose.position.z,    
            current_odom.twist.twist.linear.x,
            current_odom.twist.twist.linear.y,
            current_odom.twist.twist.linear.z,
            # robot_euler_angles[0], # roll [rad]
            # robot_euler_angles[1], # pitch [rad]
            # robot_euler_angles[2], # yaw [rad]
            -linear_acc_np_W[0], # x,y accel is inverted for some reasons
            -linear_acc_np_W[1],
            linear_acc_np_W[2] - 9.81,
            robot_pose.orientation.x,
            robot_pose.orientation.y,
            robot_pose.orientation.z,
            robot_pose.orientation.w,
            current_odom.twist.twist.angular.x,
            current_odom.twist.twist.angular.y,
            current_odom.twist.twist.angular.z]
            )

            valid_obs = True
        else:
            new_obs = None
            current_di = None
            if EVALUATE_MODE:
                current_mask = None
            valid_obs = False
        if EVALUATE_MODE:
            return new_obs, current_di, current_mask, valid_obs
        else:
            return new_obs, current_di, valid_obs

    def spawn_robot(self, pose = None):
        #rospy.loginfo('Pausing physics')
        self.pause_physics_proxy(EmptyRequest())

        new_position = ModelState()
        new_position.model_name = 'delta'
        new_position.reference_frame = 'world'
        
        # Fill in the new position of the robot
        # print('spawn robot')
        if (pose == None):
            new_position.pose.position.x = UNITY_START_POS[0]
            new_position.pose.position.y = UNITY_START_POS[1]
            new_position.pose.position.z = UNITY_START_POS[2] + TAKEOFF_HEIGHT
            quat_init = R.from_euler('z', UNITY_START_POS[3], degrees=False).as_quat()
            new_position.pose.orientation.x = quat_init[0]
            new_position.pose.orientation.y = quat_init[1]
            new_position.pose.orientation.z = quat_init[2]
            new_position.pose.orientation.w = quat_init[3]
        else:
            new_position.pose = pose

        # Fill in the new twist of the robot
        new_position.twist.linear.x = 0
        new_position.twist.linear.y = 0
        new_position.twist.linear.z = 0
        new_position.twist.angular.x = 0
        new_position.twist.angular.y = 0
        new_position.twist.angular.z = 0
        #rospy.loginfo('Placing robot')
        print('place robot')
        self.model_state_publisher.publish(new_position)

        self.timeout = False
        self.done = False

        if (self.timeout_timer != None):
            self.timeout_timer.shutdown()

        #rospy.loginfo('Unpausing physics')
        self.unpause_physics_proxy(EmptyRequest())

        self.robot_odom.clear()
        self.di_is_new = False

        rospy.sleep(0.1) # wait for robot to get new odometry
        while (len(self.robot_odom) == 0): 
            rospy.sleep(0.001)
            pass        
        self.collide = False
        self.outside_range = False
        return new_position.pose, self.collide

    def start_experiment(self, rollout_idx):
        # self.msg_handler.publish_reset() # this will just call the following command
        self.callback_success_reset()
        place_quad_at_start(self.msg_handler)
        print("Doing experiment {}".format(rollout_idx))
        # Save point_cloud
        self.msg_handler.publish_save_pc()

    def reset(self):
        self.reset_low_level_controller()
        self.spawn_robot(None) # put the robot at UNITY_START_POS + TAKE_OFF
        self.reset_low_level_controller()
        # maintain current pose in robot frame
        self.goal_init_publisher.publish(self.zero_goal) # delta model is controlled in vehicle frame

        # setup_sim(self.msg_handler, config=self.settings) # similar to spawn_robot(None)
        self.start_experiment(self.rollout_idx)
        
        # wait until we receive feedback /start_flying from agile_autonomy
        while not self.reference_initialized:
            rospy.sleep(0.1)
        print('reset done!')
        self.reset_timer(self.timeout_length)

        return self.get_new_obs()

    def step(self, action):
        self.done, info = super().step(action)

        # reach the goal?
        if (self.reach_end_of_traj()): # reach final waypoint?
            info = {'status':'reach final WP'}
            self.done = True

        # outside env's range?
        if self.outside_range:
            self.done = True
            info = {'status':'outside'} 
            self.outside_range = False

        return self.done, info

    def check_task_progress(self, _timer):
        # go here if there are problems with the generation of traj
        # No need to check anymore
        if self.maneuver_complete:
            return
        if not self.planner_succed:
            print("Stopping experiment because planner failed!")
            self.publish_stop_recording_msg()
            self.exp_failed = True
            return

        # check if pointcloud is ready
        if self.pcd is None or (self.pc_min is None):
            return
        # check if reference is ready
        if not self.reference_initialized:
            return

        # if (self.reference_progress / (self.reference_len)) > self.end_ref_percentage:
        #     print("It worked well. (Arrived at %d / %d)" % (self.reference_progress, self.reference_len))
        #     self.publish_stop_recording_msg()

        # check if crashed into something
        if (len(self.robot_odom) == 0):
            return
        self.odometry = self.robot_odom[0]
        quad_position = [self.odometry.pose.pose.position.x,
                         self.odometry.pose.pose.position.y,
                         self.odometry.pose.pose.position.z]

        # Check if crashed into ground or outside a box (check in z, x, y)
        if (quad_position[2] < self.pc_min[2]) or (quad_position[2] > self.pc_max[2]):
            self.collide = True
        if (quad_position[0] < self.pc_min[0]) or (quad_position[0] > self.pc_max[0]) or \
           (quad_position[1] < self.pc_min[1]) or (quad_position[1] > self.pc_max[1]):
            print("Stopping experiment because quadrotor outside allowed range!")
            print(quad_position)
            # self.publish_stop_recording_msg() # calling this will register the current trial as success!
            self.outside_range = True
            return
        # if self.reference_progress > 50: # first second used to warm up
        self.update_metrics(quad_position)

    def update_metrics(self, quad_position):
        # Meters until crash
        if self.metrics['number_crashes'] == 0:
            current_velocity = np.array([self.odometry.twist.twist.linear.x,
                                         self.odometry.twist.twist.linear.y,
                                         self.odometry.twist.twist.linear.z]).reshape((3, 1))
            travelled_dist = current_velocity * 1. / 20.  # frequency of update
            travelled_dist = np.linalg.norm(travelled_dist)  # integrate
            self.metrics['travelled_dist'] += travelled_dist

        if self.metrics['travelled_dist'] < 5.0:
            # no recording in the first 5 m due to transient
            return
        # Number of crashes per maneuver
        [_, __, dist_squared] = self.pcd_tree.search_knn_vector_3d(quad_position, 1)
        closest_distance = np.sqrt(dist_squared)[0]

        if self.metrics['closest_distance'] > closest_distance and self.metrics['number_crashes'] == 0:
            self.metrics['closest_distance'] = closest_distance

        if closest_distance < CRASHED_THR and (not self.crashed):
            # it crashed into something, stop recording. Will not consider a condition to break the experiment now
            print("Crashing into something!")
            self.metrics['number_crashes'] += 1
            self.crashed = True
            # uncomment if you want to stop after crash
            # self.publish_stop_recording_msg()

            self.collide = True # RECORD THE COLLISION EVENT!!!
        # make sure to not count double crashes
        if self.crashed and closest_distance > 1.5 * CRASHED_THR:
            self.crashed = False

    # in our implementation, calling this will register the current trial as success
    def publish_stop_recording_msg(self):
        # Send message to cpp side to stop recording data
        self.maneuver_complete = True
        self.use_network = False
        print("Giving a stop from python")
        msg = Bool()
        msg.data = False
        self.data_pub.publish(msg)

    def change_environment(self): # this is handled by agile_autonomy when self.msg_handler.publish_save_pc() is called
        pass

if __name__ == '__main__':

    rospy.loginfo('Ready')
    rospy.spin()