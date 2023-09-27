import sys
sys.path.append('.')
import time
import rospy
import numpy as np
from scipy.spatial.transform import Rotation as R
import collections
import tf # follow README to build geometry and geometry2 packages with python3 (ONLY for ROS versions < Noetic)
from cv_bridge import CvBridge # follow README to build vision_opencv package with python3 (ONLY for ROS versions < Noetic)
from mav_msgs.msg import RateThrust
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Imu, PointCloud2
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose, Twist
from gazebo_msgs.msg import ContactsState, ModelState
from std_srvs.srv import Empty, EmptyRequest, EmptyResponse
from lmf_planner_srvs_msgs.srv import set_goal_dir, set_goal_dirResponse
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from config import *
if PLANNING_TYPE == 3:
    from voxblox_msgs.srv import InfoGainBaseline
from utilities import create_point_cloud
import cv2

if EVALUATE_MODE == False:
    EPISODE_TIMEOUT = 30 # seconds    

class RotorsWrappers:
    def __init__(self):
        rospy.init_node('rotors_wrapper', anonymous=True)

        self.current_goal = None
        self.get_params()

        self.done = False
        self.timeout = False
        self.timeout_timer = None
        self.timeout_length = EPISODE_TIMEOUT

        self.robot_odom = collections.deque(maxlen=ODOM_QUEUE_LEN)
        self.imu = collections.deque(maxlen=ODOM_QUEUE_LEN)
        # Saves last ODOM_QUEUE_LEN images
        self.di_list = collections.deque(maxlen=DI_QUEUE_LEN)
        self.di_stamp = collections.deque(maxlen=DI_QUEUE_LEN)
        # self.camera_list = collections.deque([])
        self.mask_queue = collections.deque(maxlen=DI_QUEUE_LEN)
        if (EVALUATE_MODE == True) and (PLANNING_TYPE == 1): # when evaluating seVAE
            self.latent_list = collections.deque(maxlen=DI_QUEUE_LEN)
        self.di_is_new = False
        self.bridge = CvBridge()

        self.seed()        

        # TF broadcaster
        self.br = tf.TransformBroadcaster()
        self.odom_timestamp = rospy.Time.now()

        # ROS publishers/subcribers
        self.contact_subcriber = rospy.Subscriber("/delta/delta_contact", ContactsState, self.contact_callback)
        self.contact_collision_check_subcriber = rospy.Subscriber("/delta_collision_check/delta_collision_check_contact", ContactsState, self.contact_collision_check_callback)
        self.odom_subscriber = rospy.Subscriber(SIM_ODOM_TOPIC, Odometry, self.odom_callback)
        if (EVALUATE_MODE == True) and (PLANNING_TYPE >= 2): # when evaluating infogain model
            self.pcl_with_interestingness_publisher = rospy.Publisher("/pcl_with_interestingness", PointCloud2)
            if SIM_USE_GRAYSCALE_FILTER:
                # /delta/rgbd/camera_depth/camera/image_raw: mono8
                # /delta/rgbd/camera_rgb/image_raw: bgr8
                self.mask_publisher = rospy.Publisher(SIM_MASK_TOPIC, Image)
                self.mask_image_subscriber = rospy.Subscriber(SIM_MASK_TOPIC, Image, self.mask_callback, queue_size=1)
                self.grayscale_image_subscriber = rospy.Subscriber(SIM_GRAYSCALE_TOPIC, Image, self.grayscale_img_callback, queue_size=1)
            else:
                self.mask_image_subscriber = rospy.Subscriber(SIM_MASK_TOPIC, Image, self.mask_callback, queue_size=1)
        if (EVALUATE_MODE == True) and (PLANNING_TYPE == 1): # when evaluating seVAE
            self.latent_subscriber = rospy.Subscriber(SIM_LATENT_TOPIC, Float32MultiArray, self.latent_callback, queue_size=1)
        else:
            self.depth_image_subscriber = rospy.Subscriber(SIM_DEPTH_TOPIC, Image, self.depth_img_callback, queue_size=1)
        self.imu_subscriber = rospy.Subscriber(SIM_IMU_TOPIC, Imu, self.imu_callback)

        self.goal_training_publisher = rospy.Publisher("/delta/goal_training", Pose)
        self.goal_in_vehicle_publisher = rospy.Publisher("/delta/goal_in_vehicle", Odometry) # for debug
        self.goal_init_publisher = rospy.Publisher("/delta/goal", Pose)
        self.goal_robot2_init_publisher = rospy.Publisher("/delta_collision_check/goal", Pose)
        self.trajectory_publisher = rospy.Publisher(TRAJECTORY_TOPIC, MarkerArray)
        self.cmd_publisher = rospy.Publisher("/delta/command/rate_thrust", RateThrust)
        self.cmd_vel_publisher = rospy.Publisher(SIM_CMD_TOPIC, Twist)
        self.model_state_publisher = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        self.sphere_marker_pub = rospy.Publisher('goal_published',
                                                 MarkerArray,
                                                 queue_size=1)

        self.pause_physics_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_physics_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
 
        self.reset_controller = rospy.ServiceProxy('/delta/pid_reset', Empty)
        if PLANNING_TYPE == 3:
            self.calc_info_gain_baseline = rospy.ServiceProxy('/voxblox_node/baseline_info_gain', InfoGainBaseline)
            self.clear_voxblox_map = rospy.ServiceProxy('/voxblox_node/clear_map', Empty)

        # service
        rospy.Service('start_planner', Empty, self.start_planner)
        rospy.Service('stop_planner', Empty, self.stop_planner)
        rospy.Service('set_goal_dir', set_goal_dir, self.set_goal_dir)

        self.collide = False
        self.collide2 = False 

        self.robot1_pose = Pose()
        self.robot1_pose.position.x = 0.0
        self.robot1_pose.position.y = self.max_wp_y + 10.0
        self.robot1_pose.position.z = self.max_wp_z + 50.0
        self.robot1_pose.orientation.x = 0
        self.robot1_pose.orientation.y = 0
        self.robot1_pose.orientation.z = 0
        self.robot1_pose.orientation.w = 1

        self.robot2_pose = Pose()
        self.robot2_pose.position.x = self.min_wp_x - 50.0
        self.robot2_pose.position.y = self.max_wp_y + 50.0
        self.robot2_pose.position.z = self.max_wp_z + 50.0
        self.robot2_pose.orientation.x = 0
        self.robot2_pose.orientation.y = 0
        self.robot2_pose.orientation.z = 0
        self.robot2_pose.orientation.w = 1     

        self.zero_goal = Pose()
        self.zero_goal.position.x = 0.0
        self.zero_goal.position.y = 0.0
        self.zero_goal.position.z = 0.0
        self.zero_goal.orientation.x = 0
        self.zero_goal.orientation.y = 0
        self.zero_goal.orientation.z = 0
        self.zero_goal.orientation.w = 1              

        self.spawn_robot2(self.robot2_pose)
        self.goal_robot2_init_publisher.publish(self.robot2_pose)

        # self.reset()

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)

    def get_params(self):
        self.initial_goal_generation_radius = rospy.get_param('initial_goal_generation_radius', 2.0)
        self.set_goal_generation_radius(self.initial_goal_generation_radius)
        self.waypoint_radius = rospy.get_param('waypoint_radius', 0.5)
        self.robot_collision_frame = rospy.get_param(
            'robot_collision_frame',
            'delta::delta/base_link::delta/base_link_fixed_joint_lump__delta_collision_collision'
        )
        self.robot2_collision_frame = rospy.get_param(
            'robot2_collision_frame',
            'delta_collision_check::delta_collision_check/base_link::delta_collision_check/base_link_fixed_joint_lump__delta_collision_check_collision_collision'
        )
        self.ground_collision_frame = rospy.get_param(
            'ground_collision_frame', 'ground_plane::link::collision')

        if EVALUATE_MODE:
            self.max_wp_x = rospy.get_param('max_waypoint_x', MAX_INITIAL_X)
            self.max_wp_y = rospy.get_param('max_waypoint_y', MAX_INITIAL_Y)
            self.max_wp_z = rospy.get_param('max_waypoint_z', MAX_INITIAL_Z) 

            self.min_wp_x = rospy.get_param('min_waypoint_x', MIN_INITIAL_X)
            self.min_wp_y = rospy.get_param('min_waypoint_y', MIN_INITIAL_Y)
            self.min_wp_z = rospy.get_param('min_waypoint_z', MIN_INITIAL_Z)
        else:    
            self.max_wp_x = rospy.get_param('max_waypoint_x', 40.0)
            self.max_wp_y = rospy.get_param('max_waypoint_y', 40.0)
            self.max_wp_z = rospy.get_param('max_waypoint_z', 8.0) 

            self.min_wp_x = rospy.get_param('min_waypoint_x', 0.0)
            self.min_wp_y = rospy.get_param('min_waypoint_y', 0.0)
            self.min_wp_z = rospy.get_param('min_waypoint_z', 1.0)

    def step(self, action):
        if len(action) == 3: # acc cmd
            command = RateThrust()
            command.header.stamp = rospy.Time.now()
            command.angular_rates.x = 0.0
            command.angular_rates.y = 0.0
            command.angular_rates.z = 0.0
            command.thrust.x = action[0][0]
            command.thrust.y = action[0][1]
            command.thrust.z = action[0][2]
            actions = np.array([command.thrust.x, command.thrust.y, command.thrust.z])
            # print('action:', action)
            self.cmd_publisher.publish(command)
        else: # have yaw_rate (or yaw) cmd
            command = Twist()
            command.linear.x = action[0][0]
            command.linear.y = action[0][1]
            command.linear.z = action[0][2]
            command.angular.x = 0.0
            command.angular.y = 0.0
            command.angular.z = action[0][3]
            self.cmd_vel_publisher.publish(command)

        info = {'status':'none'}
        self.done = False  

        # collide?
        if self.collide:
            self.done = True
            info = {'status':'collide'} 
            self.collide = False  

        # time out?
        if self.timeout:
            self.timeout = False
            self.done = True
            #print('timeout')
            info = {'status':'timeout'}

        # check limit of env
        # current_pos = self.robot_odom[0].pose.pose.position
        # if (current_pos.z > 10.0):
        #     self.done = True
        #     info = {'status':'out_of_env'}

        return (self.done, info)

    def yaw_in_spot(self, goal_dir):
        # yawing in one spot
        command = Twist()
        command.linear.x = 0.0
        command.linear.y = 0.0
        command.linear.z = 0.0
        command.angular.x = 0.0
        command.angular.y = 0.0
        command.angular.z = goal_dir
        self.cmd_vel_publisher.publish(command)

    def get_new_obs(self):
        if (len(self.robot_odom) > 0) & (len(self.imu) > 0) & self.di_is_new:
            self.di_is_new = False
            if (EVALUATE_MODE == True) and (PLANNING_TYPE == 1):
                current_di = self.latent_list[0]
            else:
                current_di = self.di_list[0]
            if EVALUATE_MODE:
                if len(self.mask_queue) > 0:
                    current_mask = self.mask_queue[0]
                else:
                    current_mask = None

            current_odom = self.robot_odom[0]
            linear_acc = self.imu[0].linear_acceleration
            linear_acc_np = np.array([linear_acc.x, linear_acc.y, linear_acc.z])

            robot_pose = current_odom.pose.pose
            # r_robot = R.from_quat([robot_pose.orientation.x, robot_pose.orientation.y, robot_pose.orientation.z, robot_pose.orientation.w])
            # robot_euler_angles = r_robot.as_euler('xyz', degrees=False)
            #linear_acc_np = r_robot.inv().apply(linear_acc_np)
            
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
            -linear_acc_np[0], # x,y accel is inverted for some reasons
            -linear_acc_np[1],
            linear_acc_np[2] - 9.81,
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

    def odom_callback(self, msg):
        #print("received odom msg")
        self.robot_odom.appendleft(msg)
        if (len(self.robot_odom) > ODOM_QUEUE_LEN): # save the last ODOM_QUEUE_LEN odom msg
            self.robot_odom.pop()
        
        # publish vehicle frame
        if EVALUATE_MODE == True:
            time_diff = np.abs((msg.header.stamp - self.odom_timestamp).to_sec())
            if (time_diff >= 0.05): # publish at 20 Hz 
                self.odom_timestamp = msg.header.stamp
                odom_quat = msg.pose.pose.orientation
                r_robot = R.from_quat([odom_quat.x, odom_quat.y, odom_quat.z, odom_quat.w])
                robot_euler_angles = r_robot.as_euler('xyz', degrees=False)
                self.br.sendTransform((0.0, 0.0, 0.0),
                                tf.transformations.quaternion_from_euler(-robot_euler_angles[0], -robot_euler_angles[1], 0),
                                msg.header.stamp,
                                "/vehicle",
                                "/delta/odometry_sensor1")

    def imu_callback(self, msg):
        #print("received imu msg")
        self.imu.appendleft(msg)
        if (len(self.imu) > ODOM_QUEUE_LEN): # save the last ODOM_QUEUE_LEN imu msg
            self.imu.pop()

    def latent_callback(self, data):
        self.latent_list.appendleft(data.data)
        # print('data.data:', data.data)
        self.di_is_new = True

    def depth_img_callback(self, data):
        try:
            # https://github.com/ros-perception/vision_opencv/blob/b6842b061da413823c8a01289ff8ba353c857ada/cv_bridge/python/cv_bridge/core.py
            img = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough') # use np.ndarray internally
        except Exception as e:
            print(e)
            return
        img = np.copy(img)
        
        # remove nan, normalize [0,255] and cast to uint8
        np.clip(img, 0, MAX_RANGE, out=img) # this doesn't change NaN values

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

    def grayscale_img_callback(self, data):
        try:
            grayscale_img = self.bridge.imgmsg_to_cv2(data, desired_encoding='mono8')
        except Exception as e:
            print(e)
            pass
        grayscale_img = np.copy(grayscale_img).astype('float32')
        
        mask = cv2.inRange(grayscale_img, MIN_PIXEL_VALUE, MAX_PIXEL_VALUE) / 255  # scale to 0->1 range

        if (self.mask_publisher.get_num_connections() > 0):
            mask_msg = Image()
            mask_msg.header.stamp = data.header.stamp
            mask_msg.height = DI_SHAPE[0]
            mask_msg.width = DI_SHAPE[1]
            mask_msg.encoding = '8UC1'
            mask_msg.is_bigendian = 0
            mask_msg.step = DI_SHAPE[1] # 1 byte for each pixel
            mask_msg.data = np.reshape(np.array((mask*255).astype('uint8')), (DI_SHAPE[0] * DI_SHAPE[1],)).tolist()
            self.mask_publisher.publish(mask_msg)

    def mask_callback(self, data):
        mask = self.mask_msg_to_img(data)
        self.mask_queue.appendleft(mask)
        if (len(self.di_list) > 0):
            # publish pcl with intensity field as interestingness level for evaluation with voxblox
            # only for evaluation
            time_diff = np.abs((data.header.stamp - self.di_stamp[0]).to_sec())
            if (time_diff < 0.04): # 0.2 for yolo in sim
                self.publish_pcl_with_interestingness(self.di_list[0], mask, self.di_stamp[0])

    def publish_pcl_with_interestingness(self, di_current, mask, time_stamp = None):
        pixel_idx = np.indices((DI_SHAPE[0], DI_SHAPE[1]))
        z = di_current.ravel()
        x = (pixel_idx[1].ravel() - DEPTH_CX) * z / DEPTH_FX
        y = (pixel_idx[0].ravel() - DEPTH_CY) * z / DEPTH_FY   
        valid_idx = np.where(z < MAX_METRICS_RANGE) # only counts voxels within ... z range
        z_valid = z[valid_idx]
        x_valid = x[valid_idx]
        y_valid = y[valid_idx]
        mask_valid = mask.ravel()[valid_idx]

        p_C = np.vstack((x_valid, y_valid, z_valid))
        p_B = t_BC + R_BC @ p_C  # convert to body frame: delta/odometry_sensor1
        #p_W = R_WB_0 @ p_B  + np.expand_dims(p_WB_0, axis=1)
        pcl_with_interestingness = np.vstack((p_B, mask_valid))
        pcl_msg = create_point_cloud(pcl_with_interestingness.T, 'delta/odometry_sensor1', time_stamp)
        self.pcl_with_interestingness_publisher.publish(pcl_msg)

    def contact_callback(self, msg):
        # Check inside the models states for robot's contact state
        for i in range(len(msg.states)):
            if (msg.states[i].collision1_name == self.robot_collision_frame):
                #print('Contact found!')
                rospy.logdebug('Contact found!')
                self.collide = True
                if (msg.states[i].collision2_name ==
                        self.ground_collision_frame):
                    rospy.logdebug('Robot colliding with the ground')
                else:
                    rospy.logdebug(
                        'Robot colliding with something else (not ground)')
                    #self.reset()
            else:
                rospy.logdebug('Contact not found yet ...')

    def contact_collision_check_callback(self, msg):
        # Check inside the models states for robot's contact state
        for i in range(len(msg.states)):
            if (msg.states[i].collision1_name == self.robot2_collision_frame):
                print('Contact robot2 found!')
                rospy.logdebug('Contact robot2 found!')
                self.collide2 = True
                if (msg.states[i].collision2_name ==
                        self.ground_collision_frame):
                    rospy.logdebug('Robot2 colliding with the ground')
                else:
                    rospy.logdebug(
                        'Robot2 colliding with something else (not ground)')
                    #self.reset()
            else:
                rospy.logdebug('Contact robot2 not found yet ...')                

    def register_start_cb(self, start_cb):
        self.start_cb = start_cb
    
    def start_planner(self, req):
        self.start_cb()
        return EmptyResponse()

    def register_stop_cb(self, stop_cb):
        self.stop_cb = stop_cb

    def stop_planner(self, req):
        self.stop_cb()
        return EmptyResponse() 

    def register_goal_cb(self, goal_cb):
        self.goal_cb = goal_cb

    def set_goal_dir(self, req):
        response = set_goal_dirResponse()
        response.success = self.goal_cb(req.goal_dir)
        return response

    # Input:    robot_odom  : Odometry()
    #           goal        : Pose(), in vehicle frame
    # Return:   current_goal  : Pose(), in world frame
    def transform_goal_to_world_frame(self, robot_odom, goal):
        current_goal = Pose()
        
        r_goal = R.from_quat([goal.orientation.x, goal.orientation.y, goal.orientation.z, goal.orientation.w])
        goal_euler_angles = r_goal.as_euler('zyx', degrees=False)
        
        robot_pose = robot_odom.pose.pose
        r_robot = R.from_quat([robot_pose.orientation.x, robot_pose.orientation.y, robot_pose.orientation.z, robot_pose.orientation.w])
        robot_euler_angles = r_robot.as_euler('zyx', degrees=False)

        r_goal_in_world = R.from_euler('z', goal_euler_angles[0] + robot_euler_angles[0], degrees=False)
        goal_pos_in_vehicle = np.array([goal.position.x, goal.position.y, goal.position.z])
        robot_pos = np.array([robot_pose.position.x, robot_pose.position.y, robot_pose.position.z])
        goal_pos_in_world = R.from_euler('z', robot_euler_angles[0], degrees=False).as_matrix().dot(goal_pos_in_vehicle) + robot_pos
        # print('R abc:', R.from_euler('z', robot_euler_angles[0], degrees=False).as_matrix())
        # print('goal_pos_in_vehicle:', goal_pos_in_vehicle)
        # print('robot_pos:', robot_pos)
        # print('goal_pos_in_world:', goal_pos_in_world)

        current_goal.position.x = goal_pos_in_world[0]
        current_goal.position.y = goal_pos_in_world[1]
        current_goal.position.z = goal_pos_in_world[2]

        current_goal_quat = r_goal_in_world.as_quat()
        current_goal.orientation.x = current_goal_quat[0]
        current_goal.orientation.y = current_goal_quat[1]
        current_goal.orientation.z = current_goal_quat[2]
        current_goal.orientation.w = current_goal_quat[3]

        return current_goal

    # Input:    robot_odom  : Odometry()
    #           goal        : Pose(), in world frame
    # Return:   goal_odom   : Odometry(), in vehicle frame
    #           robot_euler_angles: np.array(), zyx order
    def transform_goal_to_vehicle_frame(self, robot_odom, goal):
        goal_odom = Odometry()
        
        r_goal = R.from_quat([goal.orientation.x, goal.orientation.y, goal.orientation.z, goal.orientation.w])
        goal_euler_angles = r_goal.as_euler('zyx', degrees=False)
        
        robot_pose = robot_odom.pose.pose
        r_robot = R.from_quat([robot_pose.orientation.x, robot_pose.orientation.y, robot_pose.orientation.z, robot_pose.orientation.w])
        robot_euler_angles = r_robot.as_euler('zyx', degrees=False)

        r_goal_in_vechile = R.from_euler('z', goal_euler_angles[0] - robot_euler_angles[0], degrees=False)
        goal_pos = np.array([goal.position.x, goal.position.y, goal.position.z])
        robot_pos = np.array([robot_pose.position.x, robot_pose.position.y, robot_pose.position.z])
        goal_pos_in_vehicle = R.from_euler('z', -robot_euler_angles[0], degrees=False).as_matrix().dot((goal_pos - robot_pos))
        # print('goal_pos:', goal_pos)
        # print('robot_pos:', robot_pos)
        # print('R:', R.from_euler('z', -robot_euler_angles[0], degrees=False).as_matrix())

        goal_odom.header.stamp = robot_odom.header.stamp
        goal_odom.header.frame_id = "/delta" #"vehicle_frame"
        goal_odom.pose.pose.position.x = goal_pos_in_vehicle[0]
        goal_odom.pose.pose.position.y = goal_pos_in_vehicle[1]
        goal_odom.pose.pose.position.z = goal_pos_in_vehicle[2]
        goal_quat_in_vehicle = r_goal_in_vechile.as_quat()
        goal_odom.pose.pose.orientation.x = goal_quat_in_vehicle[0]
        goal_odom.pose.pose.orientation.y = goal_quat_in_vehicle[1]
        goal_odom.pose.pose.orientation.z = goal_quat_in_vehicle[2]
        goal_odom.pose.pose.orientation.w = goal_quat_in_vehicle[3]

        goal_odom.twist.twist.linear.x = -robot_odom.twist.twist.linear.x
        goal_odom.twist.twist.linear.y = -robot_odom.twist.twist.linear.y
        goal_odom.twist.twist.linear.z = -robot_odom.twist.twist.linear.z
        goal_odom.twist.twist.angular.x = -robot_odom.twist.twist.angular.x
        goal_odom.twist.twist.angular.y = -robot_odom.twist.twist.angular.y
        goal_odom.twist.twist.angular.z = -robot_odom.twist.twist.angular.z

        self.goal_in_vehicle_publisher.publish(goal_odom)

        return goal_odom, robot_euler_angles

    # Input:    robot_pose  : Pose()
    # Return:   current_goal    : Pose(), in world frame
    #           r               : float
    def generate_new_goal(self, robot_pose): ## this will probably need to be changed? 
        # Generate and return a pose in the sphere centered at the robot frame with radius as the goal_generation_radius

        # https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly/5838055#5838055
        goal = Pose()
        # sphere_marker_array = MarkerArray()
        u = self.np_random.random()
        v = self.np_random.random()
        theta = u * 2.0 * np.pi
        phi = np.arccos(2.0 * v - 1.0)
        while np.isnan(phi):
            phi = np.arccos(2.0 * v - 1.0)
        r = self.goal_generation_radius
        # r = self.goal_generation_radius * np.cbrt(self.np_random.random())
        # if r < 3.0:
        #     r = 3.0
        sinTheta = np.sin(theta)
        cosTheta = np.cos(theta)
        sinPhi = np.sin(phi)
        cosPhi = np.cos(phi)
        x = r * sinPhi * cosTheta
        y = r * sinPhi * sinTheta
        z = r * cosPhi

        # limit z of goal
        [x, y, z] = np.clip([x,y,z], [self.min_wp_x - robot_pose.position.x, self.min_wp_y - robot_pose.position.y, self.min_wp_z - robot_pose.position.z],
                                    [self.max_wp_x - robot_pose.position.x, self.max_wp_y - robot_pose.position.y, self.max_wp_z - robot_pose.position.z])


        # rospy.loginfo_throttle(2, 'New Goal: (%.3f , %.3f , %.3f)', x, y, z)
        goal.position.x = x
        goal.position.y = y
        goal.position.z = z
        goal.orientation.x = 0
        goal.orientation.y = 0
        goal.orientation.z = 0
        goal.orientation.w = 1
        # Convert this goal into the world frame and set it as the current goal
        robot_odom = Odometry()
        robot_odom.pose.pose = robot_pose
        current_goal = self.transform_goal_to_world_frame(robot_odom, goal)

        return current_goal, r

    def draw_new_goal(self, p):
        markerArray = MarkerArray()
        count = 0
        MARKERS_MAX = 20
        marker = Marker()
        marker.header.frame_id = "world"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose = p

        #rospy.loginfo('Draw new goal: (%.3f , %.3f , %.3f)', p.position.x, p.position.y, p.position.z)

        # We add the new marker to the MarkerArray, removing the oldest
        # marker from it when necessary
        if (count > MARKERS_MAX):
            markerArray.markers.pop(0)

        markerArray.markers.append(marker)
        # Renumber the marker IDs
        id = 0
        for m in markerArray.markers:
            m.id = id
            id += 1

        # Publish the MarkerArray
        self.sphere_marker_pub.publish(markerArray)

        count += 1

    def mask_msg_to_img(self, msg):
        mask_img = np.ndarray((msg.height, msg.width), 'B', msg.data, 0)

        # cv2.imshow("mask img", mask_img)
        # cv2.waitKey(1)

        mask_img = mask_img.astype('float32') / 255 # in range 0->1

        return mask_img

    def visualize_trajectory(self, trajectory_lib, collision_score_combined, collision_score_timestamp=np.array([]), best_indx=None, worst_indx=None, safe_indx=np.array([])):
        if self.trajectory_publisher.get_num_connections() > 0:
            marker_array = MarkerArray()
            id = 0
            for j in range(NUM_SEQUENCE_TO_EVALUATE):
                if VISUALIZATION_MODE == 0:
                    points = trajectory_lib[j]
                    # for i in np.array([1,points.shape[0]-1]): # reduce number of markers
                    for i in np.array([points.shape[0]-1]):
                        marker = Marker()
                        marker.id = id
                        id = id + 1
                        marker.header.frame_id =  "vehicle"
                        marker.type = marker.SPHERE
                        marker.action = marker.ADD
                        if j == best_indx: 
                            marker.scale.x = 0.3
                            marker.scale.y = 0.3
                            marker.scale.z = 0.3

                            marker.color.a = 1.0
                            marker.color.r = 0.0
                            marker.color.g = 0.0
                            marker.color.b = 1.0
                        elif j in safe_indx:
                            marker.scale.x = 0.3
                            marker.scale.y = 0.3
                            marker.scale.z = 0.3                            
                            
                            marker.color.a = 1.0
                            marker.color.r = 0.0
                            marker.color.g = 1.0
                            marker.color.b = 0.0                        
                        else:
                            marker.scale.x = 0.1
                            marker.scale.y = 0.1
                            marker.scale.z = 0.1
                            
                            marker.color.a = 1.0
                            marker.color.r = 1.0
                            marker.color.g = 0.647 # orange
                            marker.color.b = 0.0

                        pose = Pose()
                        pose.position.x = points[i, 0]
                        pose.position.y = points[i, 1]
                        pose.position.z = points[i, 2]
                        pose.orientation.x = 0
                        pose.orientation.y = 0
                        pose.orientation.z = 0
                        pose.orientation.w = 1

                        marker.pose = pose
                        marker.text = str(collision_score_combined[j,0])
                        marker_array.markers.append(marker)
                elif VISUALIZATION_MODE == 1:
                    points = trajectory_lib[j]
                    collision_scores = collision_score_timestamp[j]
                    for i in range(points.shape[0]):
                        collision_score = collision_scores[i]
                        marker = Marker()
                        marker.id = id
                        id = id + 1
                        marker.header.frame_id = "vehicle"
                        marker.type = marker.SPHERE
                        marker.action = marker.ADD
                        marker.scale.x = 0.2
                        marker.scale.y = 0.2
                        marker.scale.z = 0.2
                        if collision_score < 0.1:
                            marker.color.a = 1.0
                            marker.color.r = 0.0
                            marker.color.g = 0.0
                            marker.color.b = 1.0
                        elif (collision_score >= 0.1) and (collision_score < 0.2): 
                            marker.color.a = 0.5
                            marker.color.r = 1.0
                            marker.color.g = 1.0
                            marker.color.b = 0.0
                        else:                
                            marker.color.a = 1.0
                            marker.color.r = 1.0
                            marker.color.g = 0.0
                            marker.color.b = 0.0
                        
                        pose = Pose()
                        pose.position.x = points[i, 0]
                        pose.position.y = points[i, 1]
                        pose.position.z = points[i, 2]
                        pose.orientation.x = 0
                        pose.orientation.y = 0
                        pose.orientation.z = 0
                        pose.orientation.w = 1

                        marker.pose = pose
                        # marker.text = str(collision_score_combined[j,0])
                        marker_array.markers.append(marker)
                elif VISUALIZATION_MODE == 3:
                    for k in range(N_E): # ensemble
                        points = trajectory_lib[k,j]
                        # for i in np.array([1,points.shape[0]-1]): # reduce number of markers
                        for i in np.array([points.shape[0]-1]):
                            marker = Marker()
                            marker.id = id
                            id = id + 1
                            marker.header.frame_id =  "vehicle"
                            marker.type = marker.SPHERE
                            marker.action = marker.ADD
                            if j == best_indx[k]:
                                marker.scale.x = 0.3
                                marker.scale.y = 0.3
                                marker.scale.z = 0.3

                                marker.color.a = 1.0
                                marker.color.r = 0.0
                                marker.color.g = 0.0
                                marker.color.b = 1.0
                            elif j in safe_indx[k]:
                                marker.scale.x = 0.3
                                marker.scale.y = 0.3
                                marker.scale.z = 0.3

                                marker.color.a = 1.0
                                marker.color.r = 0.0
                                marker.color.g = 1.0
                                marker.color.b = 0.0
                            else:
                                marker.scale.x = 0.1
                                marker.scale.y = 0.1
                                marker.scale.z = 0.1

                                marker.color.a = 1.0
                                marker.color.r = 1.0
                                marker.color.g = 0.647 # orange
                                marker.color.b = 0.0

                            pose = Pose()
                            pose.position.x = points[i, 0]
                            pose.position.y = points[i, 1]
                            pose.position.z = points[i, 2]
                            pose.orientation.x = 0
                            pose.orientation.y = 0
                            pose.orientation.z = 0
                            pose.orientation.w = 1

                            marker.pose = pose
                            # marker.text = str(collision_score_combined[j,0])
                            marker_array.markers.append(marker)
            # Publish the MarkerArray
            self.trajectory_publisher.publish(marker_array)

    def timer_callback(self, event):
        self.timeout = True

    def set_goal_generation_radius(self, radius):
        self.goal_generation_radius = radius

    def get_goal_generation_radius(self):
        return self.goal_generation_radius

    def pause(self):
        #rospy.loginfo('Pausing physics')
        self.pause_physics_proxy(EmptyRequest())

    def unpause(self):
        #rospy.loginfo('Unpausing physics')
        self.unpause_physics_proxy(EmptyRequest())

    def reset_low_level_controller(self):
        #rospy.loginfo('Pausing physics')
        self.reset_controller(EmptyRequest())

    def stop_robot(self):
        self.goal_init_publisher.publish(self.zero_goal)

    def reset(self):
        # put the robot somewhere outside of the valid region
        self.reset_low_level_controller() # the robot MUST stay still here
        self.spawn_robot(self.robot1_pose)
        # check if the start position collides with env

        start_pose, collide = self.spawn_robot2(None)
        while collide:
            #rospy.loginfo('INVALID start pose: (%.3f , %.3f , %.3f)', start_pose.position.x, start_pose.position.y, start_pose.position.z)
            start_pose, collide = self.spawn_robot2(None)

        #rospy.loginfo('New start pose: (%.3f , %.3f , %.3f)', start_pose.position.x, start_pose.position.y, start_pose.position.z)
        
        # # check if the end position collides with env
        goal, r = self.generate_new_goal(start_pose)
        # _, collide = self.spawn_robot2(goal)
        # while collide:
        #     #rospy.loginfo('INVALID end goal: (%.3f , %.3f , %.3f)', goal.position.x, goal.position.y, goal.position.z)
        #     goal, r = self.generate_new_goal(start_pose)
        #     _, collide = self.spawn_robot2(goal)
        
        #rospy.loginfo('New end goal: (%.3f , %.3f , %.3f)', goal.position.x, goal.position.y, goal.position.z)

        # put the robot2 somewhere
        self.spawn_robot2(self.robot2_pose)
        self.goal_robot2_init_publisher.publish(self.robot2_pose) # collision_check model is controlled in world frame

        # put the robot at the start pose
        self.spawn_robot(start_pose)
        self.reset_low_level_controller()

        self.current_goal = goal
        self.reset_low_level_controller()
        #self.goal_training_publisher.publish(goal)
        
        # maintain current pose in robot frame
        self.goal_init_publisher.publish(self.zero_goal) # delta model is controlled in vehicle frame
        self.reset_timer(self.timeout_length)

        return self.get_new_obs()

    # Input:    position  : Pose()
    # Return:   position  : Pose(), in world frame
    #           collide   : bool
    def spawn_robot(self, pose = None):
        #rospy.loginfo('Pausing physics')
        self.pause_physics_proxy(EmptyRequest())

        new_position = ModelState()
        new_position.model_name = 'delta'
        new_position.reference_frame = 'world'
        
        # Fill in the new position of the robot
        if (pose == None):
            # randomize initial position (TODO: angle?, velocity?)
            state_high = np.array([self.max_wp_x, self.max_wp_y, self.max_wp_z], dtype=np.float32)
            state_low = np.array([self.min_wp_x, self.min_wp_y, self.min_wp_z], dtype=np.float32)
            state_init = self.np_random.uniform(low=state_low, high=state_high, size=(3,))
            new_position.pose.position.x = state_init[0]
            new_position.pose.position.y = state_init[1]
            new_position.pose.position.z = state_init[2]
            if EVALUATE_MODE:
                yaw_init = self.np_random.uniform(low=np.deg2rad(MIN_INITIAL_YAW), high=np.deg2rad(MAX_INITIAL_YAW))
            else:
                yaw_init = self.np_random.uniform(low=-np.pi, high=np.pi)    
            quat_init = R.from_euler('z', yaw_init, degrees=False).as_quat()
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
        self.model_state_publisher.publish(new_position)

        self.timeout = False
        self.done = False

        if (self.timeout_timer != None):
            self.timeout_timer.shutdown()

        #rospy.loginfo('Unpausing physics')
        self.unpause_physics_proxy(EmptyRequest())

        self.robot_odom.clear()
        self.imu.clear()
        self.di_is_new = False

        rospy.sleep(0.1) # wait for robot to get new odometry
        while (len(self.robot_odom) == 0): 
            rospy.sleep(0.001)
            pass        
        self.collide = False
        return new_position.pose, self.collide

    def spawn_robot2(self, pose = None):
        #rospy.loginfo('Pausing physics')
        self.pause_physics_proxy(EmptyRequest())

        new_position = ModelState()
        new_position.model_name = 'delta_collision_check'
        new_position.reference_frame = 'world'
        
        # Fill in the new position of the robot
        if (pose == None):
            # randomize initial position (TODO: angle?, velocity?)
            state_high = np.array([self.max_wp_x, self.max_wp_y, self.max_wp_z], dtype=np.float32)
            state_low = np.array([self.min_wp_x, self.min_wp_y, self.min_wp_z], dtype=np.float32)
            state_init = self.np_random.uniform(low=state_low, high=state_high, size=(3,))
            new_position.pose.position.x = state_init[0]
            new_position.pose.position.y = state_init[1]
            new_position.pose.position.z = state_init[2]
            if EVALUATE_MODE:
                yaw_init = self.np_random.uniform(low=np.deg2rad(MIN_INITIAL_YAW), high=np.deg2rad(MAX_INITIAL_YAW))
            else:
                yaw_init = self.np_random.uniform(low=-np.pi, high=np.pi)
            quat_init = R.from_euler('z', yaw_init, degrees=False).as_quat()
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
        self.model_state_publisher.publish(new_position)

        self.collide2 = False

        #rospy.loginfo('Unpausing physics')
        self.unpause_physics_proxy(EmptyRequest())

        self.collide2 = False

        rospy.sleep(0.1)      
        return new_position.pose, self.collide2


    def randomize_position(self, obj_type, nb_random_elements, nb_elements):
        idx_range = self.np_random.choice(range(nb_elements), nb_random_elements, replace=False) # sampling without replacement
        for i in idx_range:
            new_position = ModelState()
            new_position.model_name = obj_type + '_' + str(i) 
            new_position.reference_frame = 'world'
            # randomize initial position
            state_high = np.array([self.max_wp_x + 1.0, self.max_wp_y + 1.0, 6.0], dtype=np.float32)
            state_low = np.array([self.min_wp_x - 1.0, self.min_wp_y - 1.0, 0.5], dtype=np.float32)
            state_init = self.np_random.uniform(low=state_low, high=state_high, size=(3,))
            new_position.pose.position.x = state_init[0]
            new_position.pose.position.y = state_init[1]
            if obj_type == 'unit_sphere':
                new_position.pose.position.z = state_init[2]
            else: 
                new_position.pose.position.z = 0.0
            # new_position.pose.position.z = state_init[2]
            new_position.pose.orientation.x = 0
            new_position.pose.orientation.y = 0
            angle = self.np_random.uniform(-np.pi, np.pi)
            new_position.pose.orientation.z = np.sin(angle/2)
            new_position.pose.orientation.w = np.cos(angle/2)

            self.model_state_publisher.publish(new_position)
            time.sleep(0.03) # need time between different publishments    
    
    def not_used_obj(self, obj_type, nb_elements):
        start_x = -10.0
        start_y = 10.0
        if obj_type == 'Simple Block':
            start_x = -10.0
        elif obj_type == 'Simple Pyramid':
            start_x = -20.0
        elif obj_type == 'Simple U':
            start_x = -30.0
        elif obj_type == 'Simple Stone':
            start_x = -40.0
        elif obj_type == 'unit_sphere':
            start_x = -50.0
        elif obj_type == 'chair':
            start_x = -60.0
        elif obj_type == 'table_low_poly':
            start_x = -70.0
        elif obj_type == 'grey_wall_obstacle':
            start_x = -80.0
        elif obj_type == 'tree':
            start_x = -90.0
        elif obj_type == 'thin_tree':
            start_x = -15.0
        elif obj_type == 'fence':
            start_x = -100.0
        elif obj_type == 'diagonal_no_side':
            start_x = -110.0                       
        elif obj_type == 'diagonal_no_side_cut':
            start_x = -120.0

        itr = 0
        for i in range(nb_elements):
            # if i == 18: ## two rows for the stones 
            #     start_x = -10
            #     itr = 0
            new_position = ModelState()
            new_position.model_name = obj_type + '_' + str(i) 
            new_position.reference_frame = 'world'
            # randomize initial position
            new_position.pose.position.x = start_x 
            # if obj_type == 'unit_sphere':
            #     new_position.pose.position.y = start_y + itr*6
            #     new_position.pose.position.z = 2.0

            # else: 
            #     new_position.pose.position.y = start_y + itr*2
            #     new_position.pose.position.z = 0
            new_position.pose.position.y = start_y + itr*3
            new_position.pose.position.z = 0
            new_position.pose.orientation.x = 0
            new_position.pose.orientation.y = 0
            new_position.pose.orientation.z = 0
            new_position.pose.orientation.w = 1
            self.model_state_publisher.publish(new_position)
            time.sleep(0.03) # need time between different publishments  
            itr += 1
  
    def change_environment(self):
        self.pause_physics_proxy(EmptyRequest())
        
        # number of objects in .world file (random_corridor.world)
        nb_blocks = 12
        nb_pyramids = 18
        nb_u = 12
        nb_stones = 49
        nb_spheres = 10
        nb_chairs = 30
        nb_tables = 30
        nb_walls = 10
        nb_trees = 8
        nb_fences = 25
        nb_diagonal_no_side = 10
        nb_diagonal_no_side_cut = 10

        nb_random_blocks    = self.np_random.integers(3, nb_blocks-5)
        nb_random_pyramids  = self.np_random.integers(10, nb_pyramids-6)
        nb_random_u         = self.np_random.integers(8, nb_u)
        nb_random_stones    = self.np_random.integers(20, nb_stones)
        nb_random_spheres = nb_spheres
        nb_random_chairs = nb_chairs
        nb_random_tables = nb_tables
        nb_random_walls = nb_walls
        nb_random_trees = nb_trees
        nb_random_fences = self.np_random.integers(15, nb_fences-5)
        nb_random_diagonal_no_side = self.np_random.integers(5, nb_diagonal_no_side)
        nb_random_diagonal_no_side_cut = self.np_random.integers(5, nb_diagonal_no_side_cut)

        # clear old objects in corridor
        self.not_used_obj('Simple Block', nb_blocks)
        self.not_used_obj('Simple Pyramid', nb_pyramids)
        self.not_used_obj('Simple U', nb_u)
        self.not_used_obj('Simple Stone', nb_stones)
        self.not_used_obj('unit_sphere', nb_spheres)
        self.not_used_obj('chair', nb_chairs)
        self.not_used_obj('table_low_poly', nb_tables)        
        self.not_used_obj('grey_wall_obstacle', nb_walls)
        self.not_used_obj('tree', nb_trees)
        self.not_used_obj('fence', nb_fences)
        self.not_used_obj('diagonal_no_side', nb_diagonal_no_side)
        self.not_used_obj('diagonal_no_side_cut', nb_diagonal_no_side_cut)        

        if PLANNING_TYPE == 1: # randomize thin obstacles for seVAE
            nb_thin_trees = 200
            nb_random_thin_trees = np.random.randint(25, nb_thin_trees-50)
            nb_random_fences = np.random.randint(5, nb_fences-10)
            self.not_used_obj('thin_tree', nb_thin_trees)
            self.randomize_position('thin_tree', nb_random_thin_trees, nb_thin_trees)

        # spawn random number in corridor
        self.randomize_position('Simple Block', nb_random_blocks, nb_blocks)
        self.randomize_position('Simple Pyramid', nb_random_pyramids, nb_pyramids)
        self.randomize_position('Simple U', nb_random_u, nb_u)
        self.randomize_position('Simple Stone', nb_random_stones, nb_stones)
        self.randomize_position('unit_sphere', nb_random_spheres, nb_spheres)
        self.randomize_position('chair', nb_random_chairs, nb_chairs)
        self.randomize_position('table_low_poly', nb_random_tables, nb_tables)
        self.randomize_position('grey_wall_obstacle', nb_random_walls, nb_walls)
        self.randomize_position('tree', nb_random_trees, nb_trees)
        self.randomize_position('fence', nb_random_fences, nb_fences)
        self.randomize_position('diagonal_no_side', nb_random_diagonal_no_side, nb_diagonal_no_side)
        self.randomize_position('diagonal_no_side_cut', nb_random_diagonal_no_side_cut, nb_diagonal_no_side_cut)

        self.unpause_physics_proxy(EmptyRequest())

    def reset_timer(self, time):
        #rospy.loginfo('Resetting the timeout timer')
        if (self.timeout_timer != None):
            self.timeout_timer.shutdown()
        # self.timeout_timer = rospy.Timer(rospy.Duration(self.goal_generation_radius * 5), self.timer_callback)
        if time <= 0:
            time = 1.0
        self.timeout_timer = rospy.Timer(rospy.Duration(time), self.timer_callback)

    def render(self):
        return None

    def close(self):
        pass

    def call_info_gain_baseline(self, robot_pose):
        return self.calc_info_gain_baseline(robot_pose).info_gain

    def clear_map(self):
        self.clear_voxblox_map()

if __name__ == '__main__':

    rospy.loginfo('Ready')
    rospy.spin()
