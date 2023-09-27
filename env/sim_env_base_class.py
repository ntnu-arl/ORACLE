import sys
sys.path.append('.')
from config import *
from utilities import create_point_cloud

import collections
from scipy.spatial.transform import Rotation as R

import rospy
from lmf_planner_srvs_msgs.srv import set_goal_dir, set_goal_dirResponse
import tf
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Imu, PointCloud2
from geometry_msgs.msg import Pose, Twist
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from gazebo_msgs.msg import ModelState
from mav_msgs.msg import RateThrust
from std_srvs.srv import Empty, EmptyRequest, EmptyResponse
from std_msgs.msg import Float32MultiArray

class SimEnvBaseClass():
    def __init__(self):
        rospy.init_node('sim_wrapper', anonymous=True)
        
        self.get_params()

        self.done = False
        self.timeout = False
        self.timeout_timer = None
        if EVALUATE_MODE:
            self.timeout_length = EPISODE_TIMEOUT # seconds
        else:
            self.timeout_length = 30 # seconds

        self.robot_odom = collections.deque(maxlen=ODOM_QUEUE_LEN)
        self.imu = collections.deque(maxlen=ODOM_QUEUE_LEN)
        # Saves last ODOM_QUEUE_LEN images
        self.di_list = collections.deque(maxlen=DI_QUEUE_LEN)
        self.di_stamp = collections.deque(maxlen=DI_QUEUE_LEN)
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

        self.goal_init_publisher = rospy.Publisher("/delta/goal", Pose)
        self.trajectory_publisher = rospy.Publisher(TRAJECTORY_TOPIC, MarkerArray)
        self.cmd_publisher = rospy.Publisher("/delta/command/rate_thrust", RateThrust)
        self.cmd_vel_publisher = rospy.Publisher(SIM_CMD_TOPIC, Twist)
        self.model_state_publisher = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)

        self.pause_physics_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_physics_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
 
        self.reset_controller = rospy.ServiceProxy('/delta/pid_reset', Empty)

        # service
        rospy.Service('start_planner', Empty, self.start_planner)
        rospy.Service('stop_planner', Empty, self.stop_planner)
        rospy.Service('set_goal_dir', set_goal_dir, self.set_goal_dir)

        self.collide = False 

        self.robot1_pose = Pose()
        self.robot1_pose.position.x = 0.0
        self.robot1_pose.position.y = self.max_wp_y + 10.0
        self.robot1_pose.position.z = self.max_wp_z + 50.0
        self.robot1_pose.orientation.x = 0
        self.robot1_pose.orientation.y = 0
        self.robot1_pose.orientation.z = 0
        self.robot1_pose.orientation.w = 1     

        self.zero_goal = Pose()
        self.zero_goal.position.x = 0.0
        self.zero_goal.position.y = 0.0
        self.zero_goal.position.z = 0.0
        self.zero_goal.orientation.x = 0
        self.zero_goal.orientation.y = 0
        self.zero_goal.orientation.z = 0
        self.zero_goal.orientation.w = 1

    def get_params(self):
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

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)

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

    def mask_msg_to_img(self, msg):
        mask_img = np.ndarray((msg.height, msg.width), 'B', msg.data, 0)

        # cv2.imshow("mask img", mask_img)
        # cv2.waitKey(1)

        mask_img = mask_img.astype('float32') / 255 # in range 0->1

        return mask_img

    def mask_callback(self, data):
        mask = self.mask_msg_to_img(data)
        self.mask_queue.appendleft(mask)
        if (len(self.di_list) > 0):
            # publish pcl with intensity field as interestingness level for evaluation with voxblox
            # only for evaluation
            time_diff = np.abs((data.header.stamp - self.di_stamp[0]).to_sec())
            if (time_diff < 0.04): # 0.2 for yolo in sim
                self.publish_pcl_with_interestingness(self.di_list[0], mask, self.di_stamp[0])

    def depth_img_callback(self, data):
        pass

    def latent_callback(self, data):
        self.latent_list.appendleft(data.data)
        # print('data.data:', data.data)
        self.di_is_new = True

    def odom_callback(self, msg):
        self.robot_odom.appendleft(msg)
        
        # publish vehicle frame
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

    def timer_callback(self, event):
        self.timeout = True

    def reset_timer(self, time):
        #rospy.loginfo('Resetting the timeout timer')
        if (self.timeout_timer != None):
            self.timeout_timer.shutdown()
        # self.timeout_timer = rospy.Timer(rospy.Duration(self.goal_generation_radius * 5), self.timer_callback)
        if time <= 0:
            time = 1.0
        self.timeout_timer = rospy.Timer(rospy.Duration(time), self.timer_callback)

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
            # actions = np.array([command.thrust.x, command.thrust.y, command.thrust.z])
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