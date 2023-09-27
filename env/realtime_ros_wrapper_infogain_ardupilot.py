#!/usr/bin/env python3
import sys
sys.path.append('.')
import numpy as np
import collections

import rospy
import message_filters
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, Twist
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty, EmptyResponse, SetBool, SetBoolRequest
from lmf_planner_srvs_msgs.srv import set_goal_dir, set_goal_dirResponse

import cv2
from utilities import FilterKernel, HSVFilter
from config import *
import timeit

class RealtimeRosWrapperInfoGainArdupilot:
    def __init__(self):
        rospy.init_node('realtime_planner_infogain_ardupilot', anonymous=True)
        self.di_queue = collections.deque(maxlen=DI_QUEUE_LEN)
        # self.color_queue = collections.deque(maxlen=DI_QUEUE_LEN)
        self.mask_queue = collections.deque(maxlen=DI_QUEUE_LEN)
        self.odom_queue = collections.deque(maxlen=ODOM_QUEUE_LEN)

        self.zero_goal = Pose()
        self.zero_goal.position.x = 0.0
        self.zero_goal.position.y = 0.0
        self.zero_goal.position.z = 0.0
        self.zero_goal.orientation.x = 0
        self.zero_goal.orientation.y = 0
        self.zero_goal.orientation.z = 0
        self.zero_goal.orientation.w = 1

        # ros publishers/subcribers
        self.trajectory_publisher = rospy.Publisher(TRAJECTORY_TOPIC, MarkerArray, queue_size=1)
        self.cmd_vel_publisher = rospy.Publisher(ROBOT_CMD_TOPIC, Twist, tcp_nodelay=False, queue_size=1)
        self.filter_img_publisher = rospy.Publisher('filtered_image', Image, queue_size=1)
        # self.mask_publisher = rospy.Publisher('current_mask', Image, queue_size=1)
        self.goal_publisher = rospy.Publisher('goal', Pose, tcp_nodelay=False, queue_size=1)
    
        # self.depth_image_subscriber = message_filters.Subscriber("/d455/aligned_depth_to_color/image_raw", Image) # TODO: change align_depth=True in lmf_d455_t265_driver.launch
        self.depth_image_subscriber = rospy.Subscriber(ROBOT_DEPTH_TOPIC, Image, self.depth_image_callback, queue_size=1)
        # self.depth_image_subscriber = message_filters.Subscriber("/camera/depth/image_rect_raw", Image)
        self.mask_image_subscriber = rospy.Subscriber(ROBOT_MASK_TOPIC, Image, self.mask_callback, queue_size=1)
        # self.ts = message_filters.TimeSynchronizer([self.depth_image_subscriber, self.color_image_subscriber], 10)
        # self.ts.registerCallback(self.img_callback)
        # self.depth_image_subscriber = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.img_callback, tcp_nodelay=False, queue_size=1)
        self.odom_subscriber = rospy.Subscriber(ROBOT_ODOM_TOPIC, Odometry, self.odom_callback, queue_size=1)
        
        # service
        rospy.Service('start_planner', Empty, self.start_planner)
        rospy.Service('stop_planner', Empty, self.stop_planner)
        rospy.Service('set_goal_dir', set_goal_dir, self.set_goal_dir)

        self.start_detector_proxy = rospy.ServiceProxy('/start_detector', SetBool)

    def fill_in_fast(self, depth_map, max_depth=MAX_RANGE + 0.1, custom_kernel=FilterKernel.DIAMOND_KERNEL_5,
                    extrapolate = False, blur_type='bilateral', show_process=False):
        """Fast, in-place depth completion.
        Args:
            depth_map: projected depths
            max_depth: max depth value for inversion
            custom_kernel: kernel to apply initial dilation
            extrapolate: whether to extrapolate by extending depths to top of
                the frame, and applying a 31x31 full kernel dilation
            blur_type:
                'bilateral' - preserves local structure (recommended)
                'gaussian' - provides lower RMSE
        Returns:
            depth_map: dense depth map
        """
        #process_dict = collections.OrderedDict()
        #process_dict['depth_map in'] = copy.copy(depth_map)

        # Invert
        valid_pixels = (depth_map > 0.1) # TODO: 0.1 or min_range?
        depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]
        
        # Dilate
        depth_map = cv2.dilate(depth_map, custom_kernel)
        #process_dict['depth_map dilate'] = copy.copy(depth_map)

        # Hole closing
        depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FilterKernel.FULL_KERNEL_5)
        #process_dict['depth_map hole closing'] = copy.copy(depth_map)

        # Fill empty spaces with dilated values
        empty_pixels = (depth_map < 0.1) # TODO: 0.1 or min_range? (below also)
        dilated = cv2.dilate(depth_map, FilterKernel.FULL_KERNEL_7)
        depth_map[empty_pixels] = dilated[empty_pixels]

        # Extend highest pixel to top of image
        if extrapolate:
            top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
            top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]

            for pixel_col_idx in range(depth_map.shape[1]):
                depth_map[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = \
                    top_pixel_values[pixel_col_idx]

            # Large Fill
            empty_pixels = depth_map < 0.1
            dilated = cv2.dilate(depth_map, FilterKernel.FULL_KERNEL_31)
            depth_map[empty_pixels] = dilated[empty_pixels]

        # Median blur
        depth_map = cv2.medianBlur(depth_map, 5)

        # Bilateral or Gaussian blur
        if blur_type == 'bilateral':
            # Bilateral blur
            depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)

        elif blur_type == 'gaussian':
            # Gaussian blur
            valid_pixels = (depth_map > 0.1)
            blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
            depth_map[valid_pixels] = blurred[valid_pixels]
            
        # Invert
        valid_pixels = (depth_map > 0.1)
        depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]
        #process_dict['depth_map hole filtering'] = copy.copy(depth_map)
        #if show_process:
        #    return depth_map, process_dict
        return depth_map

    def msg_to_img(self, msg, visualization=False):
        #bridge = CvBridge()
        #img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        #img = np.array(struct.unpack('H'*msg.height*msg.width, msg.data))
        #img = img.reshape((msg.height,msg.width))
        img = np.ndarray((msg.height, msg.width), '<H', msg.data, 0)
        #print('msg.data:', msg.data)
        img = img.astype('float32') * 0.001 # convert pixel value to meter

        # remove nan, 
        np.clip(img, 0, MAX_RANGE, out=img) # this doesn't change NaN values
        img[np.isnan(img)] = MAX_RANGE

        # Visualization: normalize to [0,255] and cast to uint8
        # if visualization:
        #     img_scaled = img * 255 / MAX_RANGE
        #     img_scaled = img_scaled.astype('uint8')
        #     img_scaled = np.array(img_scaled)
        #     cv2.imshow("depth_img raw", img_scaled)

        # preprocess depth image
        if USE_D455_HOLE_FILLING:
            img_filtered = img
        else: # process by ourselves
            extrapolate = True
            blur_type = 'gaussian'
            # start = timeit.default_timer()
            img_filtered = self.fill_in_fast(img, extrapolate=extrapolate, blur_type=blur_type, custom_kernel=FilterKernel.FULL_KERNEL_31) # TODO: check the kernel
            np.clip(img_filtered, 0, MAX_RANGE, out=img_filtered)
            # print('Image preprocessing time (ms): ', (timeit.default_timer() - start) * 1000)

        # Visualization: normalize to [0,255] and cast to uint8
        img_filtered_uint8 = img_filtered * 255 / MAX_RANGE
        img_filtered_uint8 = img_filtered_uint8.astype('uint8')
        # img_filtered = np.array(img_filtered)
        #if visualization:
            # cv2.imshow("depth_img filtered", img_filtered)
            # cv2.waitKey(1)
        
        # bridge = CvBridge()
        # msg_filtered = bridge.cv2_to_imgmsg(img_filtered, "mono8")
        msg_filtered = Image()
        msg_filtered.height = msg.height
        msg_filtered.width = msg.width
        msg_filtered.encoding = '8UC1'
        msg_filtered.is_bigendian = 0
        msg_filtered.step = msg_filtered.width # 1 byte for each pixel
        msg_filtered.data = np.reshape(img_filtered_uint8, (msg.height * msg.width,)).tolist()

        return img_filtered, msg_filtered

    def color_msg_to_img(self, msg, visualization=False):
        img = np.ndarray((msg.height, msg.width, 3), 'B', msg.data, 0)
        # img = img.astype('float32') * 0.001 # convert pixel value to meter

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # apply the range on a mask
        mask = cv2.inRange(hsv, HSVFilter.HSVLOW_YELLOW, HSVFilter.HSVHIGH_YELLOW) # binary mask 0 or 255

        # remove noise
        mask = cv2.erode(mask, FilterKernel.FULL_KERNEL_3)
        mask = cv2.dilate(mask, FilterKernel.FULL_KERNEL_3)

        #if visualization:
            # maskedFrame = cv2.bitwise_and(img, img, mask = mask)
            # cv2.imshow("depth_img filtered", maskedFrame)
            # cv2.waitKey(1)

        mask_msg = Image()
        mask_msg.height = msg.height
        mask_msg.width = msg.width
        mask_msg.encoding = '8UC1'
        mask_msg.is_bigendian = 0
        mask_msg.step = mask_msg.width # 1 byte for each pixel
        mask_msg.data = np.reshape(mask, (msg.height * msg.width,)).tolist()

        mask = mask.astype('float32') / 255 # # binary mask 0 or 1
        # print('mask:', np.min(mask))

        return mask, mask_msg

    def mask_msg_to_img(self, msg):
        mask_img = np.ndarray((msg.height, msg.width), 'B', msg.data, 0)

        # cv2.imshow("mask img", mask_img)
        # cv2.waitKey(1)

        mask_img = mask_img.astype('float32') / 255 # in range 0->1

        return mask_img

    def visualize_trajectory(self, trajectory_lib, collision_score_combined, collision_score_timestamp=np.array([]), best_indx=None, worst_indx=None, safe_indx=np.array([])):
        if self.trajectory_publisher.get_num_connections() > 0:
            marker_array = MarkerArray()
            num_sequence = trajectory_lib.shape[0]
            id = 0
            for j in range(num_sequence):
                points = trajectory_lib[j]
                if VISUALIZATION_MODE == 0:
                    # for i in np.array([1,points.shape[0]-1]): # reduce number of markers
                    for i in np.array([points.shape[0]-1]):
                        marker = Marker()
                        marker.id = id
                        id = id + 1
                        marker.header.frame_id =  "state"
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
                    collision_scores = collision_score_timestamp[j]
                    for i in range(points.shape[0]):
                        collision_score = collision_scores[i]
                        marker = Marker()
                        marker.id = id
                        id = id + 1
                        marker.header.frame_id = "state"
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
                        marker.text = str(collision_score_combined[j,0])
                        marker_array.markers.append(marker)
            # Publish the MarkerArray
            self.trajectory_publisher.publish(marker_array)

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

    def start_detector(self):
        req = SetBoolRequest()
        req.data = True
        self.start_detector_proxy(req)

    def img_callback(self, depth_image, color_image):
        self.di_queue.appendleft(depth_image)
        self.color_queue.appendleft(color_image)

    def depth_image_callback(self, depth_image):
        self.di_queue.appendleft(depth_image)

    # def color_image_callback(self, color_image):
    #     self.color_queue.appendleft(color_image)

    def mask_callback(self, mask):
        self.mask_queue.appendleft(mask)

    def odom_callback(self, data):
        self.odom_queue.appendleft(data)

    def stop_robot(self):
        self.goal_publisher.publish(self.zero_goal)

    def step(self, action):
        # # convert vel cmd from vehicle frame to body frame: NO NEED
        # robot_odom = self.odom_queue[0]
        # robot_pose = robot_odom.pose.pose
        # r_robot = R.from_quat(np.array([robot_pose.orientation.x, robot_pose.orientation.y, robot_pose.orientation.z, robot_pose.orientation.w]))
        # robot_euler_angles = r_robot.as_euler('xyz', degrees=False)
        # velocity_W_psi = np.array([action[0][0], action[0][1], action[0][2]])
        # velocity_B = R.from_euler('xy', robot_euler_angles[0:2], degrees=False).as_matrix().transpose() @ (velocity_W_psi)        

        velocity_W_psi = np.array([action[0][0], action[0][1], action[0][2]])
        command = Twist()
        command.linear.x = velocity_W_psi[0]
        command.linear.y = velocity_W_psi[1]
        command.linear.z = velocity_W_psi[2]
        command.angular.x = 0.0
        command.angular.y = 0.0
        command.angular.z = action[0][3] # reference yaw angle HERE!
        self.cmd_vel_publisher.publish(command)
        
        info = {'status':'none'}
        done = False
        return done, info

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
        if (len(self.di_queue) == DI_QUEUE_LEN) & (len(self.odom_queue) > 0):
            di_data = self.di_queue[0]
            self.di_queue.pop() # remove the oldest depth image
            if len(self.mask_queue) > 0:
                current_mask = self.mask_msg_to_img(self.mask_queue[0])
                # self.mask_queue.pop()
            else:
                current_mask = None
            robot_odom = self.odom_queue[0]
            # filter the depth image
            current_di, filtered_img = self.msg_to_img(di_data, True)
            self.filter_img_publisher.publish(filtered_img)

            # fill the robot_state array
            robot_pose = robot_odom.pose.pose
            robot_twist = robot_odom.twist.twist 
            robot_state = np.array([
                robot_pose.position.x,
                robot_pose.position.y,
                robot_pose.position.z,    
                robot_twist.linear.x,
                robot_twist.linear.y,
                robot_twist.linear.z,
                0.0, #linear_acc[0]
                0.0, #linear_acc[1]
                0.0, #linear_acc[2] - 9.81,
                robot_pose.orientation.x, # TODO: check if the attitude is correct
                robot_pose.orientation.y,
                robot_pose.orientation.z,
                robot_pose.orientation.w,
                robot_twist.angular.x,
                robot_twist.angular.y,
                robot_twist.angular.z])
            #print('robot_state:', robot_state)
            valid_obs = True
        else:
            robot_state = None
            current_di = None
            current_mask = None
            valid_obs = False        
        return robot_state, current_di, current_mask, valid_obs