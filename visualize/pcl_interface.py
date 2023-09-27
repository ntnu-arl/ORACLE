import sys
sys.path.append('.')    
import numpy as np    
import collections
from config import *
from utilities import create_point_cloud
import rospy
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

# this file is to publishing pcl with the intensity field as interestingness value of the voxel (from a detection mask)
# update t_BC and the header in publish_pcl_with_interestingness according to your setup
t_BC = np.array([[0.2], [0.0], [-0.05]]) # match firefly model from mav_active_3d_planning
SIM_MASK_TOPIC = '/current_mask'
SIM_DEPTH_TOPIC = '/firefly/rgbd/camera_depth/depth'
SIM_ODOM_TOPIC = '/firefly/ground_truth/odometry'

class PclInterface:
    def __init__(self):
        rospy.init_node('pcl_interface', anonymous=True)
        self.di_list = collections.deque(maxlen=DI_QUEUE_LEN)
        self.di_stamp = collections.deque(maxlen=DI_QUEUE_LEN)
        self.mask_queue = collections.deque(maxlen=DI_QUEUE_LEN)
        self.mask_stamp = collections.deque(maxlen=DI_QUEUE_LEN)

        self.di_is_new = False
        self.mask_is_new = False
        self.bridge = CvBridge()

        self.pcl_with_interestingness_publisher = rospy.Publisher("/pcl_with_interestingness", PointCloud2)
        self.mask_image_subscriber = rospy.Subscriber(SIM_MASK_TOPIC, Image, self.mask_callback, queue_size=1)
        self.depth_image_subscriber = rospy.Subscriber(SIM_DEPTH_TOPIC, Image, self.depth_img_callback, queue_size=1)
        self.odom_subscriber = rospy.Subscriber(SIM_ODOM_TOPIC, Odometry, self.odom_callback, queue_size=1)

        self.odom_timestamp = rospy.Time.now()
        self.first_odom = True
        self.prev_pos = np.array([])
        self.traveled_dist = 0.0

    def odom_callback(self, data):
        if self.first_odom:
            self.first_odom = False
            self.prev_pos = np.array([data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z])
        robot_pos = np.array([data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z])
        self.traveled_dist += np.linalg.norm(robot_pos - self.prev_pos)
        self.prev_pos = robot_pos
        
        time_diff = np.abs((data.header.stamp - self.odom_timestamp).to_sec())
        if (time_diff >= 1.0): # print at 1 Hz 
            self.odom_timestamp = data.header.stamp
            print('Traveled distance:', self.traveled_dist)
        

    def mask_msg_to_img(self, msg):
        mask_img = np.ndarray((msg.height, msg.width), 'B', msg.data, 0)

        # cv2.imshow("mask img", mask_img)
        # cv2.waitKey(1)

        mask_img = mask_img.astype('float32') / 255 # in range 0->1

        return mask_img

    def mask_callback(self, data):
        mask = self.mask_msg_to_img(data)
        self.mask_queue.appendleft(mask)
        self.mask_stamp.appendleft(data.header.stamp)
        self.mask_is_new = True
        # if (len(self.di_list) > 0):
        #     # publish pcl with intensity field as interestingness level for evaluation with voxblox
        #     # only for evaluation
        #     time_diff = np.abs((data.header.stamp - self.di_stamp[0]).to_sec())
        #     if (time_diff < 0.04): # 0.2 for yolo in sim
        #         self.publish_pcl_with_interestingness(self.di_list[0], mask, self.di_stamp[0])

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
        pcl_msg = create_point_cloud(pcl_with_interestingness.T, 'firefly/odometry_sensor1', time_stamp)
        self.pcl_with_interestingness_publisher.publish(pcl_msg)

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

    def publish_loop(self):
        if (len(self.di_list) > 0) & (len(self.mask_queue) > 0) & (self.di_is_new or self.mask_is_new):
            if self.di_is_new:
                self.di_is_new = False
            if self.mask_is_new:
                self.mask_is_new = False
            # publish pcl with intensity field as interestingness level for evaluation with voxblox
            # only for evaluation
            time_diff = np.abs((self.mask_stamp[0] - self.di_stamp[0]).to_sec())
            if (time_diff < 0.04): # 0.2 for yolo in sim
                self.publish_pcl_with_interestingness(self.di_list[0], self.mask_queue[0], self.di_stamp[0])
        rospy.sleep(0.005)

if __name__ == '__main__':
    rospy.loginfo('Ready')
    pcl_interface = PclInterface()
    while not rospy.is_shutdown():
        pcl_interface.publish_loop()

