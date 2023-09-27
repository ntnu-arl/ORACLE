import os
import sys
sys.path.append('.')
import tensorflow as tf
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.ndimage.filters import gaussian_filter
# from skimage import data
# import cv2
# import copy # only for visulization purpose
# import skimage.io as io

try:
   import cPickle as pickle
except:
   import pickle

import gflags
from common_flags import FLAGS
from config import *
from utilities import binary_blobs, deterministic_binary_blobs, create_point_cloud

import timeit

if TRAIN_INFOGAIN:
    import rospy
    from std_msgs.msg import Float32MultiArray
    from voxblox_msgs.srv import InfoGain

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

class DataLabeler:
    def __init__(self, load_path, save_path):
        self.load_path = load_path
        self.save_path = save_path
        # for randomize the blobs
        self.rs = np.random.default_rng(seed=None)

        self.receive_info_gain = False

        # ROS stuff
        if TRAIN_INFOGAIN:
            rospy.init_node('data_info_gain', anonymous=True)
            # self.pcl_publisher = rospy.Publisher('/delta/rgbd/camera_depth/depth/points_oracle', PointCloud2, queue_size=1, latch=True)
            self.info_gain_subscriber = rospy.Subscriber('/delta/info_gain', Float32MultiArray, self.info_gain_callback)
            self.calc_info_srv = rospy.ServiceProxy('/voxblox_node/calc_info_gain', InfoGain)

    def serialize_example(self, image, actions_seq, robot_state, pca_state, label, info_gain_t0_label,
                          info_gain_label, pos_label, image_shape, action_horizon):
        feature = {
            'image': _bytes_feature(image),
            'actions': _bytes_feature(actions_seq),
            'robot_state': _bytes_feature(robot_state),
            'pca_state': _bytes_feature(pca_state),
            'label': _bytes_feature(label),
            'info_gain_t0_label': _float_feature(info_gain_t0_label),
            'info_gain_label': _bytes_feature(info_gain_label),
            'pos_label': _bytes_feature(pos_label),
            'height': _int64_feature(image_shape[0]),
            'width': _int64_feature(image_shape[1]),
            'depth': _int64_feature(image_shape[2]),
            'action_horizon': _int64_feature(action_horizon),
        }
        #  Create a Features message using tf.train.Example.
        example_proto = tf.train.Example(
            features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def process_cmd(self, action_seq_current, robot_yaw):
        action_seq_current_np = np.array(action_seq_current)
        cmd_vel_x = action_seq_current_np[:, :, 0]
        cmd_vel_z = action_seq_current_np[:, :, 2]
        cmd_yaw = action_seq_current_np[:, :, 3]
        # offset robot's yaw from cmd's yaw, then wrap around +- pi
        cmd_yaw_relative = (cmd_yaw - robot_yaw + np.pi) % (2 * np.pi) - np.pi
        action_seq_current = np.stack(
            [cmd_vel_x, cmd_vel_z, cmd_yaw_relative], axis=2)
        action_seq_current_flip_yaw = np.stack(
            [cmd_vel_x, cmd_vel_z, -cmd_yaw_relative], axis=2)
        return action_seq_current.tolist(), action_seq_current_flip_yaw.tolist()

    def generate_blobs(self, blob_size_fraction=(0.3, 0.3), volume_fraction=0.05, seed=None, blur=True, sigma=(5, 5), pixel=None):
        blobs = binary_blobs(
            shape=(DI_SHAPE[0], DI_SHAPE[1]),
            blob_size_fraction=blob_size_fraction,
            volume_fraction=volume_fraction,
            seed=seed,
            pixel=pixel)
        blobs = blobs.astype('float32')
        # print('blobs:', np.shape(blobs))
        if blur:
            return gaussian_filter(blobs, sigma=sigma)
        else:
            return blobs

    def generate_deterministic_blobs(self, pixel, blob_size_fraction=(0.3, 0.3), volume_fraction=0.05, blur=True, sigma=(5, 5)):
        blobs = deterministic_binary_blobs(
            pixel=pixel,
            shape=(DI_SHAPE[0], DI_SHAPE[1]),
            blob_size_fraction=blob_size_fraction,
            volume_fraction=volume_fraction)
        blobs = blobs.astype('float32')
        # print('blobs:', np.shape(blobs))
        if blur:
            return gaussian_filter(blobs, sigma=sigma)
        else:
            return blobs

    def info_gain_callback(self, msg):
        self.info_gain = msg.data.ravel()
        self.receive_info_gain = True

    def calculate_label_infogain(self, collision_label, robot_yaw, di_current, p_WBs, R_WBs, pca=None):
        num_valid_pose = len(p_WBs) - 1
        p_WB_0 = p_WBs[0]
        R_yaw = R.from_euler('z', -robot_yaw, degrees=False).as_matrix()
        if (self.rs.random() < 0.5): # generate the mask in the moving direction of the robot with a probability
        # TODO: add empty mask here also    
            p_WB_t = p_WBs[-1]
            # convert from world frame to yaw-rotated world frame (vehicle frame)
            # calculate projected end-point
            pos_label = R_yaw @ (p_WB_t - p_WB_0)
            pos_label = np.expand_dims(pos_label, axis=1)
            pos_label_C = R_BC.transpose() @ (pos_label - t_BC)
            x_px = np.rint(pos_label_C[0, 0] * DEPTH_FX / pos_label_C[2, 0] + DEPTH_CX)
            x_px = x_px.astype(int)
            y_px = np.rint(pos_label_C[1, 0] * DEPTH_FY / pos_label_C[2, 0] + DEPTH_CY)
            y_px = y_px.astype(int)

            # create the mask
            if (x_px >= 0) and (x_px < DI_SHAPE[1]) and (y_px >= 0) and (y_px < DI_SHAPE[0]):
                pixel = (y_px, x_px)
            else:
                pixel = None
        else:
            pixel = None
        
        blob_size_fraction = self.rs.uniform(low=[0.25, 0.25], high=[0.5, 0.5])
        volume_fraction = self.rs.uniform(low=0.1, high=0.25)
        sigma_x = self.rs.integers(low=1, high=10, endpoint=True)
        sigma_y = self.rs.integers(low=1, high=10, endpoint=True)
        mask = self.generate_blobs(
            pixel=pixel, blur=True, sigma=(sigma_y, sigma_x), blob_size_fraction=blob_size_fraction, volume_fraction=volume_fraction)

        # blob_size_fraction = (0.5, 0.5)
        # volume_fraction = 0.1
        # pixel = np.array([[135], [240]]) # [list of y_px], [list of x_px]
        # mask = self.generate_deterministic_blobs(
        #     pixel=pixel, blur=False, blob_size_fraction=blob_size_fraction, volume_fraction=volume_fraction)
        # mask[:, 240:] = 0.0

        # cv2.imshow('mask', mask)
        # cv2.waitKey(0)
        
        mask = np.expand_dims(mask, axis=2)  # (270, 480, 1)

        # create the pcl
        R_WB_0 = R_WBs[0].as_matrix()
        
        pixel_idx = np.indices((DI_SHAPE[0], DI_SHAPE[1]))
        z = di_current.ravel()
        x = (pixel_idx[1].ravel() - DEPTH_CX) * z / DEPTH_FX
        y = (pixel_idx[0].ravel() - DEPTH_CY) * z / DEPTH_FY   

        valid_idx = np.where(z < MAX_RANGE) # remove original NaN points so voxblox won't integrate them
        z_valid = z[valid_idx]
        x_valid = x[valid_idx]
        y_valid = y[valid_idx]
        mask_valid = mask.ravel()[valid_idx]

        p_C = np.vstack((x_valid, y_valid, z_valid))
        p_B = t_BC + R_BC @ p_C  # conver to body frame: delta/odometry_sensor1
        #p_W = R_WB_0 @ p_B  + np.expand_dims(p_WB_0, axis=1)
        pcl_with_mask = np.vstack((p_B, mask_valid))
        
        # calculate future robot positions in current vehicle frame, and future camera pose in current body frame 
        pos_yaw_labels = [np.zeros(4)] * ACTION_HORIZON
        pos_yaw_flip_labels = [np.zeros(4)] * ACTION_HORIZON
        camera_poses = np.zeros(6 * ACTION_HORIZON, dtype=np.float32)
        info_gain_t0_label = 0 # ignored
        for t in range(num_valid_pose):
            p_WB_t = p_WBs[t + 1]
            robot_euler_angles_t = R_WBs[t + 1].as_euler('xyz', degrees=False)
            R_WB_t = R_WBs[t + 1].as_matrix()
            # calculate robot's pos label in initial vehicle frame
            pos_label = R_yaw @ (p_WB_t - p_WB_0)
            pos_flip_label = np.copy(pos_label)
            pos_flip_label[1] = -pos_label[1]  # flip y
            # print('pos_label:', pos_label, ", robot_yaw:", np.rad2deg(robot_yaw), ", vector:", p_WB_t - p_WB_0)
            yaw_angle_realative = (robot_euler_angles_t[2] - robot_yaw + np.pi) % (2 * np.pi) - np.pi
            pos_yaw_label = np.append(pos_label, yaw_angle_realative)
            pos_yaw_flip_label = np.append(pos_flip_label, -yaw_angle_realative)
            pos_yaw_labels[t] = pos_yaw_label

            pos_yaw_flip_labels[t] = pos_yaw_flip_label
            # calculate camera pose in body frame (delta/odometry_sensor1)
            R_B0_Bt = R_WB_0.T @ R_WB_t
            cam_euler_angles = R.from_matrix(R_B0_Bt).as_euler('xyz', degrees=False)
            cam_pos = R_WB_0.T @ (p_WB_t - p_WB_0).reshape(3,1) + R_B0_Bt @ t_BC
            camera_poses[6*t : 6*t + 3] = cam_pos.ravel()
            camera_poses[6*t + 3 : 6*t + 6] = cam_euler_angles.ravel()

        # print('camera_poses:', camera_poses)
        pcl = create_point_cloud(pcl_with_mask.T, 'delta/odometry_sensor1')
        start_full = timeit.default_timer()
        info_gain_response = self.calc_info_srv(pcl, camera_poses) # send to voxblox to calculate
        stop_full = timeit.default_timer()
        time_full = stop_full - start_full
        print('TIME (ms):', time_full * 1000)
        info_labels = info_gain_response.info_gain
        print('info_labels:', info_labels)
        
        # debugging
        # self.pcl_publisher.publish(pcl)
        # while not rospy.is_shutdown():
        # while (True):
        #     rospy.sleep(0.001)    

        # PCA calculation (ignored)
        pca_feature = np.zeros(PCA_NUM*ACTION_HORIZON)
        pca_feature_flip = np.zeros(PCA_NUM*ACTION_HORIZON)

        return collision_label, info_gain_t0_label, info_labels, pos_yaw_labels, pos_yaw_flip_labels, pca_feature, pca_feature_flip, mask

    def calculate_label(self, collision_label, robot_yaw, p_WBs, R_WBs):
        num_valid_pose = len(p_WBs) - 1
        p_WB_0 = p_WBs[0]
        R_WB_0 = R_WBs[0].as_matrix()
        R_yaw = R.from_euler('z', -robot_yaw, degrees=False).as_matrix()
        
        # calculate future robot positions in current vehicle frame, and future camera pose in current body frame 
        pos_yaw_labels = [np.zeros(4)] * ACTION_HORIZON
        pos_yaw_flip_labels = [np.zeros(4)] * ACTION_HORIZON
        for t in range(num_valid_pose):
            p_WB_t = p_WBs[t + 1]
            robot_euler_angles_t = R_WBs[t + 1].as_euler('xyz', degrees=False)
            # calculate robot's pos label in initial vehicle frame
            pos_label = R_yaw @ (p_WB_t - p_WB_0)
            pos_flip_label = np.copy(pos_label)
            pos_flip_label[1] = -pos_label[1]  # flip y
            # print('pos_label:', pos_label, ", robot_yaw:", np.rad2deg(robot_yaw), ", vector:", p_WB_t - p_WB_0)
            yaw_angle_realative = (robot_euler_angles_t[2] - robot_yaw + np.pi) % (2 * np.pi) - np.pi
            pos_yaw_label = np.append(pos_label, yaw_angle_realative)
            pos_yaw_flip_label = np.append(pos_flip_label, -yaw_angle_realative)
            pos_yaw_labels[t] = pos_yaw_label
            pos_yaw_flip_labels[t] = pos_yaw_flip_label

        return collision_label, pos_yaw_labels, pos_yaw_flip_labels

    def convert_obs_episode(self, obs_seq):
        R_WBs = []
        p_WBs = []
        # print('len(obs_seq):', len(obs_seq))
        for j in range(len(obs_seq)):
            obs_current = obs_seq[j]
            quaternion = obs_current[9:13]
            R_WB_t = R.from_quat(
                [quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
            R_WBs.append(R_WB_t)
            p_WBs.append(obs_current[0:3])
        return R_WBs, p_WBs 

    def run(self):
        # multiple files in the folder
        nb_files = int(len([f for f in os.listdir(load_path) if f.endswith(
            '.p') and os.path.isfile(os.path.join(load_path, f))]) / 5)  # five dicts
        print("NUMBER OF PICKLE STACKS", nb_files)
        for k in range(nb_files):
            obs_load = pickle.load(open(load_path + "/obs_dump" + str(k) + ".p", "rb"))
            di_load = pickle.load(open(load_path + "/di_dump" + str(k) + ".p", "rb"))
            action_load = pickle.load(open(load_path + "/action_dump" + str(k) + ".p", "rb"))
            action_index_load = pickle.load(open(load_path + "/action_index_dump" + str(k) + ".p", "rb"))
            collision_load = pickle.load(open(load_path + "/collision_dump" + str(k) + ".p", "rb"))

            filename = save_path + '/data' + str(k) + '.tfrecords'
            N_episode = len(di_load)

            with tf.io.TFRecordWriter(filename) as writer:
                for i in range(N_episode):
                    obs_episode = obs_load[i]
                    di_episode = di_load[i]
                    action_episode = action_load[i]
                    action_index_episode = action_index_load[i]
                    collision_episode = collision_load[i]
                    N_images = len(di_episode)
                    total_step_episode = len(action_episode)

                    N_sample_append = 0
                    is_first_collide_idx = False

                    for j in range(N_images):
                        di_current = di_episode[j]
                        action_index_current = action_index_episode[j]
                        obs_current = obs_episode[action_index_current]

                        # calculate robot's yaw
                        quaternion = obs_current[9:13]
                        R_WB = R.from_quat([quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
                        robot_euler_angles = R_WB.as_euler('xyz', degrees=False)
                        robot_yaw = robot_euler_angles[2]

                        # convert velocity in body frame to yaw-rotated world frame
                        velocity_B = obs_current[3:6]
                        velocity_W_psi = R.from_euler('xy', robot_euler_angles[0:2], degrees=False).as_matrix() @ (velocity_B)
                        # print('velocity_B:', velocity_B, " , velocity_W_psi:", velocity_W_psi)
                        obs_current[3:6] = velocity_W_psi

                        # flip the states
                        obs_current_flip = np.copy(obs_current)
                        obs_current_flip[4] = -obs_current[4]  # v_y
                        # ignore the acceleration since we don't use it
                        # flip the roll angle, quat: x,y,z,w format
                        obs_current_flip[9:13] = R.from_euler(
                            'xyz', [-robot_euler_angles[0], robot_euler_angles[1], robot_euler_angles[2]], degrees=False).as_quat()
                        obs_current_flip[13] = -obs_current[13]  # w_x
                        obs_current_flip[15] = -obs_current[15]  # w_z
                        
                        # append roll and pitch angles to the end of obs_current
                        obs_current = np.append(obs_current, [robot_euler_angles[0:2], np.sin(robot_euler_angles[0:2]), np.cos(robot_euler_angles[0:2])])
                        rp_angle_flip = [-robot_euler_angles[0], robot_euler_angles[1]]
                        obs_current_flip = np.append(obs_current_flip, [rp_angle_flip, np.sin(rp_angle_flip), np.cos(rp_angle_flip)])
                        
                        if (total_step_episode - action_index_current > ACTION_HORIZON + 1):
                            collision_label = np.zeros(ACTION_HORIZON)
                            action_seq_current = action_episode[action_index_current:action_index_current + ACTION_HORIZON]
                            action_seq_current, action_seq_current_flip_yaw = self.process_cmd(action_seq_current, robot_yaw)
                            obs_seq = obs_episode[action_index_current:action_index_current + ACTION_HORIZON + 1]
                            R_WBs, p_WBs = self.convert_obs_episode(obs_seq)

                            # augment horizontally flip data
                            if TRAIN_INFOGAIN:
                                # calculate info obj label
                                collision_label, info_gain_t0_label, info_obj_label, pos_yaw_label, pos_yaw_flip_label, pca_state, pca_flip_state, mask = self.calculate_label_infogain(
                                    collision_label, robot_yaw, di_episode[j], p_WBs, R_WBs, pca=None)
                                # concat blobs and di_current
                                di_current_with_mask = np.concatenate((di_current, mask), axis=2)
                            else:
                                collision_label, pos_yaw_label, pos_yaw_flip_label = self.calculate_label(collision_label, robot_yaw, p_WBs, R_WBs)
                                info_gain_t0_label = 0.0
                                info_obj_label = np.zeros(ACTION_HORIZON)
                                pca_state = np.zeros(PCA_NUM*ACTION_HORIZON)
                                pca_flip_state = np.zeros(PCA_NUM*ACTION_HORIZON)
                                di_current_with_mask = di_current
                            
                            di_current_with_mask_flip = np.flip(di_current_with_mask, 1)


                            example = self.serialize_example(tf.io.serialize_tensor((di_current_with_mask * 1000).astype('uint16')),
                                                        tf.io.serialize_tensor(np.array(action_seq_current, dtype=np.float32)),
                                                        tf.io.serialize_tensor(obs_current.astype('float32')),
                                                        tf.io.serialize_tensor(pca_state.astype('float32')),
                                                        tf.io.serialize_tensor(np.array(collision_label, dtype=np.float32)),
                                                        np.float32(info_gain_t0_label),
                                                        tf.io.serialize_tensor(np.array(info_obj_label, dtype=np.float32)),
                                                        tf.io.serialize_tensor(np.array(pos_yaw_label, dtype=np.float32)),
                                                        DI_WITH_MASK_SHAPE, ACTION_HORIZON)
                            writer.write(example)
                            example_flip = self.serialize_example(tf.io.serialize_tensor((di_current_with_mask_flip * 1000).astype('uint16')),
                                                            tf.io.serialize_tensor(np.array(action_seq_current_flip_yaw, dtype=np.float32)),
                                                            tf.io.serialize_tensor(obs_current_flip.astype('float32')),
                                                            tf.io.serialize_tensor(pca_flip_state.astype('float32')),
                                                            tf.io.serialize_tensor(np.array(collision_label, dtype=np.float32)),
                                                            np.float32(info_gain_t0_label),
                                                            tf.io.serialize_tensor(np.array(info_obj_label, dtype=np.float32)),
                                                            tf.io.serialize_tensor(np.array(pos_yaw_flip_label, dtype=np.float32)),
                                                            DI_WITH_MASK_SHAPE, ACTION_HORIZON)
                            writer.write(example_flip)
                        elif collision_episode:
                            if not is_first_collide_idx:
                                is_first_collide_idx = True
                                N_positive_sample = N_images - j # N_negative_sample = j
                                N_sample_append = 1 + (j // N_positive_sample)
                                #print('N_sample_append:', N_sample_append)
                                #print('N_positve_sample:', N_positive_sample, ',j:', j)

                            # di_current_serialize = tf.io.serialize_tensor(di_current)
                            # di_flip = np.flip(di_current, 1)
                            # di_flip_serialize = tf.io.serialize_tensor(di_flip)
                            action_seq_current = action_episode[action_index_current:total_step_episode - 1] # ignore last action since it's not executed yet
                            N_append_length = ACTION_HORIZON + 1 - (total_step_episode - action_index_current) # number of actions to append at the end
                            # print('action_index_current:', action_index_current, ',total_step_episode:', total_step_episode, 'N_append_length:', N_append_length)
                            if N_append_length < ACTION_HORIZON:
                                collision_label = np.zeros(ACTION_HORIZON - N_append_length - 1).tolist() + np.ones(N_append_length + 1).tolist()
                            else:
                                collision_label = np.ones(ACTION_HORIZON)
                            # print('collision_label:', collision_label)
                            
                            obs_seq = obs_episode[action_index_current:]
                            R_WBs, p_WBs = self.convert_obs_episode(obs_seq)
                            for k in range(N_sample_append):
                                # append random actions at the end: balance the dataset here 
                                action_append = self.rs.uniform(MIN_CMD_APPEND, MAX_CMD_APPEND, (N_append_length, 1, ACTION_SHAPE))
                                action_append = action_append.tolist()
                                action_seq_current_tmp = action_seq_current + action_append
                                action_seq_current_tmp, action_seq_current_tmp_flip_yaw = self.process_cmd(action_seq_current_tmp, robot_yaw)
                                
                                # augment horizontally flip data
                                if TRAIN_INFOGAIN:
                                    # calculate info obj label
                                    collision_label, info_gain_t0_label, info_obj_label, pos_yaw_label, pos_yaw_flip_label, pca_state, pca_flip_state, mask = self.calculate_label_infogain(
                                        collision_label, robot_yaw, di_episode[j], p_WBs, R_WBs, pca=None)
                                    # concat blobs and di_current
                                    di_current_with_mask = np.concatenate((di_current, mask), axis=2)
                                else:
                                    collision_label, pos_yaw_label, pos_yaw_flip_label = self.calculate_label(collision_label, robot_yaw, p_WBs, R_WBs)
                                    info_gain_t0_label = 0.0
                                    info_obj_label = np.zeros(ACTION_HORIZON)
                                    pca_state = np.zeros(PCA_NUM*ACTION_HORIZON)
                                    pca_flip_state = np.zeros(PCA_NUM*ACTION_HORIZON)
                                    di_current_with_mask = di_current
                                
                                di_current_with_mask_flip = np.flip(di_current_with_mask, 1)

                                example = self.serialize_example(tf.io.serialize_tensor((di_current_with_mask * 1000).astype('uint16')),
                                                        tf.io.serialize_tensor(np.array(action_seq_current_tmp, dtype=np.float32)),
                                                        tf.io.serialize_tensor(obs_current.astype('float32')),
                                                        tf.io.serialize_tensor(pca_state.astype('float32')),
                                                        tf.io.serialize_tensor(np.array(collision_label, dtype=np.float32)),
                                                        np.float32(info_gain_t0_label),
                                                        tf.io.serialize_tensor(np.array(info_obj_label, dtype=np.float32)),
                                                        tf.io.serialize_tensor(np.array(pos_yaw_label, dtype=np.float32)),
                                                        DI_WITH_MASK_SHAPE, ACTION_HORIZON)
                                writer.write(example)
                                example_flip = self.serialize_example(tf.io.serialize_tensor((di_current_with_mask_flip * 1000).astype('uint16')),
                                                        tf.io.serialize_tensor(np.array(action_seq_current_tmp_flip_yaw, dtype=np.float32)),
                                                        tf.io.serialize_tensor(obs_current_flip.astype('float32')),
                                                        tf.io.serialize_tensor(pca_flip_state.astype('float32')),
                                                        tf.io.serialize_tensor(np.array(collision_label, dtype=np.float32)),
                                                        np.float32(info_gain_t0_label),
                                                        tf.io.serialize_tensor(np.array(info_obj_label, dtype=np.float32)),
                                                        tf.io.serialize_tensor(np.array(pos_yaw_flip_label, dtype=np.float32)),
                                                        DI_WITH_MASK_SHAPE, ACTION_HORIZON)
                                writer.write(example_flip)

if __name__ == "__main__":
    try:
        argv = FLAGS(sys.argv)  # parse flags
    except gflags.FlagsError:
        print('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
        sys.exit(1)

    load_path = FLAGS.load_path
    save_path = FLAGS.save_tf_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print('loading pickle object from files in folder:', load_path + '/')
    print('save tfrecord to folder:', save_path + '/')

    info_gain_labeler = DataLabeler(load_path, save_path)
    info_gain_labeler.run()