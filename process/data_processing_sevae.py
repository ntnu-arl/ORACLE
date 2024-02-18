import os
import sys
sys.path.append('.')
import tensorflow as tf
import numpy as np
from scipy.spatial.transform import Rotation as R
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

class SevaeDataLabeler:
    def __init__(self, load_path, save_path):
        self.load_path = load_path
        self.save_path = save_path
        # for randomize the blobs
        self.rs = np.random.default_rng(seed=None)


    def serialize_example(self, latent_space, actions_seq, robot_state, pca_state, label, info_gain_t0_label,
                          info_gain_label, pos_label, image_shape, action_horizon):
        feature = {
            'latent_space': _bytes_feature(latent_space),
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
            '.p') and os.path.isfile(os.path.join(load_path, f))]) / 7)
        print("NUMBER OF PICKLE STACKS", nb_files)
        for k in range(nb_files):
            obs_load = pickle.load(open(load_path + "/obs_dump" + str(k) + ".p", "rb"))
            depth_latent_load = pickle.load(open(load_path + "/di_latent" + str(k) + ".p", "rb"))
            depth_flipped_latent_load = pickle.load(open(load_path + "/di_flipped_latent" + str(k) + ".p", "rb"))
            action_load = pickle.load(open(load_path + "/action_dump" + str(k) + ".p", "rb"))
            action_index_load = pickle.load(open(load_path + "/action_index_dump" + str(k) + ".p", "rb"))
            collision_load = pickle.load(open(load_path + "/collision_dump" + str(k) + ".p", "rb"))

            filename = save_path + '/data' + str(k) + '.tfrecords'
            N_episode = len(depth_latent_load)

            with tf.io.TFRecordWriter(filename) as writer:
                for i in range(N_episode):
                    obs_episode = obs_load[i]
                    depth_latent_episode = depth_latent_load[i]
                    depth_flipped_latent_episode = depth_flipped_latent_load[i]
                    action_episode = action_load[i]
                    action_index_episode = action_index_load[i]
                    collision_episode = collision_load[i]
                    N_images = len(depth_latent_episode)
                    total_step_episode = len(action_episode)

                    N_sample_append = 0
                    is_first_collide_idx = False

                    for j in range(N_images):
                        depth_latent_current = depth_latent_episode[j]
                        depth_flipped_latent_current = depth_flipped_latent_episode[j]
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
                            collision_label, pos_yaw_label, pos_yaw_flip_label = self.calculate_label(collision_label, robot_yaw, p_WBs, R_WBs)
                            info_gain_t0_label = 0.0
                            info_obj_label = np.zeros(ACTION_HORIZON)
                            pca_state = np.zeros(PCA_NUM*ACTION_HORIZON)
                            pca_flip_state = np.zeros(PCA_NUM*ACTION_HORIZON)

                            example = self.serialize_example(tf.io.serialize_tensor(depth_latent_current),
                                                        tf.io.serialize_tensor(np.array(action_seq_current, dtype=np.float32)),
                                                        tf.io.serialize_tensor(obs_current.astype('float32')),
                                                        tf.io.serialize_tensor(pca_state.astype('float32')),
                                                        tf.io.serialize_tensor(np.array(collision_label, dtype=np.float32)),
                                                        np.float32(info_gain_t0_label),
                                                        tf.io.serialize_tensor(np.array(info_obj_label, dtype=np.float32)),
                                                        tf.io.serialize_tensor(np.array(pos_yaw_label, dtype=np.float32)),
                                                        DI_WITH_MASK_SHAPE, ACTION_HORIZON)
                            writer.write(example)
                            example_flip = self.serialize_example(tf.io.serialize_tensor(depth_flipped_latent_current),
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
                            
                                collision_label, pos_yaw_label, pos_yaw_flip_label = self.calculate_label(collision_label, robot_yaw, p_WBs, R_WBs)
                                info_gain_t0_label = 0.0
                                info_obj_label = np.zeros(ACTION_HORIZON)
                                pca_state = np.zeros(PCA_NUM*ACTION_HORIZON)
                                pca_flip_state = np.zeros(PCA_NUM*ACTION_HORIZON)

                                example = self.serialize_example(tf.io.serialize_tensor(depth_latent_current),
                                                        tf.io.serialize_tensor(np.array(action_seq_current_tmp, dtype=np.float32)),
                                                        tf.io.serialize_tensor(obs_current.astype('float32')),
                                                        tf.io.serialize_tensor(pca_state.astype('float32')),
                                                        tf.io.serialize_tensor(np.array(collision_label, dtype=np.float32)),
                                                        np.float32(info_gain_t0_label),
                                                        tf.io.serialize_tensor(np.array(info_obj_label, dtype=np.float32)),
                                                        tf.io.serialize_tensor(np.array(pos_yaw_label, dtype=np.float32)),
                                                        DI_WITH_MASK_SHAPE, ACTION_HORIZON)
                                writer.write(example)
                                example_flip = self.serialize_example(tf.io.serialize_tensor(depth_flipped_latent_current),
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

    sevae_labeler = SevaeDataLabeler(load_path, save_path)
    sevae_labeler.run()