import numpy as np

RUN_IN_SIM = True

############### CHECKPOINT PATHS ###############
# Tensorflow weight files for CPN
CPN_TF_CHECKPOINT_PATH = ['model_weights/vel_3_5/net1/saved-model.hdf5', # SKIP_STEP_GENERATE = 3, ACTION_HORIZON = 14, VEL_MAX = 3.5 m/s
                        'model_weights/vel_3_5/net2/saved-model.hdf5',
                        'model_weights/vel_3_5/net3/saved-model.hdf5',
                        'model_weights/vel_3_5/net4/saved-model.hdf5',
                        'model_weights/vel_3_5/net5/saved-model.hdf5']

# folders containing TRT files (need to run optimize scripts to create TRT files)
CPN_TRT_CHECKPOINT_PATH = ['model_weights/vel_3_5/net1',
                        'model_weights/vel_3_5/net2',
                        'model_weights/vel_3_5/net3',
                        'model_weights/vel_3_5/net4',
                        'model_weights/vel_3_5/net5']

# Tensorflow weight files for seVAE-CPN
seVAE_CPN_TF_CHECKPOINT_PATH = ['model_weights/VAE_EVO_back_to_origin/net1/saved-model.hdf5', # SKIP_STEP_GENERATE = 5, ACTION_HORIZON = 15, VEL_MAX = 2.0 m/s
                        'model_weights/VAE_EVO_back_to_origin/net2/saved-model.hdf5',
                        'model_weights/VAE_EVO_back_to_origin/net3/saved-model.hdf5',
                        'model_weights/VAE_EVO_back_to_origin/net4/saved-model.hdf5',
                        'model_weights/VAE_EVO_back_to_origin/net5/saved-model.hdf5']

# folders containing TRT files (need to run optimize scripts to create TRT files)
seVAE_CPN_TRT_CHECKPOINT_PATH = ['model_weights/VAE_EVO_back_to_origin/net1',
                                'model_weights/VAE_EVO_back_to_origin/net2',
                                'model_weights/VAE_EVO_back_to_origin/net3',
                                'model_weights/VAE_EVO_back_to_origin/net4',
                                'model_weights/VAE_EVO_back_to_origin/net5']

# Tensorflow weight files for IPN
IPN_TF_CHECKPOINT_PATH = 'model_weights/models_info_gain_voxblox_area_1000/saved-model.hdf5' # SKIP_STEP_GENERATE = 5, ACTION_HORIZON = 15, VEL_MAX = 2.0 m/s

# folders containing TRT files (need to run optimize scripts to create TRT files)
IPN_TRT_CHECKPOINT_PATH = 'model_weights/models_info_gain_voxblox_area_1000'

############### PLANNING PARAMS ###############
PLANNING_TYPE = 1 # 0: end2end ORACLE, 1: seVAE-ORACLE, 2: A-ORACLE, 3: voxblox

# ROS topics in the real robot
ROBOT_DEPTH_TOPIC = '/d455/depth/image_rect_raw'
ROBOT_ODOM_TOPIC = '/mavros/local_position/odom_in_map'
ROBOT_CMD_TOPIC = '/cmd_velocity'
ROBOT_MASK_TOPIC = '/current_mask' # when using PLANNING_TYPE >= 2
TRAJECTORY_TOPIC = '/trajectory'
ROBOT_LATENT_TOPIC = '/latent_space' # when using PLANNING_TYPE = 1

# planning params
PLANNING_HORIZONTAL_FOV = 87.0 # degrees
PLANNING_VERTICAL_FOV = 40.0 # degrees
STEPS_TO_REPLAN = 1 # replan after ... depth images
CMD_VELOCITY = 1.5

# Motion-primitives in velocity-steering angle space
NUM_VEL_X = 1
NUM_VEL_Z = 8
NUM_YAW = 32
NUM_SEQUENCE_TO_EVALUATE = NUM_VEL_X * NUM_VEL_Z * NUM_YAW
NUM_SEQUENCE_TO_EVALUATE_INFOGAIN = NUM_SEQUENCE_TO_EVALUATE # ignored for now 

# inference type
COLLISION_USE_TENSORRT = False
INFOGAIN_USE_TENSORRT = False

# visualization
ENABLE_VISUALIZATION = True
VISUALIZATION_MODE = 0 # 0: only the first and final timestamps, 1: all timestamps, 2: based on 1st order approximation, TODO: check all modes
# first order params to create estimated trajectories from the action sequences (for visualization only)
ALPHA_VX = 0.9512 # Ts=0.4, T_sampling=3/(10*15)
ALPHA_VY = 0.9512
ALPHA_VZ = 0.9512
ALPHA_PSI = 0.9646 # Ts=1/K_yaw=1/1.8, T_sampling=3/(10*15)
# ALPHA_V = 0.92 # Ts=0.4, T_sampling=5/(10*15)
# ALPHA_PSI = 0.9418 # Ts=1/K_yaw=1/1.8, T_sampling=5/(10*15)

# use image pre-processing step in real robot?
USE_D455_HOLE_FILLING = False

# collision cost
DEADEND_COL_SCORE_THRESHOLD_HIGH = 3.0 # allow to yaw in spot to find new free direction
DEADEND_COL_SCORE_THRESHOLD_LOW = 1.5 # exit yawing in dead end mode 
TIME_WEIGHT_FACTOR = 0.04 # 0.0: zero effect
COLLISION_THRESHOLD = 0.08 #0.2 #0.04 # threshold (compared to the safest action sequence) before checking goal_dir

# waypoint condition and what to do after reaching the waypoints
WAYPOINT_FILE = 'waypoints/waypoint_flightmare.txt'
WAYPOINT_DISTANCE_THRESHOLD = 5.0
WAYPOINT_YAW_THRESHOLD = np.deg2rad(180.0)
ALLOW_YAW_AT_WAYPOINT = True

# Uncertainty-aware
N_E = 1 # number of CPNs in the ensemble
USE_UT = False

USE_ADDITIVE_GAUSSIAN_IMAGE_NOISE = False
USE_ADDITIVE_GAUSSIAN_STATE_NOISE = False

# Depth image's noise
IMAGE_NOISE_FACTOR = 0 # 0 - 0.005

# Unscented Transform
ALPHA = 1.0
BETA = 2.0
L = 3 # [v_x, v_y, v_z]
K_UKF = 0.0 #3 - L 
LAMBDA = (ALPHA**2) * (L + K_UKF) - L
P_vx = 0.16
P_vy = 0.16
P_vz = 0.16
N_SIGMA = 2*L + 1

# A-ORACLE
MAX_MASK_RANGE_EVALUATE = 10.0 # detection range
MAX_METRICS_RANGE = 10.0 # metrics range
INFOGAIN_THRESHOLD = 0.5
TIMEOUT_TYPE = 2 # 0: after TIME_ALLOWED * T_STRAIGHT, switch to normal ORACLE
# 1: (total_time_from_prev_wp + current_dist2goal / CMD_VELOCITY) < TIME_ALLOWED * T_STRAIGHT
# 2: total_dist_from_prev_wp + current_dist2goal < TIME_ALLOWED * D_STRAIGHT
TIME_ALLOWED = 100.0

SIM_USE_GRAYSCALE_FILTER = False
# False: subcribe to SIM_MASK_TOPIC
# True: subcribe to SIM_GRAYSCALE_TOPIC and perform cv2.inRange with below thresholds to create the detection mask
MIN_PIXEL_VALUE = 100 # range for creating the mask in sim, used only when SIM_USE_GRAYSCALE_FILTER = True
MAX_PIXEL_VALUE = 255

############### DATA COLLECTION PARAMS ###############

# ROS topics in sim
SIM_DEPTH_TOPIC = '/delta/agile_autonomy/unity_depth' # '/delta/agile_autonomy/sgm_depth'
SIM_ODOM_TOPIC = '/delta/odometry_sensor1/odometry'
SIM_CMD_TOPIC = '/delta/cmd_velocity'
SIM_IMU_TOPIC = '/delta/ground_truth/imu'
SIM_MASK_TOPIC = '/current_mask'
SIM_GRAYSCALE_TOPIC = '/delta/rgbd/camera_depth/camera/image_raw'
SIM_LATENT_TOPIC = '/latent_space'

# range params
MAX_RANGE = 10.0 # max range of depth image [m]
MAX_RANGE_INV = 1.0 / MAX_RANGE
RANGE_SCALE = MAX_RANGE / 255.0 # from 0-255 to 0-MAX_RANGE
RANGE_SCALE_INV = 255.0 / MAX_RANGE


HORIZONTAL_FOV = 87.0 # degrees
VERTICAL_FOV = 58.0 # degrees
DEPTH_TS = 1.0/15 # inv of depth sensor fps
THRESHOLD_DISTANCE = 0.4 # threshold distance to record data [m]
SKIP_STEP_GENERATE = 5 # number of depth images to skip before recording data
# make sure: ACTION_HORIZON * SKIP_STEP_GENERATE * DEPTH_TS * VEL_MAX <= MAX_RANGE
ACTION_HORIZON = 15 # predict the collision probabilities only after SKIP_STEP_GENERATE steps
STEPS_TO_REGENERATE = SKIP_STEP_GENERATE * ACTION_HORIZON # number of depth images after which a new trajectory is generated

VEL_MAX = MAX_RANGE / (ACTION_HORIZON * SKIP_STEP_GENERATE * DEPTH_TS) # m/s
MIN_CMD_APPEND = np.array([0.0, 0.0, -VEL_MAX, -np.deg2rad(HORIZONTAL_FOV/2)])
MAX_CMD_APPEND = np.array([VEL_MAX, 0.0, VEL_MAX, np.deg2rad(HORIZONTAL_FOV/2)])


############### TRAINING AND OPTIMIZING PARAMS ###############
TRAIN_INFOGAIN = False
EVALUATE_MODE = True # only used when RUN_IN_SIM = True

DROPOUT_KEEP_RATE = 0.5 # dropout layer at the end of Dronet (during training)

# input shape
ACTION_SHAPE_EVALUATE = 3 # [velocity_x, velocity_z, steering_angle]
ACTION_SHAPE = 4 # [velocity_x, velocity_y, velocity_z, steering_angle]
STATE_INPUT_SHAPE = 6 # [vx, vy, vz, yaw_rate, roll, pitch]
PCA_NUM = 3 # TODO remove this
DI_SHAPE = (270, 480, 1)
DI_WITH_MASK_SHAPE = (DI_SHAPE[0], DI_SHAPE[1], DI_SHAPE[2] + 1)
SKIP_STEP_INFERENCE_INFOGAIN = 2 #4 # only evaluate once after ... steps for infogain network
ACTION_HORIZON_INFERENCE_INFOGAIN = int((ACTION_HORIZON-1)/SKIP_STEP_INFERENCE_INFOGAIN) + 1
# tensorRT add_gather and add_slice in Xavier NX return error when horizon <= 3 !!!!

# seVAE latent vector
DI_LATENT_SIZE = 128

############### OTHER PARAMS ###############
# sensor queue size
DI_QUEUE_LEN = 1 # capacity of di_queue
ODOM_QUEUE_LEN = 10 # capacity of odom_queue

# depth to pcl
# from /d455/depth/image_rect_raw
DEPTH_CX = 240.932861328125
DEPTH_CY = 137.50704956054688
DEPTH_FX = 240.9515380859375
DEPTH_FY = 240.9515380859375

# TF camera-body
CAM_PITCH = 0 # degrees, positive: pitch down, negative: pitch up
R_BC = np.array([[np.cos(np.deg2rad(CAM_PITCH)), 0, np.sin(np.deg2rad(CAM_PITCH))], 
                [0, 1, 0], 
                [-np.sin(np.deg2rad(CAM_PITCH)), 0, np.cos(np.deg2rad(CAM_PITCH))]]) @ \
        np.array([[0, 0, 1], 
                [-1, 0, 0], 
                [0, -1, 0]]) # assuming depth camera only pitches up/down
t_BC = np.array([[0.09], [0.04], [0.024]])

# simulated evaluation's params
NUM_EPISODES_EVALUATE = 100
EPISODE_TIMEOUT = 100 # seconds
MAX_INITIAL_X = 0.0 #2.5 # meters
MAX_INITIAL_Y = 0.0 #9.5 # meters
MAX_INITIAL_Z = 1.0 # meters
MAX_INITIAL_YAW = 0.0 #-180 # degrees
MIN_INITIAL_X = 0.0 #2.5 # meters
MIN_INITIAL_Y = 0.0 #9.5 # meters
MIN_INITIAL_Z = 1.0 # meters
MIN_INITIAL_YAW = 0.0 #-180.0 # degrees

# simulated params in Flightmare (only used when SIM_USE_FLIGHTMARE = True): match config from agile_autonomy
SIM_USE_FLIGHTMARE = True # only used when RUN_IN_SIM = True
SPACING = 4.5 # space btw trees or objects
# for comparison with Agile, need to be the same as test_time/spacings param in agile_autonomy/planner_learning/config/test_settings.yaml 
UNITY_START_POS = [-20.,20.,0.,0] # x-y-z-yaw, ignore MAX_INITIAL_..., MIN_INITIAL_... above
TAKEOFF_HEIGHT = 2.0 # the same as /hummingbird/autopilot/optitrack_start_height
CRASHED_THR = 0.18
# NOTE: change the following path to point to agile_autonomy/data_generation/data/ folder in your system
EXPERT_FOLDER = '/home/students/workspaces/agile_autonomy_ws/catkin_aa/src/agile_autonomy/data_generation/data/'