import os
import sys
sys.path.append('.')
sys.path.append('./train')
import numpy as np
import tensorflow as tf
import tensorrt as trt
from utilities import bcolors, GiB
from training import *
from config import *
import gflags
from common_flags import FLAGS

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
DTYPE = trt.float32

def populate_network(network, weights):
    # Initialize data, hiddenIn, cellIn, and seqLenIn inputs into RNN Layer
    robot_state_size = np.shape(weights['robot_state/dense0.weight'])[0]
    concat_size = np.shape(weights['obs_lowd/dense0.weight'])[0]
    robot_state_feature_size = np.shape(weights['robot_state/dense1.bias'])[0]
    di_feature_size = concat_size - robot_state_feature_size
    robot_state_input = network.add_input(name='robot_state_input', dtype=DTYPE, shape=trt.Dims([robot_state_size]))
    di_feature = network.add_input(name='di_feature', dtype=DTYPE, shape=trt.Dims([di_feature_size]))

    robot_state_input_reshape = network.add_shuffle(input=robot_state_input)
    robot_state_input_reshape.reshape_dims = trt.Dims([1, robot_state_size])
    # create action_input NN    
    fc_state_dense0_w = network.add_constant(weights=weights['robot_state/dense0.weight'], shape=np.shape(weights['robot_state/dense0.weight']))
    
    fc_state_dense0_mult = network.add_matrix_multiply(robot_state_input_reshape.get_output(0), trt.MatrixOperation.NONE, fc_state_dense0_w.get_output(0), trt.MatrixOperation.NONE)
    fc_state_dense0_b = network.add_constant(weights=weights['robot_state/dense0.bias'], shape=[1, np.shape(weights['robot_state/dense0.bias'])[0]])
    fc_state_dense0_add = network.add_elementwise(fc_state_dense0_mult.get_output(0), fc_state_dense0_b.get_output(0), trt.ElementWiseOperation.SUM)
    state_relu0 = network.add_activation(input=fc_state_dense0_add.get_output(0), type=trt.ActivationType.RELU) # (1,32)

    fc_state_dense1_w = network.add_constant(weights=weights['robot_state/dense1.weight'], shape=np.shape(weights['robot_state/dense1.weight']))
    fc_state_dense1_mult = network.add_matrix_multiply(state_relu0.get_output(0), trt.MatrixOperation.NONE, fc_state_dense1_w.get_output(0), trt.MatrixOperation.NONE)
    fc_state_dense1_b = network.add_constant(weights=weights['robot_state/dense1.bias'], shape=[1, np.shape(weights['robot_state/dense1.bias'])[0]])
    state_fc1 = network.add_elementwise(fc_state_dense1_mult.get_output(0), fc_state_dense1_b.get_output(0), trt.ElementWiseOperation.SUM)

    # create concatenate layer
    di_feature_reshape = network.add_shuffle(input=di_feature)
    di_feature_reshape.reshape_dims = trt.Dims([1, di_feature_size])           
    conc1 = network.add_concatenation(inputs = [state_fc1.get_output(0), di_feature_reshape.get_output(0)])
    conc1.axis = 1

    # create depth_state output NN
    fc_output_dense0_w = network.add_constant(weights=weights['obs_lowd/dense0.weight'], shape=np.shape(weights['obs_lowd/dense0.weight']))
    fc_output_dense0_mult = network.add_matrix_multiply(conc1.get_output(0), trt.MatrixOperation.NONE, fc_output_dense0_w.get_output(0), trt.MatrixOperation.NONE)
    fc_output_dense0_b = network.add_constant(weights=weights['obs_lowd/dense0.bias'], shape=[1, np.shape(weights['obs_lowd/dense0.bias'])[0]])
    fc_output_dense0_add = network.add_elementwise(fc_output_dense0_mult.get_output(0), fc_output_dense0_b.get_output(0), trt.ElementWiseOperation.SUM)
    output_relu1 = network.add_activation(input=fc_output_dense0_add.get_output(0), type=trt.ActivationType.RELU)

    fc_output_dense1_w = network.add_constant(weights=weights['obs_lowd/dense1.weight'], shape=np.shape(weights['obs_lowd/dense1.weight']))
    fc_output_dense1_mult = network.add_matrix_multiply(output_relu1.get_output(0), trt.MatrixOperation.NONE, fc_output_dense1_w.get_output(0), trt.MatrixOperation.NONE)
    fc_output_dense1_b = network.add_constant(weights=weights['obs_lowd/dense1.bias'], shape=[1, np.shape(weights['obs_lowd/dense1.bias'])[0]])
    fc_output_dense1_add = network.add_elementwise(fc_output_dense1_mult.get_output(0), fc_output_dense1_b.get_output(0), trt.ElementWiseOperation.SUM)

    fc_output_dense1_add.get_output(0).name = 'output_combiner'
    network.mark_output(tensor=fc_output_dense1_add.get_output(0))  

def build_engine(weights):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, builder.create_builder_config() as config:
        builder.max_batch_size = N_SIGMA
        config.max_workspace_size = GiB(1)
        print('CONFIG.FLAGS FP16:', config.get_flag(trt.BuilderFlag.FP16))
        print('CONFIG.FLAGS INT8:', config.get_flag(trt.BuilderFlag.INT8))
        print('CONFIG.FLAGS DEBUG:', config.get_flag(trt.BuilderFlag.DEBUG))
        print('CONFIG.FLAGS GPU_FALLBACK:', config.get_flag(trt.BuilderFlag.GPU_FALLBACK))
        print('CONFIG.FLAGS STRICT_TYPES:', config.get_flag(trt.BuilderFlag.STRICT_TYPES))
        print('CONFIG.FLAGS REFIT:', config.get_flag(trt.BuilderFlag.REFIT))
        # print('CONFIG.FLAGS DISABLE_TIMING_CACHE:', config.get_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE))
        # print('CONFIG.FLAGS TF32:', config.get_flag(trt.BuilderFlag.TF32))
        print('network.has_implicit_batch_dimension:', network.has_implicit_batch_dimension)
        # Populate the network using the weights from the saved model.
        populate_network(network, weights)
        # Build and return an engine.
        return builder.build_engine(network, config)

if __name__ == "__main__":
    # Limiting GPU memory growth: https://www.tensorflow.org/guide/gpu
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(bcolors.OKBLUE, len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs", bcolors.ENDC)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(bcolors.FAIL + "GPU error" + bcolors.ENDC)
            print(e)

    try:
        argv = FLAGS(sys.argv)  # parse flags
    except gflags.FlagsError:
        print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
        sys.exit(1)

    checkpoint_path = FLAGS.checkpoint_path
    if not os.path.exists(checkpoint_path):
        print ('ERROR: checkpoint_path ' + checkpoint_path + ' does NOT exist')
        sys.exit(1)

    custom_predictor_model_tmp = TrainCPN(depth_image_shape=DI_SHAPE)
    custom_predictor_model = InferenceCPN(depth_image_shape=DI_SHAPE)

    h_input_state = np.zeros((1, STATE_INPUT_SHAPE), dtype = np.float32)
    h_input_img = np.zeros((1, DI_SHAPE[0], DI_SHAPE[1], DI_SHAPE[2]), dtype = np.float32)
    h_inputs = [h_input_state, h_input_img]

    action_input = np.zeros((1, ACTION_HORIZON, ACTION_SHAPE_EVALUATE), dtype = np.float32)
    custom_predictor_model_tmp(h_inputs + [action_input])

    # Loads the weights
    custom_predictor_model_tmp.load_weights(checkpoint_path)
    custom_predictor_model.load_model(custom_predictor_model_tmp.get_model())

    # get the CNN and RNN keras models
    combiner_model = custom_predictor_model.get_depth_state_combiner()

    # build network with Python API
    weights = {}
    weights_robot_state_dense0 = combiner_model.get_layer('robot_state/dense0').get_weights()
    weights['robot_state/dense0.weight'] = weights_robot_state_dense0[0] # (2, 32)
    weights['robot_state/dense0.bias'] = weights_robot_state_dense0[1] # (32,)

    weights_robot_state_dense1 = combiner_model.get_layer('robot_state/dense1').get_weights()
    weights['robot_state/dense1.weight'] = weights_robot_state_dense1[0]
    weights['robot_state/dense1.bias'] = weights_robot_state_dense1[1]

    weights_output_dense0 = combiner_model.get_layer('obs_lowd/dense0').get_weights()
    weights['obs_lowd/dense0.weight'] = weights_output_dense0[0]
    weights['obs_lowd/dense0.bias'] = weights_output_dense0[1]    

    weights_output_dense1 = combiner_model.get_layer('obs_lowd/dense1').get_weights()
    weights['obs_lowd/dense1.weight'] = weights_output_dense1[0]
    weights['obs_lowd/dense1.bias'] = weights_output_dense1[1]

    # build, test and save the engine
    with build_engine(weights) as engine:
        with open("collision_depth_state_combiner_engine_fp32.trt", "wb") as f:
            f.write(engine.serialize())

    print("Done saving!")      