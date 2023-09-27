import os
import sys
sys.path.append('.')
sys.path.append('./train')
import numpy as np
import tensorflow as tf
import tensorrt as trt
from utilities import bcolors, GiB, get_batch_norm_params
from training import *
from config import *
import gflags
from common_flags import FLAGS

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
DTYPE = trt.float32

def populate_network(network, weights):
    image_input = network.add_input(name='image_input', dtype=DTYPE, shape=trt.Dims3([DI_WITH_MASK_SHAPE[0], DI_WITH_MASK_SHAPE[1], DI_WITH_MASK_SHAPE[2]]))
    image_input_permute = network.add_shuffle(input=image_input)
    image_input_permute.first_transpose = trt.Permutation([2, 0, 1]) # input for convolution layer follows CHW format
    image_input_permute.name = 'image_input_permute'

    # first stage
    conv2d_0 = network.add_convolution_nd(image_input_permute.get_output(0), 32, (5, 5), weights['conv2d_0/weight'], weights['conv2d_0/bias'])
    conv2d_0.stride_nd = trt.DimsHW(2, 2)
    conv2d_0.padding_mode = trt.PaddingMode.SAME_UPPER

    maxpool_0 = network.add_pooling_nd(conv2d_0.get_output(0), trt.PoolingType.MAX, (3, 3))
    maxpool_0.stride_nd = trt.DimsHW(2, 2)
    maxpool_0.padding_mode = trt.PaddingMode.SAME_UPPER

    # first residual block
    shift1, scale1, power1 = get_batch_norm_params(weights, 'batch_normalization_0')    
    batch_norm0 = network.add_scale(maxpool_0.get_output(0), trt.ScaleMode.CHANNEL, shift=shift1, scale=scale1, power=power1)
    batch_norm0_relu = network.add_activation(input=batch_norm0.get_output(0), type=trt.ActivationType.RELU)    
    conv2d_1 = network.add_convolution_nd(batch_norm0_relu.get_output(0), 32, (3, 3), weights['conv2d_1/weight'], weights['conv2d_1/bias'])
    conv2d_1.stride_nd = trt.DimsHW(2, 2)
    conv2d_1.padding_mode = trt.PaddingMode.SAME_UPPER

    shift2, scale2, power2 = get_batch_norm_params(weights, 'batch_normalization_1')    
    batch_norm1 = network.add_scale(conv2d_1.get_output(0), trt.ScaleMode.CHANNEL, shift=shift2, scale=scale2, power=power2)
    batch_norm1_relu = network.add_activation(input=batch_norm1.get_output(0), type=trt.ActivationType.RELU)    
    conv2d_2 = network.add_convolution_nd(batch_norm1_relu.get_output(0), 32, (3, 3), weights['conv2d_2/weight'], weights['conv2d_2/bias'])    
    conv2d_2.stride_nd = trt.DimsHW(1, 1)
    conv2d_2.padding_nd = trt.DimsHW(1, 1)

    conv2d_3 = network.add_convolution_nd(maxpool_0.get_output(0), 32, (1, 1), weights['conv2d_3/weight'], weights['conv2d_3/bias'])
    conv2d_3.stride_nd = trt.DimsHW(2, 2)
    conv2d_3.padding_mode = trt.PaddingMode.SAME_UPPER

    add1 = network.add_elementwise(conv2d_2.get_output(0), conv2d_3.get_output(0), trt.ElementWiseOperation.SUM)

    # output
    shift3, scale3, power3 = get_batch_norm_params(weights, 'batch_normalization_6')    
    batch_norm6 = network.add_scale(add1.get_output(0), trt.ScaleMode.CHANNEL, shift=shift3, scale=scale3, power=power3)
    batch_norm6_relu = network.add_activation(input=batch_norm6.get_output(0), type=trt.ActivationType.RELU)    
    output_conv2d_0 = network.add_convolution_nd(batch_norm6_relu.get_output(0), 32, (3, 3), weights['output_conv2d_0/weight'], weights['output_conv2d_0/bias'])    
    output_conv2d_0.stride_nd = trt.DimsHW(1, 1)
    output_conv2d_0.padding_nd = trt.DimsHW(1, 1)    

    output_conv2d_0_permute = network.add_shuffle(input=output_conv2d_0.get_output(0))
    output_conv2d_0_permute.first_transpose = trt.Permutation([1, 2, 0]) # CHW to HWC
    output_conv2d_0_permute.name = 'output_conv2d_0_permute'

    network.mark_output(tensor=output_conv2d_0_permute.get_output(0))

def build_engine(weights):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, builder.create_builder_config() as config:
        builder.max_batch_size = 1
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
        # Populate the network using weights from the PyTorch model.
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

    custom_predictor_model_tmp = TrainIPN(depth_image_shape=DI_WITH_MASK_SHAPE)
    custom_info_gain_model = InferenceIPN(depth_image_shape=DI_WITH_MASK_SHAPE)                

    h_input_state = np.zeros((1, STATE_INPUT_SHAPE), dtype = np.float32)
    h_input_img = np.zeros((1, DI_SHAPE[0], DI_SHAPE[1], DI_WITH_MASK_SHAPE[2]), dtype = np.float32)
    h_inputs = [h_input_state, h_input_img]

    action_input = np.zeros((1, ACTION_HORIZON, ACTION_SHAPE_EVALUATE), dtype = np.float32)
    custom_predictor_model_tmp(h_inputs + [action_input])    
    
    # Loads the weights
    custom_predictor_model_tmp.load_weights(checkpoint_path)
    custom_info_gain_model.load_model(custom_predictor_model_tmp.get_model())

    # get the CNN and RNN keras models
    cnn_only_model = custom_info_gain_model.get_feature_extractor()

    # build network with Python API
    weights = {}

    # batch_normalization
    weights['batch_normalization_0/gamma'], weights['batch_normalization_0/beta'], weights['batch_normalization_0/mean'], weights[
        'batch_normalization_0/var'] = cnn_only_model.get_layer('batch_normalization_0').get_weights()

    weights['batch_normalization_1/gamma'], weights['batch_normalization_1/beta'], weights['batch_normalization_1/mean'], weights[
        'batch_normalization_1/var'] = cnn_only_model.get_layer('batch_normalization_1').get_weights()

    weights['batch_normalization_6/gamma'], weights['batch_normalization_6/beta'], weights['batch_normalization_6/mean'], weights[
        'batch_normalization_6/var'] = cnn_only_model.get_layer('batch_normalization_6').get_weights()

    # conv2d
    weights_conv2d_0 = cnn_only_model.get_layer('conv2d_0').get_weights()
    weights['conv2d_0/weight'] = np.ascontiguousarray(weights_conv2d_0[0].transpose((3, 2, 0, 1)).reshape(-1))
    weights['conv2d_0/bias'] = weights_conv2d_0[1]

    weights_conv2d_1 = cnn_only_model.get_layer('conv2d_1').get_weights()
    weights['conv2d_1/weight'] = np.ascontiguousarray(weights_conv2d_1[0].transpose((3, 2, 0, 1)).reshape(-1))
    weights['conv2d_1/bias'] = weights_conv2d_1[1]

    weights_conv2d_2 = cnn_only_model.get_layer('conv2d_2').get_weights()
    weights['conv2d_2/weight'] = np.ascontiguousarray(weights_conv2d_2[0].transpose((3, 2, 0, 1)).reshape(-1))
    weights['conv2d_2/bias'] = weights_conv2d_2[1]
    
    weights_conv2d_3 = cnn_only_model.get_layer('conv2d_3').get_weights()
    weights['conv2d_3/weight'] = np.ascontiguousarray(weights_conv2d_3[0].transpose((3, 2, 0, 1)).reshape(-1))
    weights['conv2d_3/bias'] = weights_conv2d_3[1]

    weights_output_conv2d_0 = cnn_only_model.get_layer('output_conv2d_0').get_weights()
    weights['output_conv2d_0/weight'] = np.ascontiguousarray(weights_output_conv2d_0[0].transpose((3, 2, 0, 1)).reshape(-1))
    weights['output_conv2d_0/bias'] = weights_output_conv2d_0[1]    

    # build, test and save the engine
    with build_engine(weights) as engine:
        with open("infogain_image_feature_engine_fp32.trt", "wb") as f:
            f.write(engine.serialize())

    print("Done saving!")      