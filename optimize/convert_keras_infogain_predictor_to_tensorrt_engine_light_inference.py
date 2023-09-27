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
BATCH_SIZE_PREDICTOR = NUM_SEQUENCE_TO_EVALUATE_INFOGAIN

def populate_network(network, weights):
    # Initialize data, hiddenIn, cellIn, and seqLenIn inputs into RNN Layer
    HIDDEN_UNIT = int(np.shape(weights['robot_state/dense2.bias'])[0] / 2)
    robot_state_input = network.add_input(name='robot_state_input', dtype=DTYPE, shape=trt.Dims([STATE_INPUT_SHAPE]))
    cnn_feature_input = network.add_input(name='cnn_feature_input', dtype=DTYPE, shape=trt.Dims3([34, 60, 32])) # TODO: fix hard code
    action_input = network.add_input(name='action_input', dtype=DTYPE, shape=trt.Dims2(ACTION_HORIZON, ACTION_SHAPE_EVALUATE))

    # create action_input NN
    fc_action_dense0_w = network.add_constant(weights=weights['action/dense0.weight'], shape=np.shape(weights['action/dense0.weight']))
    fc_action_dense0_mult = network.add_matrix_multiply(action_input, trt.MatrixOperation.NONE, fc_action_dense0_w.get_output(0), trt.MatrixOperation.NONE)
    fc_action_dense0_mult.name = 'fc_action_dense0_mult'
    fc_action_dense0_b = network.add_constant(weights=weights['action/dense0.bias'], shape=[1, np.shape(weights['action/dense0.bias'])[0]])
    fc_action_dense0_add = network.add_elementwise(fc_action_dense0_mult.get_output(0), fc_action_dense0_b.get_output(0), trt.ElementWiseOperation.SUM)
    action_relu0 = network.add_activation(input=fc_action_dense0_add.get_output(0), type=trt.ActivationType.RELU) # (18,16)

    fc_action_dense1_w = network.add_constant(weights=weights['action/dense1.weight'], shape=np.shape(weights['action/dense1.weight']))
    fc_action_dense1_mult = network.add_matrix_multiply(action_relu0.get_output(0), trt.MatrixOperation.NONE, fc_action_dense1_w.get_output(0), trt.MatrixOperation.NONE)
    fc_action_dense1_mult.name = 'fc_action_dense1_mult'
    fc_action_dense1_b = network.add_constant(weights=weights['action/dense1.bias'], shape=[1, np.shape(weights['action/dense1.bias'])[0]])
    action_fc1 = network.add_elementwise(fc_action_dense1_mult.get_output(0), fc_action_dense1_b.get_output(0), trt.ElementWiseOperation.SUM)

    # print('action_fc1:', action_fc1.shape)

    # create state input NN
    robot_state_size = np.shape(weights['robot_state/dense0.weight'])[0]
    robot_state_input_reshape = network.add_shuffle(input=robot_state_input)
    robot_state_input_reshape.reshape_dims = trt.Dims([1, robot_state_size])
    fc_robot_state_dense0_w = network.add_constant(weights=weights['robot_state/dense0.weight'], shape=np.shape(weights['robot_state/dense0.weight']))
    fc_robot_state_dense0_mult = network.add_matrix_multiply(robot_state_input_reshape.get_output(0), trt.MatrixOperation.NONE, fc_robot_state_dense0_w.get_output(0), trt.MatrixOperation.NONE)
    fc_robot_state_dense0_mult.name = 'fc_robot_state_dense0_mult'
    fc_robot_state_dense0_b = network.add_constant(weights=weights['robot_state/dense0.bias'], shape=[1, np.shape(weights['robot_state/dense0.bias'])[0]])
    fc_robot_state_dense0_add = network.add_elementwise(fc_robot_state_dense0_mult.get_output(0), fc_robot_state_dense0_b.get_output(0), trt.ElementWiseOperation.SUM)
    robot_state_relu0 = network.add_activation(input=fc_robot_state_dense0_add.get_output(0), type=trt.ActivationType.RELU) # (18,16)

    fc_robot_state_dense1_w = network.add_constant(weights=weights['robot_state/dense1.weight'], shape=np.shape(weights['robot_state/dense1.weight']))
    fc_robot_state_dense1_mult = network.add_matrix_multiply(robot_state_relu0.get_output(0), trt.MatrixOperation.NONE, fc_robot_state_dense1_w.get_output(0), trt.MatrixOperation.NONE)
    fc_robot_state_dense1_mult.name = 'fc_robot_state_dense1_mult'
    fc_robot_state_dense1_b = network.add_constant(weights=weights['robot_state/dense1.bias'], shape=[1, np.shape(weights['robot_state/dense1.bias'])[0]])
    fc_robot_state_dense1_add = network.add_elementwise(fc_robot_state_dense1_mult.get_output(0), fc_robot_state_dense1_b.get_output(0), trt.ElementWiseOperation.SUM)
    robot_state_relu1 = network.add_activation(input=fc_robot_state_dense1_add.get_output(0), type=trt.ActivationType.RELU)    

    fc_robot_state_dense2_w = network.add_constant(weights=weights['robot_state/dense2.weight'], shape=np.shape(weights['robot_state/dense2.weight']))
    fc_robot_state_dense2_mult = network.add_matrix_multiply(robot_state_relu1.get_output(0), trt.MatrixOperation.NONE, fc_robot_state_dense2_w.get_output(0), trt.MatrixOperation.NONE)
    fc_robot_state_dense2_mult.name = 'fc_robot_state_dense2_mult'
    fc_robot_state_dense2_b = network.add_constant(weights=weights['robot_state/dense2.bias'], shape=[1, np.shape(weights['robot_state/dense2.bias'])[0]])
    fc_robot_state_dense2_add = network.add_elementwise(fc_robot_state_dense2_mult.get_output(0), fc_robot_state_dense2_b.get_output(0), trt.ElementWiseOperation.SUM)

    # print('fc_robot_state_dense2_add:', fc_robot_state_dense2_add.shape)

    # "split" layer
    initial_state_h = network.add_slice(fc_robot_state_dense2_add.get_output(0), start=trt.Dims([0, 0]), shape=trt.Dims([1, 16]), stride=trt.Dims([1, 1]))
    initial_state_c = network.add_slice(fc_robot_state_dense2_add.get_output(0), start=trt.Dims([0, 16]), shape=trt.Dims([1, 16]), stride=trt.Dims([1, 1]))

    print('initial_state_h:', initial_state_h.shape)
    print('initial_state_c:', initial_state_c.shape)

    initial_state_h_reshape = network.add_shuffle(input=initial_state_h.get_output(0))
    initial_state_h_reshape.reshape_dims = trt.Dims([1, HIDDEN_UNIT])
    initial_state_h_reshape.name = 'initial_state_h_reshape'
    initial_state_c_reshape = network.add_shuffle(input=initial_state_c.get_output(0))
    initial_state_c_reshape.reshape_dims = trt.Dims([1, HIDDEN_UNIT])
    initial_state_c_reshape.name = 'initial_state_c_reshape'      
    
    # create an RNN layer w/ 1 layers and 64 hidden states    
    rnn = network.add_rnn_v2(input=action_fc1.get_output(0), layer_count=1, hidden_size=HIDDEN_UNIT, max_seq_length=ACTION_HORIZON, 
                            op=trt.RNNOperation.LSTM)                        
    rnn.input_mode = trt.RNNInputMode.LINEAR
    rnn.direction = trt.RNNDirection.UNIDIRECTION

    # Set RNNv2 optional inputs
    rnn.hidden_state = initial_state_h_reshape.get_output(0)
    rnn.cell_state = initial_state_c_reshape.get_output(0)

    rnn.set_weights_for_gate(layer_index=0, gate=trt.RNNGateType.INPUT, is_w=True, weights=weights['lstm.Wi'].transpose().copy())
    rnn.set_weights_for_gate(layer_index=0, gate=trt.RNNGateType.INPUT, is_w=False, weights=weights['lstm.Ri'].transpose().copy())
    rnn.set_bias_for_gate(layer_index=0, gate=trt.RNNGateType.INPUT, is_w=True, bias=weights['lstm.Bi'])
    rnn.set_bias_for_gate(layer_index=0, gate=trt.RNNGateType.INPUT, is_w=False, bias=np.zeros(shape=(HIDDEN_UNIT), dtype=np.float32))  

    rnn.set_weights_for_gate(layer_index=0, gate=trt.RNNGateType.FORGET, is_w=True, weights=weights['lstm.Wf'].transpose().copy())
    rnn.set_weights_for_gate(layer_index=0, gate=trt.RNNGateType.FORGET, is_w=False, weights=weights['lstm.Rf'].transpose().copy())
    rnn.set_bias_for_gate(layer_index=0, gate=trt.RNNGateType.FORGET, is_w=True, bias=weights['lstm.Bf'])
    rnn.set_bias_for_gate(layer_index=0, gate=trt.RNNGateType.FORGET, is_w=False, bias=np.zeros(shape=(HIDDEN_UNIT), dtype=np.float32))    

    rnn.set_weights_for_gate(layer_index=0, gate=trt.RNNGateType.CELL, is_w=True, weights=weights['lstm.Wc'].transpose().copy())
    rnn.set_weights_for_gate(layer_index=0, gate=trt.RNNGateType.CELL, is_w=False, weights=weights['lstm.Rc'].transpose().copy())
    rnn.set_bias_for_gate(layer_index=0, gate=trt.RNNGateType.CELL, is_w=True, bias=weights['lstm.Bc'])
    rnn.set_bias_for_gate(layer_index=0, gate=trt.RNNGateType.CELL, is_w=False, bias=np.zeros(shape=(HIDDEN_UNIT), dtype=np.float32))    

    rnn.set_weights_for_gate(layer_index=0, gate=trt.RNNGateType.OUTPUT, is_w=True, weights=weights['lstm.Wo'].transpose().copy())
    rnn.set_weights_for_gate(layer_index=0, gate=trt.RNNGateType.OUTPUT, is_w=False, weights=weights['lstm.Ro'].transpose().copy())
    rnn.set_bias_for_gate(layer_index=0, gate=trt.RNNGateType.OUTPUT, is_w=True, bias=weights['lstm.Bo'])
    rnn.set_bias_for_gate(layer_index=0, gate=trt.RNNGateType.OUTPUT, is_w=False, bias=np.zeros(shape=(HIDDEN_UNIT), dtype=np.float32))

    print('rnn.get_output(0):', rnn.get_output(0).shape)
    # rnn_reduce = network.add_slice(rnn.get_output(0), start=trt.Dims([SKIP_STEP_INFERENCE_INFOGAIN-1, 0]), shape=trt.Dims([ACTION_HORIZON_INFERENCE_INFOGAIN, 16]), stride=trt.Dims([SKIP_STEP_INFERENCE_INFOGAIN, 1]))
    indices_np = np.ascontiguousarray(np.flip(np.arange(ACTION_HORIZON-1, -1, -SKIP_STEP_INFERENCE_INFOGAIN, dtype='int32')))
    print('indices_np:', indices_np)
    indices_tensor = network.add_constant(shape=trt.Dims([ACTION_HORIZON_INFERENCE_INFOGAIN]), weights=indices_np).get_output(0)
    rnn_reduce = network.add_gather(input=rnn.get_output(0), indices=indices_tensor, axis=0)
    print('rnn_reduce:', rnn_reduce.get_output(0).shape)

    # # create state_output NN
    fc_state_feature_dense0_w = network.add_constant(weights=weights['state_feature/dense0.weight'], shape=np.shape(weights['state_feature/dense0.weight']))
    fc_state_feature_dense0_mult = network.add_matrix_multiply(rnn_reduce.get_output(0), trt.MatrixOperation.NONE, fc_state_feature_dense0_w.get_output(0), trt.MatrixOperation.NONE)
    fc_state_feature_dense0_mult.name = 'fc_state_feature_dense0_mult'
    fc_state_feature_dense0_b = network.add_constant(weights=weights['state_feature/dense0.bias'], shape=[1, np.shape(weights['state_feature/dense0.bias'])[0]])
    fc_state_feature_dense0_add = network.add_elementwise(fc_state_feature_dense0_mult.get_output(0), fc_state_feature_dense0_b.get_output(0), trt.ElementWiseOperation.SUM)
    state_feature_relu0 = network.add_activation(input=fc_state_feature_dense0_add.get_output(0), type=trt.ActivationType.RELU)

    fc_state_feature_dense1_w = network.add_constant(weights=weights['state_feature/dense1.weight'], shape=np.shape(weights['state_feature/dense1.weight']))
    fc_state_feature_dense1_mult = network.add_matrix_multiply(state_feature_relu0.get_output(0), trt.MatrixOperation.NONE, fc_state_feature_dense1_w.get_output(0), trt.MatrixOperation.NONE)
    fc_state_feature_dense1_mult.name = 'fc_state_feature_dense1_mult'
    fc_state_feature_dense1_b = network.add_constant(weights=weights['state_feature/dense1.bias'], shape=[1, np.shape(weights['state_feature/dense1.bias'])[0]])
    fc_state_feature_dense1_add = network.add_elementwise(fc_state_feature_dense1_mult.get_output(0), fc_state_feature_dense1_b.get_output(0), trt.ElementWiseOperation.SUM)
    
    fc_state_feature_dense2_w = network.add_constant(weights=weights['state_feature/dense2.weight'], shape=np.shape(weights['state_feature/dense2.weight']))
    fc_state_feature_dense2_mult = network.add_matrix_multiply(fc_state_feature_dense1_add.get_output(0), trt.MatrixOperation.NONE, fc_state_feature_dense2_w.get_output(0), trt.MatrixOperation.NONE)
    fc_state_feature_dense2_mult.name = 'fc_state_feature_dense2_mult'
    fc_state_feature_dense2_b = network.add_constant(weights=weights['state_feature/dense2.bias'], shape=[1, np.shape(weights['state_feature/dense2.bias'])[0]])
    fc_state_feature_dense2_add = network.add_elementwise(fc_state_feature_dense2_mult.get_output(0), fc_state_feature_dense2_b.get_output(0), trt.ElementWiseOperation.SUM)
    state_feature_relu2 = network.add_activation(input=fc_state_feature_dense2_add.get_output(0), type=trt.ActivationType.RELU)

    fc_state_feature_dense3_w = network.add_constant(weights=weights['state_feature/dense3.weight'], shape=np.shape(weights['state_feature/dense3.weight']))
    fc_state_feature_dense3_mult = network.add_matrix_multiply(state_feature_relu2.get_output(0), trt.MatrixOperation.NONE, fc_state_feature_dense3_w.get_output(0), trt.MatrixOperation.NONE)
    fc_state_feature_dense3_mult.name = 'fc_state_feature_dense3_mult'
    fc_state_feature_dense3_b = network.add_constant(weights=weights['state_feature/dense3.bias'], shape=[1, np.shape(weights['state_feature/dense3.bias'])[0]])
    fc_state_feature_dense3_add = network.add_elementwise(fc_state_feature_dense3_mult.get_output(0), fc_state_feature_dense3_b.get_output(0), trt.ElementWiseOperation.SUM)

    # expand dim state_feature tensor
    fc_state_feature_reshape = network.add_shuffle(input=fc_state_feature_dense3_add.get_output(0))
    fc_state_feature_reshape.reshape_dims = trt.Dims([ACTION_HORIZON_INFERENCE_INFOGAIN, 32, 1, 1])
    fc_state_feature_reshape.name = 'fc_state_feature_reshape'

    # batch_norm_6:
    cnn_feature_input_permute = network.add_shuffle(input=cnn_feature_input)
    cnn_feature_input_permute.first_transpose = trt.Permutation([2, 0, 1]) # input for convolution layer follows CHW format
    cnn_feature_input_permute.reshape_dims = trt.Dims([1, 32, 34, 60])
    cnn_feature_input_permute.name = 'cnn_feature_input_permute'
    # shift0, scale0, power0 = get_batch_norm_params(weights, 'batch_normalization_6')  
    # batch_norm6 = network.add_scale(cnn_feature_input_permute.get_output(0), trt.ScaleMode.CHANNEL, shift=shift0, scale=scale0, power=power0)
    # batch_norm6.name = 'batch_norm6'
    # batch_norm6_relu = network.add_activation(input=batch_norm6.get_output(0), type=trt.ActivationType.RELU)
    # output_conv2d_0 = network.add_convolution(batch_norm6_relu.get_output(0), 32, (3, 3), weights['output_conv2d_0/weight'], weights['output_conv2d_0/bias'])
    # output_conv2d_0.stride = trt.DimsHW(1, 1)
    # output_conv2d_0.padding = trt.DimsHW(1, 1)

    # print('output_conv2d_0.output:', output_conv2d_0.get_output(0).shape)

    # add
    add1 = network.add_elementwise(cnn_feature_input_permute.get_output(0), fc_state_feature_reshape.get_output(0), trt.ElementWiseOperation.SUM) # (15, 32, 34, 60)
    add1.name = 'add1'
    # print('add1.output:', add1.get_output(0).shape) # (15, 32, 34, 60)

    # # second residual block
    shift1, scale1, power1 = get_batch_norm_params(weights, 'output_batch_normalization_0')    
    output_batch_norm0 = network.add_scale(add1.get_output(0), trt.ScaleMode.CHANNEL, shift=shift1, scale=scale1, power=power1)
    output_batch_norm0_relu = network.add_activation(input=output_batch_norm0.get_output(0), type=trt.ActivationType.RELU)    
    output_conv2d_1 = network.add_convolution(output_batch_norm0_relu.get_output(0), 32, (3, 3), weights['output_conv2d_1/weight'], weights['output_conv2d_1/bias'])
    output_conv2d_1.stride = trt.DimsHW(2, 2)
    output_conv2d_1.padding_mode = trt.PaddingMode.SAME_UPPER

    shift2, scale2, power2 = get_batch_norm_params(weights, 'output_batch_normalization_1')    
    output_batch_norm1 = network.add_scale(output_conv2d_1.get_output(0), trt.ScaleMode.CHANNEL, shift=shift2, scale=scale2, power=power2)
    output_batch_norm1_relu = network.add_activation(input=output_batch_norm1.get_output(0), type=trt.ActivationType.RELU)    
    output_conv2d_2 = network.add_convolution(output_batch_norm1_relu.get_output(0), 32, (3, 3), weights['output_conv2d_2/weight'], weights['output_conv2d_2/bias'])    
    output_conv2d_2.stride = trt.DimsHW(1, 1)
    output_conv2d_2.padding = trt.DimsHW(1, 1)

    output_conv2d_3 = network.add_convolution(add1.get_output(0), 32, (1, 1), weights['output_conv2d_3/weight'], weights['output_conv2d_3/bias'])
    output_conv2d_3.stride = trt.DimsHW(2, 2)
    output_conv2d_3.padding_mode = trt.PaddingMode.SAME_UPPER

    add2 = network.add_elementwise(output_conv2d_2.get_output(0), output_conv2d_3.get_output(0), trt.ElementWiseOperation.SUM)

    # third residual block
    shift3, scale3, power3 = get_batch_norm_params(weights, 'output_batch_normalization_2')    
    output_batch_norm2 = network.add_scale(add2.get_output(0), trt.ScaleMode.CHANNEL, shift=shift3, scale=scale3, power=power3)
    output_batch_norm2_relu = network.add_activation(input=output_batch_norm2.get_output(0), type=trt.ActivationType.RELU)    
    output_conv2d_4 = network.add_convolution(output_batch_norm2_relu.get_output(0), 64, (3, 3), weights['output_conv2d_4/weight'], weights['output_conv2d_4/bias'])
    output_conv2d_4.stride = trt.DimsHW(2, 2)
    output_conv2d_4.padding_mode = trt.PaddingMode.SAME_UPPER

    shift4, scale4, power4 = get_batch_norm_params(weights, 'output_batch_normalization_3')    
    output_batch_norm3 = network.add_scale(output_conv2d_4.get_output(0), trt.ScaleMode.CHANNEL, shift=shift4, scale=scale4, power=power4)
    output_batch_norm3_relu = network.add_activation(input=output_batch_norm3.get_output(0), type=trt.ActivationType.RELU)    
    output_conv2d_5 = network.add_convolution(output_batch_norm3_relu.get_output(0), 64, (3, 3), weights['output_conv2d_5/weight'], weights['output_conv2d_5/bias'])    
    output_conv2d_5.stride = trt.DimsHW(1, 1)
    output_conv2d_5.padding = trt.DimsHW(1, 1)

    output_conv2d_6 = network.add_convolution(add2.get_output(0), 64, (1, 1), weights['output_conv2d_6/weight'], weights['output_conv2d_6/bias'])
    output_conv2d_6.stride = trt.DimsHW(2, 2)
    output_conv2d_6.padding_mode = trt.PaddingMode.SAME_UPPER

    add3 = network.add_elementwise(output_conv2d_5.get_output(0), output_conv2d_6.get_output(0), trt.ElementWiseOperation.SUM)
    add3_relu = network.add_activation(input=add3.get_output(0), type=trt.ActivationType.RELU)
    
    # output
    add3_relu_reshape = network.add_shuffle(input=add3_relu.get_output(0))
    add3_relu_reshape.first_transpose = trt.Permutation([0, 2, 3, 1]) # transpose from CHW to HWC format
    add3_relu_reshape.reshape_dims = trt.Dims([ACTION_HORIZON_INFERENCE_INFOGAIN, 8640])

    info_gain_output_dense0_w = network.add_constant(weights=weights['info_gain_output/dense0.weight'], shape=np.shape(weights['info_gain_output/dense0.weight']))
    info_gain_output_dense0_mult = network.add_matrix_multiply(add3_relu_reshape.get_output(0), trt.MatrixOperation.NONE, info_gain_output_dense0_w.get_output(0), trt.MatrixOperation.NONE)
    info_gain_output_dense0_b = network.add_constant(weights=weights['info_gain_output/dense0.bias'], shape=[1, np.shape(weights['info_gain_output/dense0.bias'])[0]])
    info_gain_output_dense0_add = network.add_elementwise(info_gain_output_dense0_mult.get_output(0), info_gain_output_dense0_b.get_output(0), trt.ElementWiseOperation.SUM)
    info_gain_output_dense0_softplus = network.add_activation(input=info_gain_output_dense0_add.get_output(0), type=trt.ActivationType.SOFTPLUS)

    info_gain_output_dense0_softplus.get_output(0).name = 'output_infogain'
    network.mark_output(tensor=info_gain_output_dense0_softplus.get_output(0))


def build_engine(weights):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, builder.create_builder_config() as config:
        builder.max_batch_size = BATCH_SIZE_PREDICTOR
        config.max_workspace_size = GiB(2)
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
    predictor_model = custom_info_gain_model.get_predictor()

    # build network with Python API
    weights = {}
    weights_action_dense0 = predictor_model.get_layer('action/dense0').get_weights()
    weights['action/dense0.weight'] = weights_action_dense0[0] # (2, 16)
    weights['action/dense0.bias'] = weights_action_dense0[1] # (16, )

    weights_action_dense1 = predictor_model.get_layer('action/dense1').get_weights()
    weights['action/dense1.weight'] = weights_action_dense1[0]
    weights['action/dense1.bias'] = weights_action_dense1[1]

    weights_state_input_dense0 = predictor_model.get_layer('robot_state/dense0').get_weights()
    weights['robot_state/dense0.weight'] = weights_state_input_dense0[0]
    weights['robot_state/dense0.bias'] = weights_state_input_dense0[1]    

    weights_state_input_dense1 = predictor_model.get_layer('robot_state/dense1').get_weights()
    weights['robot_state/dense1.weight'] = weights_state_input_dense1[0]
    weights['robot_state/dense1.bias'] = weights_state_input_dense1[1]

    weights_state_input_dense2 = predictor_model.get_layer('robot_state/dense2').get_weights()
    weights['robot_state/dense2.weight'] = weights_state_input_dense2[0]
    weights['robot_state/dense2.bias'] = weights_state_input_dense2[1]    

    HIDDEN_UNIT = int(np.shape(weights['robot_state/dense2.bias'])[0] / 2)
    print('HIDDEN_UNIT:', HIDDEN_UNIT)

    weights_lstm_W, weights_lstm_R, weights_lstm_B = predictor_model.get_layer('recurrent_layer_robot_state').get_weights()
    # (32, 64) (16, 64) (64,)
    # https://github.com/keras-team/keras/blob/ce2f736bac8e58fc3df27290a589fb8f762e6939/keras/layers/recurrent.py#L2436
    weights['lstm.Wi'] = np.ascontiguousarray(weights_lstm_W[:, :HIDDEN_UNIT]) 
    weights['lstm.Wf'] = np.ascontiguousarray(weights_lstm_W[:, HIDDEN_UNIT:2 * HIDDEN_UNIT])
    weights['lstm.Wc'] = np.ascontiguousarray(weights_lstm_W[:, 2 * HIDDEN_UNIT:3 * HIDDEN_UNIT])
    weights['lstm.Wo'] = np.ascontiguousarray(weights_lstm_W[:, 3 * HIDDEN_UNIT:])

    weights['lstm.Ri'] = np.ascontiguousarray(weights_lstm_R[:, :HIDDEN_UNIT]) 
    weights['lstm.Rf'] = np.ascontiguousarray(weights_lstm_R[:, HIDDEN_UNIT:2 * HIDDEN_UNIT])
    weights['lstm.Rc'] = np.ascontiguousarray(weights_lstm_R[:, 2 * HIDDEN_UNIT:3 * HIDDEN_UNIT])
    weights['lstm.Ro'] = np.ascontiguousarray(weights_lstm_R[:, 3 * HIDDEN_UNIT:])

    weights['lstm.Bi'] = np.ascontiguousarray(weights_lstm_B[:HIDDEN_UNIT]) 
    weights['lstm.Bf'] = np.ascontiguousarray(weights_lstm_B[HIDDEN_UNIT:2 * HIDDEN_UNIT])
    weights['lstm.Bc'] = np.ascontiguousarray(weights_lstm_B[2 * HIDDEN_UNIT:3 * HIDDEN_UNIT])
    weights['lstm.Bo'] = np.ascontiguousarray(weights_lstm_B[3 * HIDDEN_UNIT:])

    weights_state_feature_dense0 = predictor_model.get_layer('state_feature/dense0').get_weights()
    weights['state_feature/dense0.weight'] = weights_state_feature_dense0[0]
    weights['state_feature/dense0.bias'] = weights_state_feature_dense0[1]    

    weights_state_feature_dense1 = predictor_model.get_layer('state_feature/dense1').get_weights()
    weights['state_feature/dense1.weight'] = weights_state_feature_dense1[0]
    weights['state_feature/dense1.bias'] = weights_state_feature_dense1[1]

    weights_state_feature_dense2 = predictor_model.get_layer('state_feature/dense2').get_weights()
    weights['state_feature/dense2.weight'] = weights_state_feature_dense2[0]
    weights['state_feature/dense2.bias'] = weights_state_feature_dense2[1]

    weights_state_feature_dense3 = predictor_model.get_layer('state_feature/dense3').get_weights()
    weights['state_feature/dense3.weight'] = weights_state_feature_dense3[0]
    weights['state_feature/dense3.bias'] = weights_state_feature_dense3[1]

    # batch_normalization
    # https://scortex.io/batch-norm-folding-an-easy-way-to-improve-your-network-speed/
    # weights['batch_normalization_6/gamma'], weights['batch_normalization_6/beta'], weights['batch_normalization_6/mean'], weights[
    #     'batch_normalization_6/var'] = predictor_model.get_layer('batch_normalization_6').get_weights()

    weights['output_batch_normalization_0/gamma'], weights['output_batch_normalization_0/beta'], weights['output_batch_normalization_0/mean'], weights[
        'output_batch_normalization_0/var'] = predictor_model.get_layer('output_batch_normalization_0').get_weights()

    weights['output_batch_normalization_1/gamma'], weights['output_batch_normalization_1/beta'], weights['output_batch_normalization_1/mean'], weights[
        'output_batch_normalization_1/var'] = predictor_model.get_layer('output_batch_normalization_1').get_weights()

    weights['output_batch_normalization_2/gamma'], weights['output_batch_normalization_2/beta'], weights['output_batch_normalization_2/mean'], weights[
        'output_batch_normalization_2/var'] = predictor_model.get_layer('output_batch_normalization_2').get_weights()

    weights['output_batch_normalization_3/gamma'], weights['output_batch_normalization_3/beta'], weights['output_batch_normalization_3/mean'], weights[
        'output_batch_normalization_3/var'] = predictor_model.get_layer('output_batch_normalization_3').get_weights()                

    # output_conv2d
    # https://on-demand.gputechconf.com/gtc-cn/2019/pdf/CN9577/presentation.pdf
    # weights_output_conv2d_0 = predictor_model.get_layer('output_conv2d_0').get_weights()
    # weights['output_conv2d_0/weight'] = np.ascontiguousarray(weights_output_conv2d_0[0].transpose((3, 2, 0, 1)).reshape(-1))
    # weights['output_conv2d_0/bias'] = weights_output_conv2d_0[1]

    weights_output_conv2d_1 = predictor_model.get_layer('output_conv2d_1').get_weights()
    weights['output_conv2d_1/weight'] = np.ascontiguousarray(weights_output_conv2d_1[0].transpose((3, 2, 0, 1)).reshape(-1))
    weights['output_conv2d_1/bias'] = weights_output_conv2d_1[1]

    weights_output_conv2d_2 = predictor_model.get_layer('output_conv2d_2').get_weights()
    weights['output_conv2d_2/weight'] = np.ascontiguousarray(weights_output_conv2d_2[0].transpose((3, 2, 0, 1)).reshape(-1))
    weights['output_conv2d_2/bias'] = weights_output_conv2d_2[1]

    weights_output_conv2d_3 = predictor_model.get_layer('output_conv2d_3').get_weights()
    weights['output_conv2d_3/weight'] = np.ascontiguousarray(weights_output_conv2d_3[0].transpose((3, 2, 0, 1)).reshape(-1))
    weights['output_conv2d_3/bias'] = weights_output_conv2d_3[1]

    weights_output_conv2d_4 = predictor_model.get_layer('output_conv2d_4').get_weights()
    weights['output_conv2d_4/weight'] = np.ascontiguousarray(weights_output_conv2d_4[0].transpose((3, 2, 0, 1)).reshape(-1))
    weights['output_conv2d_4/bias'] = weights_output_conv2d_4[1]

    weights_output_conv2d_5 = predictor_model.get_layer('output_conv2d_5').get_weights()
    weights['output_conv2d_5/weight'] = np.ascontiguousarray(weights_output_conv2d_5[0].transpose((3, 2, 0, 1)).reshape(-1))
    weights['output_conv2d_5/bias'] = weights_output_conv2d_5[1]

    weights_output_conv2d_6 = predictor_model.get_layer('output_conv2d_6').get_weights()
    weights['output_conv2d_6/weight'] = np.ascontiguousarray(weights_output_conv2d_6[0].transpose((3, 2, 0, 1)).reshape(-1))
    weights['output_conv2d_6/bias'] = weights_output_conv2d_6[1]

    weights_info_gain_output_dense0 = predictor_model.get_layer('info_gain_output/dense0').get_weights()
    weights['info_gain_output/dense0.weight'] = weights_info_gain_output_dense0[0]
    weights['info_gain_output/dense0.bias'] = weights_info_gain_output_dense0[1]

    # build, test and save the engine
    with build_engine(weights) as engine:
        with open("infogain_predictor_engine_fp32.trt", "wb") as f:
            f.write(engine.serialize())

    print("Done saving!")      