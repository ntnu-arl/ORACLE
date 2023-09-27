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
BATCH_SIZE_RNN = N_SIGMA * NUM_SEQUENCE_TO_EVALUATE

def populate_network(network, weights):
    # Initialize data, hiddenIn, cellIn, and seqLenIn inputs into RNN Layer
    HIDDEN_UNIT = np.shape(weights['output_dense_1.weight'])[0]
    initial_state_h = network.add_input(name='initial_state_h', dtype=DTYPE, shape=trt.Dims([HIDDEN_UNIT])) # TODO: replace 64 with with HIDDEN_UNIT
    initial_state_c = network.add_input(name='initial_state_c', dtype=DTYPE, shape=trt.Dims([HIDDEN_UNIT]))
    action_input = network.add_input(name='action_input', dtype=DTYPE, shape=trt.Dims2(ACTION_HORIZON, ACTION_SHAPE_EVALUATE))

    # create action_input NN    
    # fc_action_dense0_w = network.add_constant(weights=weights['action/dense0.weight'], shape=trt.Dims2(2, 16))
    fc_action_dense0_w = network.add_constant(weights=weights['action/dense0.weight'], shape=np.shape(weights['action/dense0.weight']))
    fc_action_dense0_mult = network.add_matrix_multiply(action_input, trt.MatrixOperation.NONE, fc_action_dense0_w.get_output(0), trt.MatrixOperation.NONE)
    # fc_action_dense0_b = network.add_constant(weights=weights['action/dense0.bias'], shape=trt.Dims2(1, 16))
    fc_action_dense0_b = network.add_constant(weights=weights['action/dense0.bias'], shape=[1, np.shape(weights['action/dense0.bias'])[0]])
    fc_action_dense0_add = network.add_elementwise(fc_action_dense0_mult.get_output(0), fc_action_dense0_b.get_output(0), trt.ElementWiseOperation.SUM)
    action_relu0 = network.add_activation(input=fc_action_dense0_add.get_output(0), type=trt.ActivationType.RELU) # (18,16)

    # fc_action_dense1_w = network.add_constant(weights=weights['action/dense1.weight'], shape=trt.Dims2(16, 16))
    fc_action_dense1_w = network.add_constant(weights=weights['action/dense1.weight'], shape=np.shape(weights['action/dense1.weight']))
    fc_action_dense1_mult = network.add_matrix_multiply(action_relu0.get_output(0), trt.MatrixOperation.NONE, fc_action_dense1_w.get_output(0), trt.MatrixOperation.NONE)
    # fc_action_dense1_b = network.add_constant(weights=weights['action/dense1.bias'], shape=trt.Dims2(1, 16))
    fc_action_dense1_b = network.add_constant(weights=weights['action/dense1.bias'], shape=[1, np.shape(weights['action/dense1.bias'])[0]])
    action_fc1 = network.add_elementwise(fc_action_dense1_mult.get_output(0), fc_action_dense1_b.get_output(0), trt.ElementWiseOperation.SUM)

    # create an RNN layer w/ 1 layers and 64 hidden states    
    rnn = network.add_rnn_v2(input=action_fc1.get_output(0), layer_count=1, hidden_size=HIDDEN_UNIT, max_seq_length=ACTION_HORIZON, 
                            op=trt.RNNOperation.LSTM)                        
    rnn.input_mode = trt.RNNInputMode.LINEAR
    rnn.direction = trt.RNNDirection.UNIDIRECTION

    # Set RNNv2 optional inputs
    initial_state_h_reshape = network.add_shuffle(input=initial_state_h)
    initial_state_h_reshape.reshape_dims = trt.Dims([1, HIDDEN_UNIT]) 
    initial_state_c_reshape = network.add_shuffle(input=initial_state_c)
    initial_state_c_reshape.reshape_dims = trt.Dims([1, HIDDEN_UNIT])       
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

    # create action_output NN
    # fc_output_dense1_w = network.add_constant(weights=weights['output_dense_1.weight'], shape=trt.Dims2(64, 32))
    fc_output_dense1_w = network.add_constant(weights=weights['output_dense_1.weight'], shape=np.shape(weights['output_dense_1.weight']))
    fc_output_dense1_mult = network.add_matrix_multiply(rnn.get_output(0), trt.MatrixOperation.NONE, fc_output_dense1_w.get_output(0), trt.MatrixOperation.NONE)
    # fc_output_dense1_b = network.add_constant(weights=weights['output_dense_1.bias'], shape=trt.Dims2(1, 32))
    fc_output_dense1_b = network.add_constant(weights=weights['output_dense_1.bias'], shape=[1, np.shape(weights['output_dense_1.bias'])[0]])
    fc_output_dense1_add = network.add_elementwise(fc_output_dense1_mult.get_output(0), fc_output_dense1_b.get_output(0), trt.ElementWiseOperation.SUM)
    output_relu1 = network.add_activation(input=fc_output_dense1_add.get_output(0), type=trt.ActivationType.RELU)

    # fc_output_dense2_w = network.add_constant(weights=weights['output_dense_2.weight'], shape=trt.Dims2(32, 1))
    fc_output_dense2_w = network.add_constant(weights=weights['output_dense_2.weight'], shape=np.shape(weights['output_dense_2.weight']))
    fc_output_dense2_mult = network.add_matrix_multiply(output_relu1.get_output(0), trt.MatrixOperation.NONE, fc_output_dense2_w.get_output(0), trt.MatrixOperation.NONE)
    # fc_output_dense2_b = network.add_constant(weights=weights['output_dense_2.bias'], shape=trt.Dims2(1, 1))
    fc_output_dense2_b = network.add_constant(weights=weights['output_dense_2.bias'], shape=[1, np.shape(weights['output_dense_2.bias'])[0]])
    fc_output_dense2_add = network.add_elementwise(fc_output_dense2_mult.get_output(0), fc_output_dense2_b.get_output(0), trt.ElementWiseOperation.SUM)
    output_sigmoid2 = network.add_activation(input=fc_output_dense2_add.get_output(0), type=trt.ActivationType.SIGMOID)

    output_sigmoid2.get_output(0).name = 'output_sigmoid2'
    network.mark_output(tensor=output_sigmoid2.get_output(0))  

def build_engine(weights):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, builder.create_builder_config() as config:
        builder.max_batch_size = BATCH_SIZE_RNN
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

    custom_predictor_model_tmp = TrainCPNseVAE(DI_LATENT_SIZE)
    custom_predictor_model = InferenceCPNseVAE(DI_LATENT_SIZE)
    
    h_input_latent_vector = np.zeros((1, DI_LATENT_SIZE), dtype = np.float32)
    h_input_state = np.zeros((1, STATE_INPUT_SHAPE), dtype = np.float32)
    h_inputs = [h_input_state, h_input_latent_vector]

    action_input = np.zeros((1, ACTION_HORIZON, ACTION_SHAPE_EVALUATE), dtype = np.float32)
    custom_predictor_model_tmp(h_inputs + [action_input])    
    # Loads the weights
    custom_predictor_model_tmp.load_weights(checkpoint_path)
    custom_predictor_model.load_model(custom_predictor_model_tmp.get_model())

    # get the CNN and RNN keras models
    rnn_model = custom_predictor_model.get_rnn()

    # build network with Python API
    weights = {}
    weights_action_dense0 = rnn_model.get_layer('action/dense0').get_weights()
    weights['action/dense0.weight'] = weights_action_dense0[0] # (2, 16)
    weights['action/dense0.bias'] = weights_action_dense0[1] # (16, )

    weights_action_dense1 = rnn_model.get_layer('action/dense1').get_weights()
    weights['action/dense1.weight'] = weights_action_dense1[0]
    weights['action/dense1.bias'] = weights_action_dense1[1]

    weights_output_dense1 = rnn_model.get_layer('output_dense_1').get_weights()
    weights['output_dense_1.weight'] = weights_output_dense1[0]
    weights['output_dense_1.bias'] = weights_output_dense1[1]    

    weights_output_dense2 = rnn_model.get_layer('output_dense_2').get_weights()
    weights['output_dense_2.weight'] = weights_output_dense2[0]
    weights['output_dense_2.bias'] = weights_output_dense2[1]

    HIDDEN_UNIT = np.shape(weights['output_dense_1.weight'])[0]

    weights_lstm_W, weights_lstm_R, weights_lstm_B = rnn_model.get_layer('recurrent_layer').get_weights()
    # (16, 256) (64, 256) (256,)
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

    # build, test and save the engine
    with build_engine(weights) as engine:
        with open("VAE_rnn_engine_fp32_lstm.trt", "wb") as f:
            f.write(engine.serialize())

    print("Done saving!")      