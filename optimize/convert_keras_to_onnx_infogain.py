import os
import sys
sys.path.append('.')
sys.path.append('./train')
import numpy as np
import tensorflow as tf 
import onnx
# import keras2onnx
import tf2onnx
from utilities import bcolors
from training import *
from config import *
import gflags
from common_flags import FLAGS

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

    BATCH_SIZE_IMAGE_FEATURE = 1
    BATCH_SIZE_PREDICTOR = NUM_SEQUENCE_TO_EVALUATE

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
    image_feature_model = custom_info_gain_model.get_feature_extractor()
    predictor_model = custom_info_gain_model.get_predictor()

    # image_feature_onnx_model = keras2onnx.convert_keras(model=image_feature_model, name=image_feature_model.name, target_opset=9) 
    # predictor_onnx_model = keras2onnx.convert_keras(model=predictor_model, name=predictor_model.name, target_opset=9) # TODO: still not works!
    image_feature_onnx_model, _ = tf2onnx.convert.from_keras(model=image_feature_model)
    predictor_onnx_model, _ = tf2onnx.convert.from_keras(model=predictor_model)

    # Configure ONNX File Batch Size
    image_feature_inputs = image_feature_onnx_model.graph.input
    print('image_feature_inputs:', image_feature_inputs)
    for input in image_feature_inputs:
        dim1 = input.type.tensor_type.shape.dim[0]
        dim1.dim_value = BATCH_SIZE_IMAGE_FEATURE

    predictor_inputs = predictor_onnx_model.graph.input
    print('predictor_inputs:', predictor_inputs)
    for input in predictor_inputs:
        dim1 = input.type.tensor_type.shape.dim[0]
        dim1.dim_value = BATCH_SIZE_PREDICTOR
    print('predictor_inputs after:', predictor_inputs)

    model_name = "infogain_image_feature_onnx_model.onnx"
    onnx.save_model(image_feature_onnx_model, model_name)

    model_name = "infogain_predictor_onnx_model.onnx"
    onnx.save_model(predictor_onnx_model, model_name)

    print("Done saving!")      