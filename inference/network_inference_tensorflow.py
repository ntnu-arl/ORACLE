import sys
sys.path.append('.')
sys.path.append('./train')
from config import *
from training import *
from network_inference_base_class import NetworkInferenceBaseClass, InfoNetworkInferenceBaseClass 

import numpy as np

import tensorflow as tf

class NetworkInferenceTensorflowV2(NetworkInferenceBaseClass):
    def __init__(self):
        custom_predictor_model_tmp_list = []
        self.custom_predictor_model_list = []
        for i in range(N_E):
            custom_predictor_model_tmp = TrainCPN(depth_image_shape=DI_SHAPE)
            custom_predictor_model = InferenceCPN(depth_image_shape=DI_SHAPE)
            
            custom_predictor_model_tmp_list.append(custom_predictor_model_tmp)
            self.custom_predictor_model_list.append(custom_predictor_model)
            
            if i == 0:
                predictor_model = custom_predictor_model.get_model()
                predictor_model.summary()
                plot_model(predictor_model, to_file='collision_model_plot_loaded.png', show_shapes=True, show_layer_names=True)

        # Load the weights
        h_input_img = np.zeros((1, DI_SHAPE[0], DI_SHAPE[1], DI_SHAPE[2]), dtype = np.float32)

        h_input_state = np.zeros((1, STATE_INPUT_SHAPE), dtype = np.float32)
        h_inputs = [h_input_state, h_input_img]

        action_input = np.zeros((1, ACTION_HORIZON, ACTION_SHAPE_EVALUATE), dtype = np.float32)    
        
        for i in range(N_E):        
            checkpoint_path = CPN_TF_CHECKPOINT_PATH[i]
            custom_predictor_model_tmp_list[i](h_inputs + [action_input])
            custom_predictor_model_tmp_list[i].load_weights(checkpoint_path)
            self.custom_predictor_model_list[i].load_model(custom_predictor_model_tmp_list[i].get_model())

        # lists to save the networks' outputs
        self.h_output_cnn_list = []
        self.h_output_combiner_list = []
        self.h_output_rnn_list = []
        DI_FEATURE_SIZE = self.get_di_feature_size()
        INITIAL_STATE_SIZE = self.custom_predictor_model_list[0].get_initial_state_size()
        for i in range(N_E):
            h_output_cnn_array = np.array([1, DI_FEATURE_SIZE], np.float32)
            self.h_output_cnn_list.append(np.ascontiguousarray(h_output_cnn_array, np.float32))
            
            h_output_combiner_array = np.array([N_SIGMA, 2*INITIAL_STATE_SIZE], np.float32)
            self.h_output_combiner_list.append(np.ascontiguousarray(h_output_combiner_array, np.float32))
            
            h_output_rnn_array = np.array([N_SIGMA * NUM_SEQUENCE_TO_EVALUATE, ACTION_HORIZON, 1], np.float32)
            self.h_output_rnn_list.append(np.ascontiguousarray(h_output_rnn_array, np.float32))

    def call_cnn_only(self, inputs):
        for i in range(N_E):
            # inputs: [image]
            self.h_output_cnn_list[i] = self.custom_predictor_model_list[i].call_cnn_only(inputs[0])
        return self.h_output_cnn_list

    def call_depth_state_combiner(self, inputs):
        for i in range(N_E):
            # inputs: [robot's state, list of di_feature]
            self.h_output_combiner_list[i] = self.custom_predictor_model_list[i].call_depth_state_combiner([inputs[0],inputs[1][i]])
        return self.h_output_combiner_list

    def call_recurrent_net(self, inputs):
        for i in range(N_E):
            # inputs: [list of initial_h, list of initial_c, action_seq]
            self.h_output_rnn_list[i] = self.custom_predictor_model_list[i].call_recurrent_net([inputs[0][i], inputs[1][i], inputs[2]])
        return self.h_output_rnn_list

    def get_di_feature_size(self):
        return self.custom_predictor_model_list[0].get_di_feature_size()

class seVAENetworkInferenceTensorflow(NetworkInferenceBaseClass):
    def __init__(self):
        custom_predictor_model_tmp_list = []
        self.custom_predictor_model_list = []
        for i in range(N_E):
            custom_predictor_model_tmp = TrainCPNseVAE(DI_LATENT_SIZE)
            custom_predictor_model = InferenceCPNseVAE(DI_LATENT_SIZE)
            
            custom_predictor_model_tmp_list.append(custom_predictor_model_tmp)
            self.custom_predictor_model_list.append(custom_predictor_model)
            
            if i == 0:
                predictor_model = custom_predictor_model.get_model()
                predictor_model.summary()
                plot_model(predictor_model, to_file='collision_model_plot_loaded.png', show_shapes=True, show_layer_names=True)


        # Load the weights
        h_input_latent_vector = np.zeros((1, DI_LATENT_SIZE), dtype = np.float32)

        h_input_state = np.zeros((1, STATE_INPUT_SHAPE), dtype = np.float32)
        h_inputs = [h_input_state, h_input_latent_vector]

        action_input = np.zeros((1, ACTION_HORIZON, ACTION_SHAPE_EVALUATE), dtype = np.float32)    
        
        for i in range(N_E):
            checkpoint_path = seVAE_CPN_TF_CHECKPOINT_PATH[i]
            custom_predictor_model_tmp_list[i](h_inputs + [action_input])
            custom_predictor_model_tmp_list[i].load_weights(checkpoint_path)
            self.custom_predictor_model_list[i].load_model(custom_predictor_model_tmp_list[i].get_model())

        # lists to save the networks' outputs
        self.h_output_combiner_list = []
        self.h_output_rnn_list = []
        INITIAL_STATE_SIZE = self.custom_predictor_model_list[0].get_initial_state_size()
        for i in range(N_E):
            h_output_combiner_array = np.array([N_SIGMA, 2*INITIAL_STATE_SIZE], np.float32)
            self.h_output_combiner_list.append(np.ascontiguousarray(h_output_combiner_array, np.float32))
            
            h_output_rnn_array = np.array([N_SIGMA * NUM_SEQUENCE_TO_EVALUATE, ACTION_HORIZON, 1], np.float32)
            self.h_output_rnn_list.append(np.ascontiguousarray(h_output_rnn_array, np.float32))


    def call_cnn_only(self, inputs):
        pass

    def call_depth_state_combiner(self, inputs):
        for i in range(N_E):
            # inputs: [robot's state, list of di_feature]
            self.h_output_combiner_list[i] = self.custom_predictor_model_list[i].call_depth_state_combiner([inputs[0], inputs[1][i]])
        return self.h_output_combiner_list

    def call_recurrent_net(self, inputs):
        for i in range(N_E):
            # inputs: [list of initial_h, list of initial_c, action_seq]
            self.h_output_rnn_list[i] = self.custom_predictor_model_list[i].call_recurrent_net([inputs[0][i], inputs[1][i], inputs[2]])
        return self.h_output_rnn_list

    def get_di_feature_size(self):
        return DI_LATENT_SIZE

class InfoNetworkInferenceTensorflow(InfoNetworkInferenceBaseClass):
    def __init__(self):
        custom_predictor_model = TrainIPN(depth_image_shape=DI_WITH_MASK_SHAPE)
        self.custom_info_gain_model = InferenceIPN(depth_image_shape=DI_WITH_MASK_SHAPE)                

        info_gain_model = self.custom_info_gain_model.get_model()

        info_gain_model.summary()
        plot_model(info_gain_model, to_file='info_gain_model_plot_loaded.png', show_shapes=True, show_layer_names=True)
        plot_model(self.custom_info_gain_model.get_predictor(), to_file='info_gain_predictor_model_plot_loaded.png', show_shapes=True, show_layer_names=True)
        checkpoint_path = IPN_TF_CHECKPOINT_PATH

        # Load the weights
        h_input_img = np.zeros((1, DI_SHAPE[0], DI_SHAPE[1], DI_WITH_MASK_SHAPE[2]), dtype = np.float32)
        h_input_state = np.zeros((1, STATE_INPUT_SHAPE), dtype = np.float32)
        h_inputs = [h_input_state, h_input_img]

        action_input = np.zeros((1, ACTION_HORIZON, ACTION_SHAPE_EVALUATE), dtype = np.float32)    
        custom_predictor_model(h_inputs + [action_input])
        custom_predictor_model.load_weights(checkpoint_path)
        self.custom_info_gain_model.load_model(custom_predictor_model.get_model())

    def call_feature_extractor(self, inputs):
        return self.custom_info_gain_model.call_feature_extractor(inputs)

    def call_recurrent_net(self, inputs):
        return self.custom_info_gain_model.call_predictor_net(inputs)

    def get_di_feature_size(self):
        return self.custom_info_gain_model.get_di_feature_size()
