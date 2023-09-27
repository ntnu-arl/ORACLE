import sys
sys.path.append('.')
sys.path.append('./train')
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from config import *
from training import *
import numpy as np
from network_inference_base_class import NetworkInferenceBaseClass, InfoNetworkInferenceBaseClass

import timeit

class NetworkInferenceTensorRTV2(NetworkInferenceBaseClass):
    def __init__(self):
        self.context_cnn_list = []
        self.context_combiner_list = []
        self.context_rnn_list = []

        for i in range(N_E):
            ## Load TensorRT CNN model
            f_cnn = open(CPN_TRT_CHECKPOINT_PATH[i] + "/collision_cnn_only_engine_fp32.trt", "rb")
            runtime_cnn = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

            engine_cnn = runtime_cnn.deserialize_cuda_engine(f_cnn.read())
            context_cnn = engine_cnn.create_execution_context()

            ## Load TensorRT combiner model
            f_combiner = open(CPN_TRT_CHECKPOINT_PATH[i] + "/collision_depth_state_combiner_engine_fp32.trt", "rb")
            runtime_combiner = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

            engine_combiner = runtime_combiner.deserialize_cuda_engine(f_combiner.read())
            context_combiner = engine_combiner.create_execution_context()

            ## Load TensorRT RNN model
            f_rnn = open(CPN_TRT_CHECKPOINT_PATH[i] + "/collision_rnn_engine_fp32_lstm.trt", "rb")
            runtime_rnn = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

            engine_rnn = runtime_rnn.deserialize_cuda_engine(f_rnn.read())
            context_rnn = engine_rnn.create_execution_context()

            self.context_cnn_list.append(context_cnn)
            self.context_combiner_list.append(context_combiner)
            self.context_rnn_list.append(context_rnn)
        
        input_name = engine_cnn.get_binding_name(0)
        print('engine_cnn.get_binding_name', input_name)
        print('context_cnn.get_binding_shape', context_cnn.get_binding_shape(0))

        for binding in range(2):
            input_name = engine_combiner.get_binding_name(binding)
            if input_name == 'di_feature':
                self.DI_FEATURE_SIZE = context_combiner.get_binding_shape(binding)[0]
                print('DI_FEATURE_SIZE:', self.DI_FEATURE_SIZE)
            print('engine_combiner.get_binding_name', input_name)
            print('context_combiner.get_binding_shape', context_combiner.get_binding_shape(binding))
        for i in range(N_E):
            assert self.context_combiner_list[i].all_binding_shapes_specified

        for binding in range(3):
            input_name = engine_rnn.get_binding_name(binding)
            if input_name == 'initial_state_h':
                INITIAL_STATE_SIZE = context_rnn.get_binding_shape(binding)[0]
                print('INITIAL_STATE_SIZE:', INITIAL_STATE_SIZE)            
            print('engine_rnn.get_binding_name', input_name)
            print('context_rnn.get_binding_shape', context_rnn.get_binding_shape(binding))
        for i in range(N_E):
            assert self.context_rnn_list[i].all_binding_shapes_specified

        ## Allocate device memory for inputs and outputs of CNN-only network
        self.target_dtype = np.float32
        h_inputs_cnn = [np.zeros((1, DI_SHAPE[0], DI_SHAPE[1], DI_SHAPE[2]), dtype = np.float32)]
        self.d_inputs_cnn = [cuda.mem_alloc(h_input.nbytes) for h_input in h_inputs_cnn]    
        self.h_output_cnn_list = []
        self.d_output_cnn_list = []
        self.bindings_cnn_list = []
        self.stream_collision_list = []
        for i in range(N_E):  
            h_output_cnn = cuda.pagelocked_empty([1, self.DI_FEATURE_SIZE], dtype = self.target_dtype)
            self.h_output_cnn_list.append(h_output_cnn)        
            d_output_cnn = cuda.mem_alloc(h_output_cnn.nbytes)
            self.d_output_cnn_list.append(d_output_cnn)   

            bindings_cnn = [int(d_input) for d_input in self.d_inputs_cnn] + [int(self.d_output_cnn_list[i])]
            self.bindings_cnn_list.append(bindings_cnn)

            stream_collision = cuda.Stream()
            self.stream_collision_list.append(stream_collision)

        ## Allocate device memory for inputs and outputs of combiner network
        h_input_state = np.zeros((N_SIGMA, STATE_INPUT_SHAPE), dtype = np.float32)
        h_di_feature = np.zeros((N_SIGMA, self.DI_FEATURE_SIZE), dtype = np.float32)
        h_inputs_combiner = [h_input_state, h_di_feature]
        
        self.d_inputs_combiner_list = []
        self.h_output_combiner_list = []
        self.d_output_combiner_list = []
        self.bindings_combiner_list = []
        for i in range(N_E):
            d_inputs_combiner = [cuda.mem_alloc(h_input.nbytes) for h_input in h_inputs_combiner]
            self.d_inputs_combiner_list.append(d_inputs_combiner)

            h_output_combiner = cuda.pagelocked_empty([N_SIGMA, 2*INITIAL_STATE_SIZE], dtype = self.target_dtype)          
            d_output_combiner = cuda.mem_alloc(h_output_combiner.nbytes)
            self.h_output_combiner_list.append(h_output_combiner)
            self.d_output_combiner_list.append(d_output_combiner)

            bindings_combiner = [int(d_input) for d_input in d_inputs_combiner] + [int(d_output_combiner)]
            self.bindings_combiner_list.append(bindings_combiner)

        # self.stream_combiner = cuda.Stream()

        ## Allocate device memory for inputs and outputs of RNN network
        h_input_state_h = np.zeros((N_SIGMA * NUM_SEQUENCE_TO_EVALUATE, INITIAL_STATE_SIZE), dtype = np.float32)
        h_input_state_c = np.zeros((N_SIGMA * NUM_SEQUENCE_TO_EVALUATE, INITIAL_STATE_SIZE), dtype = np.float32)
        action_seq_expand = np.zeros((N_SIGMA * NUM_SEQUENCE_TO_EVALUATE, ACTION_HORIZON, ACTION_SHAPE_EVALUATE), dtype = np.float32)
        h_inputs_rnn = [h_input_state_h, h_input_state_c, action_seq_expand]
        # h_inputs_rnn = [self.action_seq_expand]
        # print('h_inputs_rnn', h_inputs_rnn)

        self.d_inputs_rnn_list = []
        self.h_output_rnn_list = []
        self.d_output_rnn_list = []
        self.bindings_rnn_list = []
        for i in range(N_E):
            d_inputs_rnn = [cuda.mem_alloc(h_input.nbytes) for h_input in h_inputs_rnn]
            self.d_inputs_rnn_list.append(d_inputs_rnn)
          
            h_output_rnn = cuda.pagelocked_empty([N_SIGMA * NUM_SEQUENCE_TO_EVALUATE, ACTION_HORIZON, 1], dtype = self.target_dtype)
            self.h_output_rnn_list.append(h_output_rnn)
            # h_output_rnn = cuda.pagelocked_empty([N_E * N_SIGMA * NUM_SEQUENCE_TO_EVALUATE, ACTION_HORIZON, 16], dtype = self.target_dtype)
            d_output_rnn = cuda.mem_alloc(h_output_rnn.nbytes)
            self.d_output_rnn_list.append(d_output_rnn)

            bindings_rnn = [int(d_input) for d_input in d_inputs_rnn] + [int(d_output_rnn)]
            self.bindings_rnn_list.append(bindings_rnn)

        # self.stream_rnn = cuda.Stream()

        print("Warming up...")
        self.call_cnn_only(h_inputs_cnn)
        self.call_depth_state_combiner(h_inputs_combiner)
        self.call_recurrent_net(h_inputs_rnn)

        print("Done warming up!")
    
    def call_cnn_only(self, batch):
        for idx in range(len(batch)): # same image for every networks -> transfer once, use first stream only
            # Transfer input data to device
            cuda.memcpy_htod_async(self.d_inputs_cnn[idx], batch[idx], self.stream_collision_list[0])
        for i in range(N_E):
            # Execute model converted from ONNX: explicit batch size
            # self.context_cnn_list[i].execute_async_v2(self.bindings_cnn_list[i], self.stream_collision_list[i].handle, None)
            
            # Execute model converted directly from TF: implicit batch size
            self.context_cnn_list[i].execute_async(batch_size=1, bindings=self.bindings_cnn_list[i], stream_handle=self.stream_collision_list[i].handle, input_consumed=None)
            
            # Synchronize the stream ? https://github.com/NVIDIA/TensorRT/blob/master/demo/BERT/inference.ipynb
            # self.stream_collision_list[i].synchronize()
            # Transfer predictions back
            cuda.memcpy_dtoh_async(self.h_output_cnn_list[i], self.d_output_cnn_list[i], self.stream_collision_list[i])
        
        for i in range(N_E):
            # Syncronize threads
            self.stream_collision_list[i].synchronize()
        
        return self.h_output_cnn_list

    def call_depth_state_combiner(self, batch):
        for i in range(N_E):
            for idx in range(len(batch)):
                # Transfer input data to device
                if idx == 0: # robot's state (same for every NNs)
                    cuda.memcpy_htod_async(self.d_inputs_combiner_list[i][idx], batch[idx], self.stream_collision_list[i])
                else:
                    cuda.memcpy_htod_async(self.d_inputs_combiner_list[i][idx], batch[idx][i], self.stream_collision_list[i])

            # Execute model
            self.context_combiner_list[i].execute_async(
                batch_size=N_SIGMA, bindings=self.bindings_combiner_list[i], stream_handle=self.stream_collision_list[i].handle, input_consumed=None)
            # Synchronize the stream ? https://github.com/NVIDIA/TensorRT/blob/master/demo/BERT/inference.ipynb
            # self.stream_collision_list[i].synchronize()
            # Transfer predictions back
            cuda.memcpy_dtoh_async(self.h_output_combiner_list[i], self.d_output_combiner_list[i], self.stream_collision_list[i])
        
        for i in range(N_E):
            # Syncronize threads
            self.stream_collision_list[i].synchronize()

        return self.h_output_combiner_list

    def call_recurrent_net(self, batch):
        for i in range(N_E):
            for idx in range(len(batch)):
                # Transfer input data to device
                if idx == 2: # action_seq (same for every NNs)
                    cuda.memcpy_htod_async(self.d_inputs_rnn_list[i][idx], batch[idx], self.stream_collision_list[i])
                else:
                    cuda.memcpy_htod_async(self.d_inputs_rnn_list[i][idx], batch[idx][i], self.stream_collision_list[i])
            # Execute model
            self.context_rnn_list[i].execute_async(batch_size=N_SIGMA * NUM_SEQUENCE_TO_EVALUATE,
                                                   bindings=self.bindings_rnn_list[i], stream_handle=self.stream_collision_list[i].handle, input_consumed=None)
            # Synchronize the stream ? https://github.com/NVIDIA/TensorRT/blob/master/demo/BERT/inference.ipynb
            # self.stream_collision_list[i].synchronize()
            # Transfer predictions back
            cuda.memcpy_dtoh_async(self.h_output_rnn_list[i], self.d_output_rnn_list[i], self.stream_collision_list[i])
        
        for i in range(N_E):
            # Syncronize threads
            self.stream_collision_list[i].synchronize()
        
        return self.h_output_rnn_list

    def get_di_feature_size(self):
        return self.DI_FEATURE_SIZE


class seVAENetworkInferenceTensorRTV2(NetworkInferenceBaseClass):
    def __init__(self):
        self.context_combiner_list = []
        self.context_rnn_list = []

        for i in range(N_E):
            # Load TensorRT combiner model
            f_combiner = open(seVAE_CPN_TRT_CHECKPOINT_PATH[i] + "/VAE_depth_state_combiner_engine_fp32.trt", "rb")
            runtime_combiner = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

            engine_combiner = runtime_combiner.deserialize_cuda_engine(f_combiner.read())
            context_combiner = engine_combiner.create_execution_context()

            # Load TensorRT RNN model
            f_rnn = open(seVAE_CPN_TRT_CHECKPOINT_PATH[i] + "/VAE_rnn_engine_fp32_lstm.trt", "rb")
            runtime_rnn = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

            engine_rnn = runtime_rnn.deserialize_cuda_engine(f_rnn.read())
            context_rnn = engine_rnn.create_execution_context()

            self.context_combiner_list.append(context_combiner)
            self.context_rnn_list.append(context_rnn)
        
        for binding in range(2):
            input_name = engine_combiner.get_binding_name(binding)
            if input_name == 'depth_latent_input':
                self.DI_FEATURE_SIZE = context_combiner.get_binding_shape(binding)[0]
                print('DI_FEATURE_SIZE:', self.DI_FEATURE_SIZE)
            print('engine_combiner.get_binding_name', input_name)
            print('context_combiner.get_binding_shape', context_combiner.get_binding_shape(binding))
        for i in range(N_E):
            assert self.context_combiner_list[i].all_binding_shapes_specified

        for binding in range(3):
            input_name = engine_rnn.get_binding_name(binding)
            if input_name == 'initial_state_h':
                INITIAL_STATE_SIZE = context_rnn.get_binding_shape(binding)[0]
                print('INITIAL_STATE_SIZE:', INITIAL_STATE_SIZE)            
            print('engine_rnn.get_binding_name', input_name)
            print('context_rnn.get_binding_shape', context_rnn.get_binding_shape(binding))
        for i in range(N_E):
            assert self.context_rnn_list[i].all_binding_shapes_specified

        self.target_dtype = np.float32
        self.stream_collision_list = []
        for i in range(N_E):
            stream_collision = cuda.Stream()
            self.stream_collision_list.append(stream_collision)

        ## Allocate device memory for inputs and outputs of combiner network
        h_input_state = np.zeros((N_SIGMA, STATE_INPUT_SHAPE), dtype = np.float32)
        h_di_feature = np.zeros((N_SIGMA, self.DI_FEATURE_SIZE), dtype = np.float32) # 128
        h_inputs_combiner = [h_input_state, h_di_feature]
        
        self.d_inputs_combiner_list = []
        self.h_output_combiner_list = []
        self.d_output_combiner_list = []
        self.bindings_combiner_list = []
        for i in range(N_E):
            d_inputs_combiner = [cuda.mem_alloc(h_input.nbytes) for h_input in h_inputs_combiner]
            self.d_inputs_combiner_list.append(d_inputs_combiner)

            h_output_combiner = cuda.pagelocked_empty([N_SIGMA, 2*INITIAL_STATE_SIZE], dtype = self.target_dtype)          
            d_output_combiner = cuda.mem_alloc(h_output_combiner.nbytes)
            self.h_output_combiner_list.append(h_output_combiner)
            self.d_output_combiner_list.append(d_output_combiner)

            bindings_combiner = [int(d_input) for d_input in d_inputs_combiner] + [int(d_output_combiner)]
            self.bindings_combiner_list.append(bindings_combiner)

        # self.stream_combiner = cuda.Stream()

        ## Allocate device memory for inputs and outputs of RNN network
        h_input_state_h = np.zeros((N_SIGMA * NUM_SEQUENCE_TO_EVALUATE, INITIAL_STATE_SIZE), dtype = np.float32)
        h_input_state_c = np.zeros((N_SIGMA * NUM_SEQUENCE_TO_EVALUATE, INITIAL_STATE_SIZE), dtype = np.float32)
        action_seq_expand = np.zeros((N_SIGMA * NUM_SEQUENCE_TO_EVALUATE, ACTION_HORIZON, ACTION_SHAPE_EVALUATE), dtype = np.float32)
        h_inputs_rnn = [h_input_state_h, h_input_state_c, action_seq_expand]
        # h_inputs_rnn = [self.action_seq_expand]
        # print('h_inputs_rnn', h_inputs_rnn)

        self.d_inputs_rnn_list = []
        self.h_output_rnn_list = []
        self.d_output_rnn_list = []
        self.bindings_rnn_list = []
        for i in range(N_E):
            d_inputs_rnn = [cuda.mem_alloc(h_input.nbytes) for h_input in h_inputs_rnn]
            self.d_inputs_rnn_list.append(d_inputs_rnn)
          
            h_output_rnn = cuda.pagelocked_empty([N_SIGMA * NUM_SEQUENCE_TO_EVALUATE, ACTION_HORIZON, 1], dtype = self.target_dtype)
            self.h_output_rnn_list.append(h_output_rnn)
            # h_output_rnn = cuda.pagelocked_empty([N_E * N_SIGMA * NUM_SEQUENCE_TO_EVALUATE, ACTION_HORIZON, 16], dtype = self.target_dtype)
            d_output_rnn = cuda.mem_alloc(h_output_rnn.nbytes)
            self.d_output_rnn_list.append(d_output_rnn)

            bindings_rnn = [int(d_input) for d_input in d_inputs_rnn] + [int(d_output_rnn)]
            self.bindings_rnn_list.append(bindings_rnn)

        # self.stream_rnn = cuda.Stream()

        print("Warming up...")
        self.call_depth_state_combiner(h_inputs_combiner)
        self.call_recurrent_net(h_inputs_rnn)

        print("Done warming up!")
    
    def call_cnn_only(self, batch):   
        pass

    def call_depth_state_combiner(self, batch):
        for i in range(N_E):
            for idx in range(len(batch)):
                # Transfer input data to device
                if idx == 0: # robot's state (same for every NNs)
                    cuda.memcpy_htod_async(self.d_inputs_combiner_list[i][idx], batch[idx], self.stream_collision_list[i])
                else:
                    cuda.memcpy_htod_async(self.d_inputs_combiner_list[i][idx], batch[idx][i], self.stream_collision_list[i])

            # Execute model
            self.context_combiner_list[i].execute_async(
                batch_size=N_SIGMA, bindings=self.bindings_combiner_list[i], stream_handle=self.stream_collision_list[i].handle, input_consumed=None)
            # Synchronize the stream ? https://github.com/NVIDIA/TensorRT/blob/master/demo/BERT/inference.ipynb
            #stream.synchronize()
            # Transfer predictions back
            cuda.memcpy_dtoh_async(self.h_output_combiner_list[i], self.d_output_combiner_list[i], self.stream_collision_list[i])
        
        for i in range(N_E):
            # Syncronize threads
            self.stream_collision_list[i].synchronize()

        return self.h_output_combiner_list

    def call_recurrent_net(self, batch):
        for i in range(N_E):
            for idx in range(len(batch)):
                # Transfer input data to device
                if idx == 2: # action_seq (same for every NNs)
                    cuda.memcpy_htod_async(self.d_inputs_rnn_list[i][idx], batch[idx], self.stream_collision_list[i])
                else:
                    cuda.memcpy_htod_async(self.d_inputs_rnn_list[i][idx], batch[idx][i], self.stream_collision_list[i])
            # Execute model
            self.context_rnn_list[i].execute_async(batch_size=N_SIGMA * NUM_SEQUENCE_TO_EVALUATE,
                                                   bindings=self.bindings_rnn_list[i], stream_handle=self.stream_collision_list[i].handle, input_consumed=None)
            # Synchronize the stream ? https://github.com/NVIDIA/TensorRT/blob/master/demo/BERT/inference.ipynb
            # stream_rnn.synchronize()
            # Transfer predictions back
            cuda.memcpy_dtoh_async(self.h_output_rnn_list[i], self.d_output_rnn_list[i], self.stream_collision_list[i])
        
        for i in range(N_E):
            # Syncronize threads
            self.stream_collision_list[i].synchronize()
        
        return self.h_output_rnn_list

    def get_di_feature_size(self):
        return self.DI_FEATURE_SIZE

class InfoNetworkInferenceTensorRT(InfoNetworkInferenceBaseClass):
    def __init__(self):
        ## Load TensorRT CNN model
        f_cnn = open(IPN_TRT_CHECKPOINT_PATH + "/infogain_image_feature_engine_fp32.trt", "rb")
        runtime_cnn = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

        engine_cnn = runtime_cnn.deserialize_cuda_engine(f_cnn.read())
        self.context_cnn = engine_cnn.create_execution_context()

        for binding in range(1):
            input_name = engine_cnn.get_binding_name(binding)            
            print('engine_cnn.get_binding_name', input_name)
            print('context_cnn.get_binding_shape', self.context_cnn.get_binding_shape(binding))
        assert self.context_cnn.all_binding_shapes_specified

        ## Load TensorRT RNN model
        f_rnn = open(IPN_TRT_CHECKPOINT_PATH + "/infogain_predictor_engine_fp32.trt", "rb")
        runtime_rnn = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

        engine_rnn = runtime_rnn.deserialize_cuda_engine(f_rnn.read())
        self.context_rnn = engine_rnn.create_execution_context()

        for binding in range(3):
            input_name = engine_rnn.get_binding_name(binding)            
            print('engine_rnn.get_binding_name', input_name)
            print('context_rnn.get_binding_shape', self.context_rnn.get_binding_shape(binding))
        assert self.context_rnn.all_binding_shapes_specified

        ## Allocate device memory for inputs and outputs of CNN-only network
        self.target_dtype = np.float32
        h_inputs_cnn = [np.zeros((1, DI_WITH_MASK_SHAPE[0], DI_WITH_MASK_SHAPE[1], DI_WITH_MASK_SHAPE[2]), dtype = np.float32)]
        h_output_cnn = cuda.pagelocked_empty([1, 34, 60, 32], dtype = self.target_dtype)        
        self.d_inputs_cnn = [cuda.mem_alloc(h_input.nbytes) for h_input in h_inputs_cnn]    
        self.d_output_cnn = cuda.mem_alloc(h_output_cnn.nbytes)   

        self.bindings_cnn = [int(d_input) for d_input in self.d_inputs_cnn] + [int(self.d_output_cnn)]

        self.stream_infogain = cuda.Stream()

        ## Allocate device memory for inputs and outputs of RNN network
        h_state = np.zeros((NUM_SEQUENCE_TO_EVALUATE_INFOGAIN, STATE_INPUT_SHAPE), dtype = np.float32)
        h_info_gain_feature = np.zeros((NUM_SEQUENCE_TO_EVALUATE_INFOGAIN, 34, 60, 32), dtype = np.float32)
        action_seq_expand = np.zeros((NUM_SEQUENCE_TO_EVALUATE_INFOGAIN, ACTION_HORIZON, ACTION_SHAPE_EVALUATE), dtype = np.float32)
        h_inputs_rnn = [h_state, h_info_gain_feature, action_seq_expand]

        h_info_gain_feature_single_batch = np.zeros((1, 34, 60, 32), dtype = np.float32)
        self.info_gain_feature_single_batch_size = h_info_gain_feature_single_batch.nbytes
        d_info_gain_feature_single_batch = cuda.mem_alloc(self.info_gain_feature_single_batch_size)
        h_inputs_rnn_single = [h_state, d_info_gain_feature_single_batch, action_seq_expand]
        
        self.d_inputs_rnn = [cuda.mem_alloc(h_input.nbytes) for h_input in h_inputs_rnn]
          
        self.h_output_rnn = cuda.pagelocked_empty([NUM_SEQUENCE_TO_EVALUATE_INFOGAIN, ACTION_HORIZON_INFERENCE_INFOGAIN, 1], dtype = self.target_dtype)
        self.d_output_rnn = cuda.mem_alloc(self.h_output_rnn.nbytes)

        self.bindings_rnn = [int(d_input) for d_input in self.d_inputs_rnn] + [int(self.d_output_rnn)]

        # self.stream_rnn = cuda.Stream()

        print("Warming up...")
        self.call_feature_extractor(h_inputs_cnn)
        self.call_recurrent_net(h_inputs_rnn_single)

        print("Done warming up!")

    def call_feature_extractor(self, batch):
        for idx in range(len(batch)):
            # Transfer input data to device
            cuda.memcpy_htod_async(self.d_inputs_cnn[idx], batch[idx], self.stream_infogain)
        # Execute model converted from ONNX: explicit batch size
        # self.context_cnn.execute_async_v2(self.bindings_cnn, self.stream_infogain.handle, None)

        # Execute model converted directly from TF: implicit batch size
        self.context_cnn.execute_async(batch_size=1, bindings=self.bindings_cnn, stream_handle=self.stream_infogain.handle, input_consumed=None)

        # Synchronize the stream ? https://github.com/NVIDIA/TensorRT/blob/master/demo/BERT/inference.ipynb
        #stream.synchronize()
        # Transfer predictions back
        # cuda.memcpy_dtoh_async(self.h_output_cnn, self.d_output_cnn, self.stream_infogain) # keep the cnn feature map in GPU
        # Syncronize threads
        self.stream_infogain.synchronize()
        
        return self.d_output_cnn

    def call_recurrent_net(self, batch):
        # start_1 = timeit.default_timer()
        for idx in range(len(batch)):
            if idx == 1: # image feature map
                # copy from device to device
                for k in range(NUM_SEQUENCE_TO_EVALUATE_INFOGAIN):
                    cuda.memcpy_dtod_async(int(self.d_inputs_rnn[1]) + k*self.info_gain_feature_single_batch_size,
                                           batch[1], self.info_gain_feature_single_batch_size, self.stream_infogain)
            else:
                # Transfer input data to device
                cuda.memcpy_htod_async(self.d_inputs_rnn[idx], batch[idx], self.stream_infogain)
        self.stream_infogain.synchronize()
        # end_1 = timeit.default_timer()
        
        # Execute model
        # start_2 = timeit.default_timer()
        self.context_rnn.execute_async(batch_size=NUM_SEQUENCE_TO_EVALUATE_INFOGAIN, bindings=self.bindings_rnn, stream_handle=self.stream_infogain.handle, input_consumed=None)
        # Synchronize the stream ? https://github.com/NVIDIA/TensorRT/blob/master/demo/BERT/inference.ipynb
        self.stream_infogain.synchronize()
        # end_2 = timeit.default_timer()
        
        # Transfer predictions back
        # start_3 = timeit.default_timer()
        cuda.memcpy_dtoh_async(self.h_output_rnn, self.d_output_rnn, self.stream_infogain)

        # Syncronize threads
        self.stream_infogain.synchronize()
        # end_3 = timeit.default_timer()
        
        # print('TIME RNN INFO:', (end_1 - start_1)*1000, " ms, ", (end_2 - start_2)*1000, " ms, ", (end_3 - start_3)*1000, " ms")

        return self.h_output_rnn

    def get_di_feature_size(self):
        return (34, 60, 32)
