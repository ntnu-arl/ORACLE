import abc

class NetworkInferenceBaseClass(abc.ABC):
    @abc.abstractmethod
    def call_cnn_only(self, inputs):
        pass

    @abc.abstractmethod
    def call_depth_state_combiner(self, inputs):
        pass

    @abc.abstractmethod
    def call_recurrent_net(self, inputs):
        pass

    @abc.abstractmethod
    def get_di_feature_size(self):
        pass

class InfoNetworkInferenceBaseClass(abc.ABC):
    @abc.abstractmethod
    def call_feature_extractor(self, inputs):
        pass    
    
    @abc.abstractmethod
    def call_recurrent_net(self, inputs):
        pass            

    @abc.abstractmethod
    def get_di_feature_size(self):
        pass