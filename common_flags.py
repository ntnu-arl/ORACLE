import gflags

FLAGS = gflags.FLAGS

# generate data
gflags.DEFINE_integer('NUM_EPISODES', 50000, 'Number of episodes to collect training data')
gflags.DEFINE_integer('INVALID_COUNT_LIMIT', 25, 'number of consecutive invalid trajectories to reset the env')
gflags.DEFINE_integer('PICKLE_SIZE', 100, 'number of episodes recorded in one pickle file')
gflags.DEFINE_integer('RANDOMIZE_WORLD', 20, 'randomize the env after ... episodes')
gflags.DEFINE_string('save_path', './saves', 'folder to save recorded data')

# data_processing
gflags.DEFINE_string('load_path', './saves', 'folder to load recorded data')
gflags.DEFINE_string('save_tf_path', './tfrecords_data', 'folder to save created TFrecord files')

# training
gflags.DEFINE_integer('training_type', 0, '0: end2end ORACLE, 1: seVAE-ORACLE, 2: A-ORACLE')
gflags.DEFINE_string('train_tf_folder', './saves/tfrecords_data/train', 'folder containing training data')
gflags.DEFINE_string('validate_tf_folder', './saves/tfrecords_data/validate', 'folder containing validate data')
gflags.DEFINE_string('model_save_path', './models', 'folder to save model weights')
gflags.DEFINE_string('metrics_log_dir', './logs/scalars', 'folder to log Tensorboard data')

# optimize
gflags.DEFINE_string('checkpoint_path', './model_weights/vel_3_5/net1/saved-model.hdf5', 'path to Tensorflow checkpoint to be converted to TensorRT file')
