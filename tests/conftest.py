"""Shared pytest fixtures for the ORACLE project."""

import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import numpy as np


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def temp_file():
    """Create a temporary file for tests."""
    fd, temp_path = tempfile.mkstemp()
    os.close(fd)
    yield Path(temp_path)
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_config():
    """Provide a sample configuration dictionary."""
    return {
        "network_type": "oracle",
        "model_path": "/workspace/model_weights/test_model.hdf5",
        "tensorrt_path": "/workspace/model_weights/test_engine.trt",
        "use_tensorrt": False,
        "batch_size": 32,
        "sequence_length": 10,
        "input_dim": 64,
        "state_dim": 12,
        "collision_threshold": 0.5,
        "info_gain_threshold": 0.1,
        "max_velocity": 3.5,
        "environment": "gazebo_corridor"
    }


@pytest.fixture
def mock_ros_node():
    """Mock ROS node for testing."""
    with patch('rospy.init_node'), \
         patch('rospy.Publisher'), \
         patch('rospy.Subscriber'), \
         patch('rospy.Rate'), \
         patch('rospy.is_shutdown', return_value=False):
        yield


@pytest.fixture
def sample_depth_image():
    """Generate a sample depth image for testing."""
    return np.random.rand(480, 640).astype(np.float32)


@pytest.fixture
def sample_state_vector():
    """Generate a sample state vector for testing."""
    return np.random.rand(12).astype(np.float32)


@pytest.fixture
def sample_waypoints():
    """Generate sample waypoints for testing."""
    return [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ]


@pytest.fixture
def mock_tensorflow():
    """Mock TensorFlow for testing without GPU dependencies."""
    mock_tf = Mock()
    mock_model = Mock()
    mock_model.predict.return_value = np.random.rand(1, 1)
    mock_tf.keras.models.load_model.return_value = mock_model
    
    with patch.dict('sys.modules', {'tensorflow': mock_tf}):
        yield mock_tf


@pytest.fixture
def mock_tensorrt():
    """Mock TensorRT for testing without TensorRT dependencies."""
    mock_trt = Mock()
    mock_engine = Mock()
    mock_context = Mock()
    mock_context.execute_v2.return_value = True
    mock_engine.create_execution_context.return_value = mock_context
    
    with patch.dict('sys.modules', {'tensorrt': mock_trt, 'pycuda': Mock()}):
        yield mock_trt


@pytest.fixture
def mock_opencv():
    """Mock OpenCV for testing."""
    mock_cv2 = Mock()
    mock_cv2.imread.return_value = np.random.rand(480, 640, 3)
    mock_cv2.imwrite.return_value = True
    
    with patch.dict('sys.modules', {'cv2': mock_cv2}):
        yield mock_cv2


@pytest.fixture
def mock_config_loader():
    """Mock configuration loading."""
    def _load_config(config_name):
        return {
            "config_name": config_name,
            "model_path": f"/workspace/model_weights/{config_name}/saved-model.hdf5",
            "use_tensorrt": False,
            "batch_size": 1
        }
    
    with patch('config.load_config', side_effect=_load_config):
        yield


@pytest.fixture
def captured_logs(caplog):
    """Capture log output with specific log level."""
    import logging
    caplog.set_level(logging.INFO)
    return caplog


@pytest.fixture
def mock_file_system():
    """Mock file system operations."""
    with patch('os.path.exists', return_value=True), \
         patch('os.path.isfile', return_value=True), \
         patch('os.path.isdir', return_value=True):
        yield


@pytest.fixture
def environment_variables():
    """Set up environment variables for testing."""
    old_env = os.environ.copy()
    test_env = {
        'CUDA_VISIBLE_DEVICES': '0',
        'TF_CPP_MIN_LOG_LEVEL': '2',
        'PYTHONPATH': '/workspace'
    }
    os.environ.update(test_env)
    yield test_env
    os.environ.clear()
    os.environ.update(old_env)


@pytest.fixture(autouse=True)
def clean_imports():
    """Clean up imported modules after each test to prevent state leakage."""
    import sys
    modules_before = set(sys.modules.keys())
    yield
    modules_after = set(sys.modules.keys())
    for module in modules_after - modules_before:
        if module.startswith(('inference', 'train', 'evaluate', 'optimize', 'process')):
            sys.modules.pop(module, None)