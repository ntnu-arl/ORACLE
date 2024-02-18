# ORACLE: Library of Deep Learning-based Safe Navigation Methods

Please check out our [wiki](https://github.com/ntnu-arl/ORACLE/wiki) for more details about this work. We describe briefly below the workflow to derive learning-based navigation policies for our drone model.

The VAE code from the paper [Semantically-enhanced Deep Collision Prediction for Autonomous Navigation using Aerial Robots](https://arxiv.org/abs/2307.11522) can be found in [this repo](https://github.com/ntnu-arl/sevae).

## 1) Setup simulation environment

Follow the instructions here: [LMF_sim](https://github.com/ntnu-arl/lmf_sim) to set up the simulation workspace. You also need to install the NVIDIA GPU driver, `CUDA toolkit`, and `cudnn` to run Tensorflow on NVIDIA GPU. A typical procedure to install them can be found in [Link](https://medium.com/@pydoni/how-to-install-cuda-11-4-cudnn-8-2-opencv-4-5-on-ubuntu-20-04-65c4aa415a7b), note that the exact versions may change depending on your system.

Additionally, create a conda environment:
```
# Follow the below procedure if you have CUDA 11
conda create -n oracle_env python=3.8 libffi=3.3
conda activate oracle_env
cd lmf_sim_ws/src/planning/ORACLE/
pip install -r requirements_cuda11.txt 

# OR follow the below procedure if you have CUDA 10.1 - cudnn 7.6
conda create -n oracle_env python=3.7 libffi=3.3
conda activate oracle_env
cd lmf_sim_ws/src/planning/ORACLE/
pip install -r requirements_cuda10_1.txt 
```

If you have CUDA 10.1 - cudnn 7.6, follow the instructions [Here](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-601/tensorrt-install-guide/index.html#installing-tar) to install the TensorRT 6.0.1 python3 wheel file in oracle_env

If you would like to try out seVAE-ORACLE, install additional Python packages in `oracle_env` from [seVAE repo](https://github.com/ntnu-arl/sevae)
```
conda activate oracle_env
cd lmf_sim_ws/src/planning/sevae
pip3 install -e .
```

If you are using ROS version < Noetic, then you need to build geometry, geometry2, and vision_opencv packages with python 3 (for `import tf, cv_bridge`) following the below instructions. 

### Build geometry, geometry2 and vision_opencv with python 3 (NO need for ROS Noetic)
First, we need to get the path to our conda env:
```
conda activate oracle_env
which python
```
You should get something like this `PATH_TO_YOUR_ORACLE_ENV/bin/python`. 

Then run the following commands (replace `PATH_TO_YOUR_ORACLE_ENV` with what you get in your terminal) to create and build a workspace containing geometry, geometry2, and vision_opencv packages:
```
mkdir ros_stuff_ws && cd ros_stuff_ws
mkdir src && cd src
git clone https://github.com/ros/geometry.git -b 1.12.0
git clone https://github.com/ros/geometry2.git -b 0.6.5
git clone https://github.com/ros-perception/vision_opencv.git -b YOUR_ROS_VERSION
catkin config -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=PATH_TO_YOUR_ORACLE_ENV/bin/python -DPYTHON_INCLUDE_DIR=PATH_TO_YOUR_ORACLE_ENV/include/python3.8 -DPYTHON_LIBRARY=PATH_TO_YOUR_ORACLE_ENV/lib/libpython3.8.so
catkin config --install
catkin build geometry geometry2 vision_opencv -DSETUPTOOLS_DEB_LAYOUT=OFF
```

Don't forget to source this folder in your terminal
```
source ros_stuff_ws/devel/setup.bash
# or source ros_stuff_ws/install/setup.bash --extend
```

## 2) Generate training data: 

Set `EVALUATE_MODE = False` and `RUN_IN_SIM = True` in `config.py` file.

Run in one terminal (NOT in `conda` virtual environment)
```
# for ORACLE or A-ORACLE
roslaunch rmf_sim rmf_sim.launch
# OR for seVAE-ORACLE
roslaunch rmf_sim rmf_sim_sevae.launch
```

Open another terminal, source `lmf_sim_ws` workspace and run inside `deep_collision_predictor` folder (**Note**: remember to set `PLANNING_TYPE=1` in `config.py` for seVAE-ORACLE!)
```
# conda activate oracle_env
python generate/generate_data_info_gain.py --save_path=path_to_folder
``` 

If `--save_path` is not specified, the default path in `common_flags.py` is used.

## 3) Process the training data:

### ORACLE and A-ORACLE
Set `TRAIN_INFOGAIN = False` (for generating ORACLE data) or `True` (for labeling A-ORACLE data with Voxblox) in `config.py` file.

If labeling data for A-ORACLE, we need to run in one terminal (NO need to run this for ORACLE)
```
roslaunch voxblox_ros voxblox_gazebo.launch
```

In another terminal, run
```
# conda activate oracle_env
python process/data_processing.py --load_path=path_to_folder --save_tf_path=path_to_folder
```

### seVAE-ORACLE
Run the script in seVAE [repo](https://github.com/ntnu-arl/sevae) to create the `di_latent.p` and `di_flipped_latent.p` pickle files. Put the latent pickles in the same folder as the other pickle files in step 2 above.

Then run
```
# conda activate oracle_env
python process/data_processing_sevae.py --load_path=path_to_folder --save_tf_path=path_to_folder
```

If `--load_path` or `--save_tf_path` is not specified, the default path in `common_flags.py` is used. \
The tfrecord files created from `data_processing.py` are saved in `save_tf_path`. \
Split the tfrecord files into 2 folders for training and validation (80/20 ratio).

## 4) Train the network:

Train ORACLE (collision prediction):
```
# conda activate oracle_env
python train/training.py --training_type=0 --train_tf_folder=path_to_folder --validate_tf_folder=path_to_folder --model_save_path=path_to_folder
```

Train seVAE-ORACLE (collision prediction):
```
# conda activate oracle_env
python train/training.py --training_type=1 --train_tf_folder=path_to_folder --validate_tf_folder=path_to_folder --model_save_path=path_to_folder
```

or train Attentive ORACLE (info-gain prediction):
```
# conda activate oracle_env
python train/training.py --training_type=2 --train_tf_folder=path_to_folder --validate_tf_folder=path_to_folder --model_save_path=path_to_folder
```

If `--train_tf_folder` or `--validate_tf_folder` or `--model_save_path` is not specified, the default path in `common_flags.py` is used.

## 5) Optimize the network for inference speed (with TensorRT, optional)

**Note**: For multi-GPU systems, you may need to `export CUDA_VISIBLE_DEVICES=0` to run TensorRT, otherwise you can get some runtime errors.

Set the path to the .hdf5 file using `--checkpoint_path` when calling python scripts in `optimize` folder. The resulting .trt or .onnx files will be created in the main folder of this package.

### ORACLE

```
# conda activate oracle_env
python3 optimize/convert_keras_cnn_to_tensorrt_engine.py --checkpoint_path=PATH_TO_HDF5_FILE
python3 optimize/convert_keras_combiner_tensorrt_engine.py --checkpoint_path=PATH_TO_HDF5_FILE
python3 optimize/convert_keras_rnn_to_tensorrt_engine.py --checkpoint_path=PATH_TO_HDF5_FILE
```

### seVAE-ORACLE

```
# conda activate oracle_env
python3 optimize/convert_keras_combiner_tensorrt_engine_sevae.py --checkpoint_path=PATH_TO_HDF5_FILE
python3 optimize/convert_keras_rnn_to_tensorrt_engine_sevae.py --checkpoint_path=PATH_TO_HDF5_FILE
```

### Attentive ORACLE

```
# conda activate oracle_env
python3 optimize/convert_keras_infogain_cnn_to_tensorrt_engine.py --checkpoint_path=PATH_TO_HDF5_FILE
python3 optimize/convert_keras_infogain_predictor_to_tensorrt_engine.py --checkpoint_path=PATH_TO_HDF5_FILE
```
or for predicting the information gain of only one step in every ... step in the future (use `SKIP_STEP_INFERENCE_INFOGAIN` param in `config.py`):
```
# conda activate oracle_env
python3 optimize/convert_keras_infogain_predictor_to_tensorrt_engine_light_inference.py --checkpoint_path=PATH_TO_HDF5_FILE
```
This can lead to even faster inference speed but will hurt the performance (`SKIP_STEP_INFERENCE_INFOGAIN = 2 or 4` is recommended).

## 6) Evaluate the planner

Choose `PLANNING_TYPE` in `config.py` file (for evaluating A-ORACLE in sim, enable the RGB camera xacro in `rmf_sim/rmf_sim/rotors/urdf/delta.gazebo`)

If using Tensorflow model for inference, set `COLLISION_USE_TENSORRT = False` or `INFOGAIN_USE_TENSORRT = False` in `config.py` file and update the path to the weight files (.hdf5 files) in `config.py`.

If using TensorRT model for inference, set `COLLISION_USE_TENSORRT = True` or `INFOGAIN_USE_TENSORRT = True` in `config.py` file and update the path to the weight folders (containing .trt files) in `config.py`. **Note**: for multi-GPU systems, you may need to `export CUDA_VISIBLE_DEVICES=0` to run TensorRT, otherwise you can get some runtime errors.

Change the `world_file` argument in `rmf_sim.launch` to choose the testing environment. We provide some testing environments in `rmf_sim/worlds` folder. Additionally, set `rviz_en` to `true` in `rmf_sim.launch` for visualization of the network's prediction. Please refer to the [wiki](https://github.com/ntnu-arl/ORACLE/wiki) for detailed instructions to run the demo simulations as well as documentation of parameters in `config.py`.

### In SIM

Set `EVALUATE_MODE = True` and `RUN_IN_SIM = True` in `config.py` file.

Run in one terminal (NOT in `conda` virtual environment)
```
roslaunch rmf_sim rmf_sim.launch
```

In another terminal, run
```
# conda activate oracle_env
source PATH_TO_lmf_sim_ws/devel/setup.bash
source PATH_TO_ros_stuff_ws/devel/setup.bash # only if your ROS version < Noetic
python evaluate/evaluate.py
```

Wait until you see the green text `START planner` printed out in the second terminal, then call the service to start the planner
```
rosservice call /start_planner "{}"
```

### In the real robot (TO BE UPDATED!)

Follow the instructions here: [LMF_ws](https://github.com/ntnu-arl/lmf_ws) to set up the software in the real robot.

Set `RUN_IN_SIM = False` in `config.py` file. Run
```
# conda activate oracle_env
source PATH_TO_lmf_ws/devel/setup.bash
python evaluate/evaluate.py
```

Wait until you see the green text `START planner` printed out in your terminal, then call the service to start the planner
```
rosservice call /start_planner "{}"
```

## References

If you use this work in your research, please cite the following publications:

**Motion Primitives-based Navigation Planning using Deep Collision Prediction**

```
@INPROCEEDINGS{Nguyen2022ORACLE,
  author={Nguyen, Huan and Fyhn, Sondre Holm and De Petris, Paolo and Alexis, Kostas},
  booktitle={2022 International Conference on Robotics and Automation (ICRA)}, 
  title={Motion Primitives-based Navigation Planning using Deep Collision Prediction}, 
  year={2022},
  volume={},
  number={},
  pages={9660-9667},
  doi={10.1109/ICRA46639.2022.9812231}}
```

**Semantically-enhanced Deep Collision Prediction for Autonomous Navigation using Aerial Robots**

```
@INPROCEEDINGS{kulkarni2023semanticallyenhanced,
  author={Kulkarni, Mihir and Nguyen, Huan and Alexis, Kostas},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Semantically-Enhanced Deep Collision Prediction for Autonomous Navigation Using Aerial Robots}, 
  year={2023},
  volume={},
  number={},
  pages={3056-3063},
  doi={10.1109/IROS55552.2023.10342297}}
```

**Uncertainty-aware visually-attentive navigation using deep neural networks**

```
@article{Nguyen2023AORACLE,
  author = {Huan Nguyen and Rasmus Andersen and Evangelos Boukas and Kostas Alexis},
  title ={Uncertainty-aware visually-attentive navigation using deep neural networks},
  journal = {The International Journal of Robotics Research},
  doi = {10.1177/02783649231218720},
  URL = {https://doi.org/10.1177/02783649231218720}
```

## Ackowledgements

We would like to acknowledge the inspiration from [rpg_public_dronet](https://github.com/uzh-rpg/rpg_public_dronet) and [badgr](https://github.com/gkahn13/badgr) for the neural network architecture of our CPN. Additionally, the code for `flightmare_wrapper.py` is strongly inspired by the one of [agile_autonomy](https://github.com/uzh-rpg/agile_autonomy).

## Contact

You can contact us for any question:
* [Huan Nguyen](mailto:ndhuan93@gmail.com)
* [Mihir Kulkarni](mailto:mihir.kulkarni@ntnu.no)
* [Kostas Alexis](mailto:konstantinos.alexis@ntnu.no)
