import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import sys
sys.path.append('.')
import tensorflow as tf
import utilities 
from utilities import bcolors

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

from tensorflow import keras
from tensorflow.keras import Model, Input 
from tensorflow.keras.layers import  MaxPooling2D, Conv2D, Flatten, Dense, Activation, Dropout, BatchNormalization 
from tensorflow.keras.layers import add
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import numpy as np
import gflags
from common_flags import FLAGS
from config import *
import datetime

# Inspired by Dronet model in https://github.com/uzh-rpg/rpg_public_dronet
class NetworkBuilder():
    @staticmethod
    def depth_image_cnn_resnet8(depth_image_shape, dropout_keep_rate=1.0):
        img_height = depth_image_shape[0]
        img_width = depth_image_shape[1]
        img_channels = depth_image_shape[2]
        p = 1 - dropout_keep_rate # TF: The Dropout layer randomly sets input units to 0 with a frequency of p
        """
        Define model architecture.
        
        # Arguments
        img_width: Target image widht.
        img_height: Target image height.
        img_channels: Target image channels.
        
        # Returns
        model: A Model instance.
        """

        # Input
        img_input = Input(shape=(img_height, img_width, img_channels))

        x1 = Conv2D(32, (5, 5), strides=[2,2], padding='same', name='conv2d_0')(img_input)
        x1 = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(x1)

        # First residual block
        x2 = BatchNormalization(name='batch_normalization_0')(x1)
        x2 = Activation('relu')(x2)
        # x2 = Dropout(p)(x2)
        x2 = Conv2D(32, (3, 3), strides=[2,2], padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4), name='conv2d_1')(x2)

        x2 = BatchNormalization(name='batch_normalization_1')(x2)
        x2 = Activation('relu')(x2)
        # x2 = Dropout(p)(x2)
        x2 = Conv2D(32, (3, 3), padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4), name='conv2d_2')(x2)

        x1 = Conv2D(32, (1, 1), strides=[2,2], padding='same', name='conv2d_3')(x1)
        x3 = add([x1, x2])

        # Second residual block
        x4 = BatchNormalization(name='batch_normalization_2')(x3)
        x4 = Activation('relu')(x4)
        # x4 = Dropout(p)(x4)
        x4 = Conv2D(64, (3, 3), strides=[2,2], padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4), name='conv2d_4')(x4)

        x4 = BatchNormalization(name='batch_normalization_3')(x4)
        x4 = Activation('relu')(x4)
        # x4 = Dropout(p)(x4)
        x4 = Conv2D(64, (3, 3), padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4), name='conv2d_5')(x4)

        x3 = Conv2D(64, (1, 1), strides=[2,2], padding='same', name='conv2d_6')(x3)
        x5 = add([x3, x4])

        # Third residual block
        x6 = BatchNormalization(name='batch_normalization_4')(x5)
        x6 = Activation('relu')(x6)
        # x6 = Dropout(p)(x6)
        x6 = Conv2D(128, (3, 3), strides=[2,2], padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4), name='conv2d_7')(x6)

        x6 = BatchNormalization(name='batch_normalization_5')(x6)
        x6 = Activation('relu')(x6)
        # x6 = Dropout(p)(x6)
        x6 = Conv2D(128, (3, 3), padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4), name='conv2d_8')(x6)

        x5 = Conv2D(128, (1, 1), strides=[2,2], padding='same', name='conv2d_9')(x5)
        x7 = add([x5, x6])

        x7 = MaxPooling2D((2, 2), padding='same')(x7)

        x = Flatten()(x7)
        x = Activation('relu')(x)
        
        x = Dropout(p)(x)

        model = Model(inputs=[img_input], outputs=[x])
        # print(model.summary())

        return model

    @staticmethod
    def action_network(action_shape):
        inputs = Input(shape=action_shape)
        x = Dense(16, activation='relu', name='action/dense0')(inputs)
        x = Dense(16, activation=None, name='action/dense1')(x)
        action_model = Model(inputs, x, name='action_model')
        return action_model   

    @staticmethod
    def depth_image_cnn_info_small_new_v3(depth_image_shape):
        img_height = depth_image_shape[0]
        img_width = depth_image_shape[1]
        img_channels = depth_image_shape[2]
        """
        Define model architecture.
        
        # Arguments
        img_width: Target image widht.
        img_height: Target image height.
        img_channels: Target image channels.
        
        # Returns
        model: A Model instance.
        """

        # Input
        img_input = Input(shape=(img_height, img_width, img_channels))

        x1 = Conv2D(32, (5, 5), strides=[2,2], padding='same', name='conv2d_0')(img_input)
        x1 = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(x1)

        # First residual block
        x2 = BatchNormalization(name='batch_normalization_0')(x1)
        x2 = Activation('relu')(x2)
        # x2 = Dropout(p)(x2)
        x2 = Conv2D(32, (3, 3), strides=[2,2], padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4), name='conv2d_1')(x2)

        x2 = BatchNormalization(name='batch_normalization_1')(x2)
        x2 = Activation('relu')(x2)
        # x2 = Dropout(p)(x2)
        x2 = Conv2D(32, (3, 3), padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4), name='conv2d_2')(x2)

        x1 = Conv2D(32, (1, 1), strides=[2,2], padding='same', name='conv2d_3')(x1)
        x3 = add([x1, x2])

        # x8 = BatchNormalization(name='batch_normalization_6')(x3)

        # x = Flatten()(x8)
        # x = Activation('relu')(x8)    
        # x = Dropout(p)(x)
        cnn_feature_layer = tf.keras.layers.BatchNormalization(name='batch_normalization_6')(x3)
        cnn_feature_layer = tf.keras.layers.Activation('relu')(cnn_feature_layer)
        cnn_feature_layer = tf.keras.layers.Conv2D(32, (3, 3), padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4), name='output_conv2d_0')(cnn_feature_layer) # (None, 34, 60, 32)
        model = Model(inputs=[img_input], outputs=[cnn_feature_layer])
        # print(model.summary())

        return model

    @staticmethod
    def build_predictor_recurrent_net():
        # Input for initial state of LSTM (putting tf.split here results in incorrect ONNX conversion -> split CNN feature vector outside of this fcn)
        initial_state_h = tf.keras.Input(shape=(64), name="initial_state_h")
        initial_state_c = tf.keras.Input(shape=(64), name="initial_state_c")
        
        # Action model as in figure
        action_submodel = NetworkBuilder.action_network(action_shape=(ACTION_HORIZON, ACTION_SHAPE_EVALUATE)) # (vel_x, vel_z, steering angle)

        # Recurrent layer
        recurrent_layer = tf.keras.layers.LSTM(64, name='recurrent_layer', return_sequences=True)(inputs=action_submodel.outputs[0], 
                                                                                                    initial_state=[initial_state_h, initial_state_c])

        output_layer = tf.keras.layers.Dense(units=32, name='output_dense_1', activation='relu')(recurrent_layer)
        output_layer = tf.keras.layers.Dense(units=1, name='output_dense_2', activation='sigmoid')(output_layer)
        
        model = tf.keras.Model(inputs=[initial_state_h, initial_state_c, action_submodel.inputs[0]], outputs=[output_layer], name='predictor_recurrent')

        return model

class LossBuilder():
    @staticmethod
    def create_mask(collision_label):
        # batch_size = tf.shape(collision_label)[0]
        # batch_size_float = tf.cast(batch_size, tf.float32)
        
        # the current collision label is wrong -> HACK
        # done = tf.concat([collision_label[:, 1:], tf.expand_dims(collision_label[:, -1], axis=1)], axis=1)
        # this is correct but hacking like above for now
        done = collision_label
        
        mask = tf.cast(1.0 - done, tf.float32)
        # mask = batch_size_float * mask / tf.reduce_sum(mask) # (batch, ACTION_HORIZON)
        mask = mask / tf.reduce_sum(mask) # (batch, ACTION_HORIZON)
        return mask

    @staticmethod
    def mse_loss_with_mask(y_true, y_pred, mask):
        model_loss = tf.reduce_sum(mask * 0.5 * tf.square(y_true - y_pred))
        return model_loss

    @staticmethod
    def mae_loss_with_mask(y_true, y_pred, mask):
        model_loss = tf.reduce_sum(mask * tf.math.abs(y_true - y_pred))
        return model_loss

    @staticmethod
    # custom binary cross entropy loss function with different positive (1) weight and negative (0) weight
    def binary_cross_entropy_loss_with_class_weight(positive_weight, negative_weight):
        def weighted_loss(y_true, y_pred):
            y_true_flatten = tf.reshape(y_true, shape=(tf.size(y_true), 1))
            y_pred_flatten = tf.reshape(y_pred, shape=(tf.size(y_pred), 1))
            sample_weight = tf.fill([tf.size(y_pred), ], positive_weight)
            sample_weight = tf.where(tf.equal(tf.squeeze(y_true_flatten), 0.0), negative_weight, sample_weight)
            model_loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(y_true_flatten, y_pred_flatten) * sample_weight)

            sample_size = tf.shape(y_true)[0] * ACTION_HORIZON
            sample_size_float = tf.cast(sample_size, tf.float32)
            model_loss = model_loss / sample_size_float

            return model_loss
        return weighted_loss

class LossAndAccuracyBatch(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        tf.summary.scalar('Batch: Train loss', data=logs["loss"], step=batch)
        tf.summary.scalar('Batch: Train accuracy', data=logs["binary_accuracy"], step=batch)
    def on_test_batch_end(self, batch, logs=None):
        tf.summary.scalar('Batch: Validation loss', data=logs["loss"], step=batch)
        tf.summary.scalar('Batch: Validation accuracy', data=logs["binary_accuracy"], step=batch)

class LossAndAccuracyEpoch(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        tf.summary.scalar('Epoch: Train loss', data=logs["loss"], step=epoch)
        tf.summary.scalar('Epoch: Train accuracy', data=logs["binary_accuracy"], step=epoch)
        train_recall = self.model.train_true_positives / (self.model.train_possible_positives + K.epsilon())
        train_precision = self.model.train_true_positives / (self.model.train_predicted_positives + K.epsilon())
        tf.summary.scalar('Epoch: Train recall', train_recall.numpy(), step=epoch)
        tf.summary.scalar('Epoch: Train precision', train_precision.numpy(), step=epoch)                
        tf.summary.scalar('Epoch: Validation loss', data=logs["val_loss"], step=epoch)
        tf.summary.scalar('Epoch: Validation accuracy', data=logs["val_binary_accuracy"], step=epoch)
        val_recall = self.model.val_true_positives / (self.model.val_possible_positives + K.epsilon())
        val_precision = self.model.val_true_positives / (self.model.val_predicted_positives + K.epsilon())        
        tf.summary.scalar('Epoch: Validation recall', val_recall.numpy(), step=epoch)
        tf.summary.scalar('Epoch: Validation precision', val_precision.numpy(), step=epoch)         

        print('Epoch:', epoch, ', train loss:', logs["loss"], ', train acc:', logs["binary_accuracy"],
            ', train recall:', train_recall.numpy(), ', train_precision:', train_precision.numpy(), 
            ', train true_pos:', self.model.train_true_positives.numpy(), ', train total_sample:', self.model.train_total_samples.numpy(),
            ', train predicted_pos:', self.model.train_predicted_positives.numpy(), ', train possible_pos:', self.model.train_possible_positives.numpy(),
            ', val loss:', logs["val_loss"], ', val acc:', logs["val_binary_accuracy"],
            ', val recall:', val_recall.numpy(), ', val_precision:', val_precision.numpy(), 
            ', val true_pos:', self.model.val_true_positives.numpy(), ', val total_sample:', self.model.val_total_samples.numpy(),
            ', val predicted_pos:', self.model.val_predicted_positives.numpy(), ', val possible_pos:', self.model.val_possible_positives.numpy())
        self.model.reset_custom_metrics()

class TrainCPN(Model):
    def __init__(self, depth_image_shape, output_bias=None, alpha1=1.0, alpha2=0.01, alpha3=0.01):
        super(TrainCPN, self).__init__()
        self.predictor_model = self.build_collision_prediction_network(depth_image_shape, output_bias)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.train_true_positives = self.add_weight(name='train_true_positives', initializer='zeros')
        self.train_total_samples = self.add_weight(name='train_total_samples', initializer='zeros')
        self.train_possible_positives = self.add_weight(name='train_possible_positives', initializer='zeros')
        self.train_predicted_positives = self.add_weight(name='train_predicted_positives', initializer='zeros')
        self.val_true_positives = self.add_weight(name='val_true_positives', initializer='zeros')
        self.val_total_samples = self.add_weight(name='val_total_samples', initializer='zeros')
        self.val_possible_positives = self.add_weight(name='val_possible_positives', initializer='zeros')
        self.val_predicted_positives = self.add_weight(name='val_predicted_positives', initializer='zeros')

    def build_collision_prediction_network(self, depth_image_shape, output_bias=None):
        # input layer
        robot_state_input = tf.keras.Input(shape=STATE_INPUT_SHAPE, name="robot_state_input")
        robot_state_processed = tf.keras.layers.Dense(32, activation='relu', name='robot_state/dense0')(robot_state_input)
        robot_state_processed = tf.keras.layers.Dense(32, activation=None, name='robot_state/dense1')(robot_state_processed)

        # CNN layers for depth image
        depth_image_submodel = NetworkBuilder.depth_image_cnn_resnet8(depth_image_shape=depth_image_shape, dropout_keep_rate=DROPOUT_KEEP_RATE)
        
        # Concatenate states and depth image
        conc1 = tf.keras.layers.concatenate([robot_state_processed, depth_image_submodel.outputs[0]], name="concatenate_depth_states")    
        
        obs_lowd = tf.keras.layers.Dense(128, activation='relu', name='obs_lowd/dense0')(conc1)
        obs_lowd = tf.keras.layers.Dense(128, activation=None, name='obs_lowd/dense1')(obs_lowd)

        initial_state_h, initial_state_c = tf.split(obs_lowd, 2, axis=1) 
        
        # Action model as in figure
        action_submodel = NetworkBuilder.action_network(action_shape=(ACTION_HORIZON, ACTION_SHAPE_EVALUATE))

        # Recurrent layer
        recurrent_layer = tf.keras.layers.LSTM(64, name='recurrent_layer', return_sequences=True)(inputs=action_submodel.outputs[0], initial_state=[initial_state_h, initial_state_c])

        output_layer = tf.keras.layers.Dense(units=32, name='output_dense_1', activation='relu')(recurrent_layer)
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)    
        output_layer = tf.keras.layers.Dense(units=1, name='output_dense_2', activation='sigmoid', bias_initializer=output_bias)(output_layer)
        
        # pos output layer
        pos_yaw_output_layer = tf.keras.layers.Dense(units=32, name='pos_output_dense_1', activation='relu')(recurrent_layer)
        pos_yaw_output_layer = tf.keras.layers.Dense(units=4, name='pos_output_dense_2', activation=None)(pos_yaw_output_layer)

        model = tf.keras.Model(inputs=[robot_state_input, depth_image_submodel.inputs[0], action_submodel.inputs[0]], outputs=[
                            output_layer, pos_yaw_output_layer[..., 0:3], pos_yaw_output_layer[..., 3]], name='actor_net')
        return model

    def compile(self, optimizer, loss=None, metrics=None):
        super(TrainCPN, self).compile(optimizer, loss, metrics)
        self.loss_tracker = keras.metrics.Mean(name="loss") # already in self.metrics
        self.outout1_binary_accuracy_tracker = keras.metrics.BinaryAccuracy(name="binary_accuracy", threshold=0.5)
        self.output2_mae_tracker = keras.metrics.Mean(name="output_2_mae")
        self.output2_mse_tracker = keras.metrics.Mean(name="output_2_mse")
        self.output3_mae_tracker = keras.metrics.Mean(name="output_3_mae")
        self.output3_mse_tracker = keras.metrics.Mean(name="output_3_mse")

    def reset_custom_metrics(self):
        self.train_true_positives.assign(0.0)
        self.train_total_samples.assign(0.0)
        self.train_possible_positives.assign(0.0)
        self.train_predicted_positives.assign(0.0)
        self.val_true_positives.assign(0.0)
        self.val_total_samples.assign(0.0)
        self.val_possible_positives.assign(0.0)
        self.val_predicted_positives.assign(0.0)        

    @tf.function
    def train_step(self, data):
        image, actions, robot_state, collision_label, info_label, pos_label, height, width, depth, action_horizon = data
        # image, actions, robot_state, collision_label, height, width, depth, action_horizon = data
        with tf.GradientTape() as tape:
            velocity_x = robot_state[:, 3]
            velocity_x = tf.expand_dims(velocity_x, axis=1)
            velocity_y = robot_state[:, 4]
            velocity_y = tf.expand_dims(velocity_y, axis=1)
            velocity_z = robot_state[:, 5]
            velocity_z = tf.expand_dims(velocity_z, axis=1)
            yaw_rate = robot_state[:, 15]
            yaw_rate = tf.expand_dims(yaw_rate, axis=1)
            roll_angle = robot_state[:, 16]
            roll_angle = tf.expand_dims(roll_angle, axis=1)
            pitch_angle = robot_state[:, 17]
            pitch_angle = tf.expand_dims(pitch_angle, axis=1)
            robot_state = tf.concat(
                [velocity_x, velocity_y, velocity_z, yaw_rate, roll_angle, pitch_angle], 1)
            # velocity_x = robot_state[:,3]
            # velocity_x = tf.expand_dims(velocity_x, axis=1)
            # velocity_z = robot_state[:,5]
            # velocity_z = tf.expand_dims(velocity_z, axis=1)
            # yaw_rate = robot_state[:,15]
            # yaw_rate = tf.expand_dims(yaw_rate, axis=1)
            # robot_state = tf.concat([velocity_x, velocity_z, yaw_rate], 1)            
            [y_pred, pos_pred, yaw_pred] = self.predictor_model([robot_state, tf.expand_dims(image[:,:,:,0], axis=-1), actions[:,:,0]], training=True)  # Forward pass
            # y_pred = self.predictor_model([robot_state, image, actions[:,:,0]], training=True)
            # (the loss function is configured in `compile()`)
            y_pred = tf.squeeze(y_pred)

            # create the mask
            mask = LossBuilder.create_mask(collision_label) # (batch, ACTION_HORIZON)
            mask_expand = tf.expand_dims(mask, axis=-1) # (batch, ACTION_HORIZON, 1)

            output1_loss = LossBuilder.binary_cross_entropy_loss_with_class_weight(positive_weight=1.0, negative_weight=1.0)(collision_label, y_pred)
            output2_mae = LossBuilder.mae_loss_with_mask(pos_label[:, :, 0:3], pos_pred, mask_expand)
            output2_mse = LossBuilder.mse_loss_with_mask(pos_label[:, :, 0:3], pos_pred, mask_expand)
            output3_mae = LossBuilder.mae_loss_with_mask(pos_label[:, :, 3], yaw_pred, mask)
            output3_mse = LossBuilder.mse_loss_with_mask(pos_label[:, :, 3], yaw_pred, mask)
            loss_raw = self.alpha1 * output1_loss + self.alpha2 * output2_mse + self.alpha3 * output3_mse
            loss = loss_raw + sum(self.losses)  # add regularization term

        self.loss_tracker.update_state(loss_raw)
        self.outout1_binary_accuracy_tracker.update_state(collision_label, y_pred)
        self.output2_mae_tracker.update_state(output2_mae)
        self.output2_mse_tracker.update_state(output2_mse)
        self.output3_mae_tracker.update_state(output3_mae)
        self.output3_mse_tracker.update_state(output3_mse)

        # Compute gradients
        gradients = tape.gradient(loss, self.predictor_model.trainable_weights)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.predictor_model.trainable_weights))

        label_bool = (collision_label > 0.5) # threshold
        self.train_possible_positives.assign_add(tf.reduce_sum(tf.cast(label_bool, dtype=tf.float32)))
        y_pred_bool = (y_pred > 0.5)
        self.train_predicted_positives.assign_add(tf.reduce_sum(tf.cast(y_pred_bool, dtype=tf.float32)))
        values = tf.logical_and(label_bool, y_pred_bool)
        values = tf.cast(values, dtype=tf.float32)
        self.train_true_positives.assign_add(tf.reduce_sum(values))                               
        self.train_total_samples.assign_add(tf.cast(tf.size(collision_label), dtype=tf.float32))            
               
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        # Unpack the data
        image, actions, robot_state, collision_label, info_label, pos_label, height, width, depth, action_horizon = data
        # image, actions, robot_state, collision_label, height, width, depth, action_horizon = data
        velocity_x = robot_state[:, 3]
        velocity_x = tf.expand_dims(velocity_x, axis=1)
        velocity_y = robot_state[:, 4]
        velocity_y = tf.expand_dims(velocity_y, axis=1)
        velocity_z = robot_state[:, 5]
        velocity_z = tf.expand_dims(velocity_z, axis=1)
        yaw_rate = robot_state[:, 15]
        yaw_rate = tf.expand_dims(yaw_rate, axis=1)
        roll_angle = robot_state[:, 16]
        roll_angle = tf.expand_dims(roll_angle, axis=1)
        pitch_angle = robot_state[:, 17]
        pitch_angle = tf.expand_dims(pitch_angle, axis=1)
        robot_state = tf.concat(
            [velocity_x, velocity_y, velocity_z, yaw_rate, roll_angle, pitch_angle], 1)
        # velocity_x = robot_state[:,3]
        # velocity_x = tf.expand_dims(velocity_x, axis=1)
        # velocity_z = robot_state[:,5]
        # velocity_z = tf.expand_dims(velocity_z, axis=1)
        # yaw_rate = robot_state[:,15]
        # yaw_rate = tf.expand_dims(yaw_rate, axis=1)
        # robot_state = tf.concat([velocity_x, velocity_z, yaw_rate], 1)
        # Compute predictions
        [y_pred, pos_pred, yaw_pred] = self.predictor_model([robot_state, tf.expand_dims(image[:,:,:,0], axis=-1), actions[:,:,0]], training=False)
        # y_pred = self.predictor_model([robot_state, image, actions[:,:,0]], training=False)  # Forward pass
        y_pred = tf.squeeze(y_pred)

        mask = LossBuilder.create_mask(collision_label) # (batch, ACTION_HORIZON)
        mask_expand = tf.expand_dims(mask, axis=-1) # (batch, ACTION_HORIZON, 1)

        output1_loss = LossBuilder.binary_cross_entropy_loss_with_class_weight(positive_weight=1.0, negative_weight=1.0)(collision_label, y_pred)
        output2_mae = LossBuilder.mae_loss_with_mask(pos_label[:, :, 0:3], pos_pred, mask_expand)
        output2_mse = LossBuilder.mse_loss_with_mask(pos_label[:, :, 0:3], pos_pred, mask_expand)
        output3_mae = LossBuilder.mae_loss_with_mask(pos_label[:, :, 3], yaw_pred, mask)
        output3_mse = LossBuilder.mse_loss_with_mask(pos_label[:, :, 3], yaw_pred, mask)
        loss_raw = self.alpha1 * output1_loss + self.alpha2 * output2_mse + self.alpha3 * output3_mse

        self.loss_tracker.update_state(loss_raw)
        self.outout1_binary_accuracy_tracker.update_state(collision_label, y_pred)
        self.output2_mae_tracker.update_state(output2_mae)
        self.output2_mse_tracker.update_state(output2_mse)
        self.output3_mae_tracker.update_state(output3_mae)
        self.output3_mse_tracker.update_state(output3_mse)

        label_bool = (collision_label > 0.5) # threshold
        self.val_possible_positives.assign_add(tf.reduce_sum(tf.cast(label_bool, dtype=tf.float32)))
        y_pred_bool = (y_pred > 0.5)
        self.val_predicted_positives.assign_add(tf.reduce_sum(tf.cast(y_pred_bool, dtype=tf.float32)))
        values = tf.logical_and(label_bool, y_pred_bool)
        values = tf.cast(values, dtype=tf.float32)
        self.val_true_positives.assign_add(tf.reduce_sum(values))                               
        self.val_total_samples.assign_add(tf.cast(tf.size(collision_label), dtype=tf.float32))   

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs):
        return self.predictor_model(inputs, training=False)

    def summary(self, line_length=None, positions=None, print_fn=None):
        return self.predictor_model.summary(line_length, positions, print_fn)

    def get_model(self):
        return self.predictor_model

class InferenceCPN():
    def __init__(self, depth_image_shape):
        self.cnn_only = NetworkBuilder.depth_image_cnn_resnet8(depth_image_shape, dropout_keep_rate=1.0)
        self.depth_state_combiner = self.build_predictor_depth_robot_state_combiner(self.cnn_only.outputs[0].shape[1])
        self.recurrent_net = NetworkBuilder.build_predictor_recurrent_net()

        self.depth_state_combiner_outputs = self.depth_state_combiner([self.depth_state_combiner.inputs[0], self.cnn_only.outputs[0]])
        self.initial_state_h, self.initial_state_c = tf.split(self.depth_state_combiner_outputs, 2, axis=1)
        
        self.model = tf.keras.Model(inputs=[self.depth_state_combiner.inputs[0], self.cnn_only.inputs[0], self.recurrent_net.inputs[2]], # robot_state, di_feature, action_seq
                                    outputs=self.recurrent_net([self.initial_state_h, self.initial_state_c, self.recurrent_net.inputs[2]]), name='actor_net')
    
    def build_predictor_depth_robot_state_combiner(self, DI_FEATURE_SHAPE):
        # input layer
        robot_state_input = tf.keras.Input(shape=STATE_INPUT_SHAPE, name="robot_state_input")
        robot_state_processed = tf.keras.layers.Dense(32, activation='relu', name='robot_state/dense0')(robot_state_input)
        robot_state_processed = tf.keras.layers.Dense(32, activation=None, name='robot_state/dense1')(robot_state_processed)

        di_feature = tf.keras.Input(shape=DI_FEATURE_SHAPE, name="di_feature")
        conc1 = tf.keras.layers.concatenate([robot_state_processed, di_feature], name="concatenate_depth_states")    

        obs_lowd = tf.keras.layers.Dense(128, activation='relu', name='obs_lowd/dense0')(conc1)
        obs_lowd = tf.keras.layers.Dense(128, activation=None, name='obs_lowd/dense1')(obs_lowd)
        model = tf.keras.Model(inputs=[robot_state_input, di_feature], outputs=[obs_lowd], name='depth_robot_state_combiner')
        return model

    def load_model(self, model):
        trainable_layers = ['robot_state/dense0', 'robot_state/dense1', 'conv2d_0', 'conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4', 'conv2d_5', 'conv2d_6', 'conv2d_7', 'conv2d_8', 'conv2d_9',
                            'batch_normalization_0', 'batch_normalization_1', 'batch_normalization_2', 'batch_normalization_3', 'batch_normalization_4', 'batch_normalization_5',
                            'action/dense0','action/dense1', 'obs_lowd/dense0', 'obs_lowd/dense1', 'recurrent_layer', 'output_dense_1', 'output_dense_2']
        # for layer in model.layers:
        #     print('layer name:', layer.name)
        
        for layer in self.cnn_only.layers:
            if layer.name in trainable_layers:
                layer.set_weights(model.get_layer(layer.name).get_weights())

        for layer in self.depth_state_combiner.layers:
            if layer.name in trainable_layers:
                layer.set_weights(model.get_layer(layer.name).get_weights())

        for layer in self.recurrent_net.layers:
            if layer.name in trainable_layers:
                layer.set_weights(model.get_layer(layer.name).get_weights())                            

    def call_cnn_only(self, inputs):
        return self.cnn_only.predict_on_batch(inputs)

    def call_depth_state_combiner(self, inputs):
        return self.depth_state_combiner.predict_on_batch(inputs)

    def call_recurrent_net(self, inputs):
        return self.recurrent_net.predict_on_batch(inputs)

    def summary(self):
        return self.model.summary()

    def get_model(self):
        return self.model

    def get_cnn_only(self):
        return self.cnn_only

    def get_depth_state_combiner(self):
        return self.depth_state_combiner

    def get_rnn(self):
        return self.recurrent_net

    def get_di_feature_size(self):
        return self.cnn_only.outputs[0].shape[1] 

    def get_initial_state_size(self):
        return self.recurrent_net.inputs[0].shape[1]

class TrainCPNseVAE(TrainCPN):
    def build_collision_prediction_network(self, depth_image_shape, output_bias=None):
        # input layer
        robot_state_input = tf.keras.Input(shape=STATE_INPUT_SHAPE, name="robot_state_input")
        robot_state_processed = tf.keras.layers.Dense(32, activation='relu', name='robot_state/dense0')(robot_state_input)
        robot_state_processed = tf.keras.layers.Dense(32, activation=None, name='robot_state/dense1')(robot_state_processed)

        # latent vector input
        depth_image_latent_input = tf.keras.Input(shape=depth_image_shape, name="depth_latent_input")
        depth_dense = tf.keras.layers.Dense(128, activation=None, name='depth_latent/dense0')(depth_image_latent_input)

        # Concatenate states and depth image
        conc1 = tf.keras.layers.concatenate([robot_state_processed, depth_dense], name="concatenate_depth_states")
        
        obs_lowd = tf.keras.layers.Dense(128, activation='relu', name='obs_lowd/dense0')(conc1)
        obs_lowd = tf.keras.layers.Dense(128, activation=None, name='obs_lowd/dense1')(obs_lowd)

        initial_state_h, initial_state_c = tf.split(obs_lowd, 2, axis=1) 
        
        # Action model as in figure
        action_submodel = NetworkBuilder.action_network(action_shape=(ACTION_HORIZON, ACTION_SHAPE_EVALUATE))

        # Recurrent layer
        recurrent_layer = tf.keras.layers.LSTM(64, name='recurrent_layer', return_sequences=True)(inputs=action_submodel.outputs[0], initial_state=[initial_state_h, initial_state_c])

        output_layer = tf.keras.layers.Dense(units=32, name='output_dense_1', activation='relu')(recurrent_layer)
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)    
        output_layer = tf.keras.layers.Dense(units=1, name='output_dense_2', activation='sigmoid', bias_initializer=output_bias)(output_layer)
        
        # pos output layer
        pos_yaw_output_layer = tf.keras.layers.Dense(units=32, name='pos_output_dense_1', activation='relu')(recurrent_layer)
        pos_yaw_output_layer = tf.keras.layers.Dense(units=4, name='pos_output_dense_2', activation=None)(pos_yaw_output_layer)

        model = tf.keras.Model(inputs=[robot_state_input, depth_image_latent_input, action_submodel.inputs[0]], outputs=[
                            output_layer, pos_yaw_output_layer[..., 0:3], pos_yaw_output_layer[..., 3]], name='actor_net')
        return model    

    @tf.function
    def train_step(self, data):
        depth_latent_input, actions, robot_state, collision_label, info_label, pos_label, height, width, depth, action_horizon = data
        # image, actions, robot_state, collision_label, height, width, depth, action_horizon = data
        with tf.GradientTape() as tape:
            velocity_x = robot_state[:, 3]
            velocity_x = tf.expand_dims(velocity_x, axis=1)
            velocity_y = robot_state[:, 4]
            velocity_y = tf.expand_dims(velocity_y, axis=1)
            velocity_z = robot_state[:, 5]
            velocity_z = tf.expand_dims(velocity_z, axis=1)
            yaw_rate = robot_state[:, 15]
            yaw_rate = tf.expand_dims(yaw_rate, axis=1)
            roll_angle = robot_state[:, 16]
            roll_angle = tf.expand_dims(roll_angle, axis=1)
            pitch_angle = robot_state[:, 17]
            pitch_angle = tf.expand_dims(pitch_angle, axis=1)
            robot_state = tf.concat(
                [velocity_x, velocity_y, velocity_z, yaw_rate, roll_angle, pitch_angle], 1)
            # velocity_x = robot_state[:,3]
            # velocity_x = tf.expand_dims(velocity_x, axis=1)
            # velocity_z = robot_state[:,5]
            # velocity_z = tf.expand_dims(velocity_z, axis=1)
            # yaw_rate = robot_state[:,15]
            # yaw_rate = tf.expand_dims(yaw_rate, axis=1)
            # robot_state = tf.concat([velocity_x, velocity_z, yaw_rate], 1)            
            [y_pred, pos_pred, yaw_pred] = self.predictor_model([robot_state, depth_latent_input, actions[:,:,0]], training=True)  # Forward pass
            # y_pred = self.predictor_model([robot_state, image, actions[:,:,0]], training=True)
            # (the loss function is configured in `compile()`)
            y_pred = tf.squeeze(y_pred)

            # create the mask
            mask = LossBuilder.create_mask(collision_label) # (batch, ACTION_HORIZON)
            mask_expand = tf.expand_dims(mask, axis=-1) # (batch, ACTION_HORIZON, 1)

            output1_loss = LossBuilder.binary_cross_entropy_loss_with_class_weight(positive_weight=1.0, negative_weight=1.0)(collision_label, y_pred)
            output2_mae = LossBuilder.mae_loss_with_mask(pos_label[:, :, 0:3], pos_pred, mask_expand)
            output2_mse = LossBuilder.mse_loss_with_mask(pos_label[:, :, 0:3], pos_pred, mask_expand)
            output3_mae = LossBuilder.mae_loss_with_mask(pos_label[:, :, 3], yaw_pred, mask)
            output3_mse = LossBuilder.mse_loss_with_mask(pos_label[:, :, 3], yaw_pred, mask)
            loss_raw = self.alpha1 * output1_loss + self.alpha2 * output2_mse + self.alpha3 * output3_mse
            loss = loss_raw + sum(self.losses)  # add regularization term

        self.loss_tracker.update_state(loss_raw)
        self.outout1_binary_accuracy_tracker.update_state(collision_label, y_pred)
        self.output2_mae_tracker.update_state(output2_mae)
        self.output2_mse_tracker.update_state(output2_mse)
        self.output3_mae_tracker.update_state(output3_mae)
        self.output3_mse_tracker.update_state(output3_mse)

        # Compute gradients
        gradients = tape.gradient(loss, self.predictor_model.trainable_weights)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.predictor_model.trainable_weights))

        label_bool = (collision_label > 0.5) # threshold
        self.train_possible_positives.assign_add(tf.reduce_sum(tf.cast(label_bool, dtype=tf.float32)))
        y_pred_bool = (y_pred > 0.5)
        self.train_predicted_positives.assign_add(tf.reduce_sum(tf.cast(y_pred_bool, dtype=tf.float32)))
        values = tf.logical_and(label_bool, y_pred_bool)
        values = tf.cast(values, dtype=tf.float32)
        self.train_true_positives.assign_add(tf.reduce_sum(values))                               
        self.train_total_samples.assign_add(tf.cast(tf.size(collision_label), dtype=tf.float32))            
               
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        # Unpack the data
        depth_latent_input, actions, robot_state, collision_label, info_label, pos_label, height, width, depth, action_horizon = data
        # image, actions, robot_state, collision_label, height, width, depth, action_horizon = data
        velocity_x = robot_state[:, 3]
        velocity_x = tf.expand_dims(velocity_x, axis=1)
        velocity_y = robot_state[:, 4]
        velocity_y = tf.expand_dims(velocity_y, axis=1)
        velocity_z = robot_state[:, 5]
        velocity_z = tf.expand_dims(velocity_z, axis=1)
        yaw_rate = robot_state[:, 15]
        yaw_rate = tf.expand_dims(yaw_rate, axis=1)
        roll_angle = robot_state[:, 16]
        roll_angle = tf.expand_dims(roll_angle, axis=1)
        pitch_angle = robot_state[:, 17]
        pitch_angle = tf.expand_dims(pitch_angle, axis=1)
        robot_state = tf.concat(
            [velocity_x, velocity_y, velocity_z, yaw_rate, roll_angle, pitch_angle], 1)
        # velocity_x = robot_state[:,3]
        # velocity_x = tf.expand_dims(velocity_x, axis=1)
        # velocity_z = robot_state[:,5]
        # velocity_z = tf.expand_dims(velocity_z, axis=1)
        # yaw_rate = robot_state[:,15]
        # yaw_rate = tf.expand_dims(yaw_rate, axis=1)
        # robot_state = tf.concat([velocity_x, velocity_z, yaw_rate], 1)
        # Compute predictions
        [y_pred, pos_pred, yaw_pred] = self.predictor_model([robot_state, depth_latent_input, actions[:,:,0]], training=False)
        # y_pred = self.predictor_model([robot_state, image, actions[:,:,0]], training=False)  # Forward pass
        y_pred = tf.squeeze(y_pred)

        mask = LossBuilder.create_mask(collision_label) # (batch, ACTION_HORIZON)
        mask_expand = tf.expand_dims(mask, axis=-1) # (batch, ACTION_HORIZON, 1)

        output1_loss = LossBuilder.binary_cross_entropy_loss_with_class_weight(positive_weight=1.0, negative_weight=1.0)(collision_label, y_pred)
        output2_mae = LossBuilder.mae_loss_with_mask(pos_label[:, :, 0:3], pos_pred, mask_expand)
        output2_mse = LossBuilder.mse_loss_with_mask(pos_label[:, :, 0:3], pos_pred, mask_expand)
        output3_mae = LossBuilder.mae_loss_with_mask(pos_label[:, :, 3], yaw_pred, mask)
        output3_mse = LossBuilder.mse_loss_with_mask(pos_label[:, :, 3], yaw_pred, mask)
        loss_raw = self.alpha1 * output1_loss + self.alpha2 * output2_mse + self.alpha3 * output3_mse

        self.loss_tracker.update_state(loss_raw)
        self.outout1_binary_accuracy_tracker.update_state(collision_label, y_pred)
        self.output2_mae_tracker.update_state(output2_mae)
        self.output2_mse_tracker.update_state(output2_mse)
        self.output3_mae_tracker.update_state(output3_mae)
        self.output3_mse_tracker.update_state(output3_mse)

        label_bool = (collision_label > 0.5) # threshold
        self.val_possible_positives.assign_add(tf.reduce_sum(tf.cast(label_bool, dtype=tf.float32)))
        y_pred_bool = (y_pred > 0.5)
        self.val_predicted_positives.assign_add(tf.reduce_sum(tf.cast(y_pred_bool, dtype=tf.float32)))
        values = tf.logical_and(label_bool, y_pred_bool)
        values = tf.cast(values, dtype=tf.float32)
        self.val_true_positives.assign_add(tf.reduce_sum(values))                               
        self.val_total_samples.assign_add(tf.cast(tf.size(collision_label), dtype=tf.float32))   

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

class InferenceCPNseVAE():
    def __init__(self, depth_latent_input):
        self.depth_state_combiner = self.build_predictor_depth_robot_state_combiner(depth_latent_input)
        self.recurrent_net = NetworkBuilder.build_predictor_recurrent_net()

        self.depth_state_combiner_outputs = self.depth_state_combiner([self.depth_state_combiner.inputs[0], self.depth_state_combiner.inputs[1]])
        self.initial_state_h, self.initial_state_c = tf.split(self.depth_state_combiner_outputs, 2, axis=1)
        
        self.model = tf.keras.Model(inputs=[self.depth_state_combiner.inputs[0], self.depth_state_combiner.inputs[1], self.recurrent_net.inputs[2]], # robot_state, di_feature, action_seq
                                    outputs=self.recurrent_net([self.initial_state_h, self.initial_state_c, self.recurrent_net.inputs[2]]), name='actor_net')
    
    def build_predictor_depth_robot_state_combiner(self, depth_latent_input):
        # input layer
        robot_state_input = tf.keras.Input(shape=STATE_INPUT_SHAPE, name="robot_state_input")
        robot_state_processed = tf.keras.layers.Dense(32, activation='relu', name='robot_state/dense0')(robot_state_input)
        robot_state_processed = tf.keras.layers.Dense(32, activation=None, name='robot_state/dense1')(robot_state_processed)

        # latent vector input
        depth_image_latent_input = tf.keras.Input(shape=depth_latent_input, name="depth_latent_input")
        depth_dense = tf.keras.layers.Dense(128, activation=None, name='depth_latent/dense0')(depth_image_latent_input)

        # Concatenate states and depth image
        conc1 = tf.keras.layers.concatenate([robot_state_processed, depth_dense], name="concatenate_depth_states")

        obs_lowd = tf.keras.layers.Dense(128, activation='relu', name='obs_lowd/dense0')(conc1)
        obs_lowd = tf.keras.layers.Dense(128, activation=None, name='obs_lowd/dense1')(obs_lowd)
        model = tf.keras.Model(inputs=[robot_state_input, depth_image_latent_input], outputs=[obs_lowd], name='depth_robot_state_combiner')
        return model

    def load_model(self, model):
        trainable_layers = ['robot_state/dense0', 'robot_state/dense1', 'depth_latent/dense0',
                            'action/dense0','action/dense1', 'obs_lowd/dense0', 'obs_lowd/dense1', 
                            'recurrent_layer', 'output_dense_1', 'output_dense_2']

        for layer in self.depth_state_combiner.layers:
            if layer.name in trainable_layers:
                layer.set_weights(model.get_layer(layer.name).get_weights())

        for layer in self.recurrent_net.layers:
            if layer.name in trainable_layers:
                layer.set_weights(model.get_layer(layer.name).get_weights())

    def call_depth_state_combiner(self, inputs):
        return self.depth_state_combiner.predict_on_batch(inputs)

    def call_recurrent_net(self, inputs):
        return self.recurrent_net.predict_on_batch(inputs)

    def summary(self):
        return self.model.summary()

    def get_model(self):
        return self.model

    def get_depth_state_combiner(self):
        return self.depth_state_combiner

    def get_rnn(self):
        return self.recurrent_net

    def get_di_feature_size(self):
        return DI_LATENT_SIZE

    def get_initial_state_size(self):
        return self.recurrent_net.inputs[0].shape[1]

class TrainIPN(Model):
    def __init__(self, depth_image_shape, alpha1=1.0, alpha2=0.01, alpha3=0.01):
        super(TrainIPN, self).__init__()
        self.predictor_model = self.build_info_gain_prediction_network(depth_image_shape)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3

    def build_info_gain_prediction_network(self, depth_image_shape):
        # state input
        robot_state_input = tf.keras.Input(
            shape=STATE_INPUT_SHAPE, name="robot_state_input")
        robot_state_processed = tf.keras.layers.Dense(
            32, activation='relu', name='robot_state/dense0')(robot_state_input)
        robot_state_processed = tf.keras.layers.Dense(
            32, activation='relu', name='robot_state/dense1')(robot_state_processed)
        robot_state_processed = tf.keras.layers.Dense(
            32, activation=None, name='robot_state/dense2')(robot_state_processed)

        initial_state_h, initial_state_c = tf.split(robot_state_processed, 2, axis=1)

        # Action model
        action_submodel = NetworkBuilder.action_network(action_shape=(
            ACTION_HORIZON, ACTION_SHAPE_EVALUATE))

        # 1D LSTM for robot's state
        recurrent_layer = tf.keras.layers.LSTM(16, name='recurrent_layer_robot_state', return_sequences=True)(
            inputs=action_submodel.outputs[0], initial_state=[initial_state_h, initial_state_c])

        state_output_layer = tf.keras.layers.Dense(
            units=32, name='state_feature/dense0', activation='relu')(recurrent_layer)
        state_output_layer = tf.keras.layers.Dense(
            units=32, name='state_feature/dense1', activation=None)(state_output_layer) # (None, ACTION_HORIZON, 32)
        
        pos_output_layer = state_output_layer[..., 0:3]
        yaw_output_layer = state_output_layer[..., 3]


        # CNN layers for depth image
        depth_image_submodel = NetworkBuilder.depth_image_cnn_info_small_new_v3(
            depth_image_shape=depth_image_shape) # (None, 34, 60, 32)
        cnn_feature_shape = depth_image_submodel.outputs[0].get_shape()
        print('cnn_feature_shape:', cnn_feature_shape)
        cnn_feature_layer = depth_image_submodel.outputs[0] # (None, 34, 60, 32)
        # cnn_feature_layer2 = tf.keras.layers.BatchNormalization(name='batch_normalization_6')(cnn_feature_layer)
        # cnn_feature_layer2 = tf.keras.layers.Activation('relu')(cnn_feature_layer2)

        # Combined with state feature
        state_fearure1 = tf.keras.layers.Dense(
                    units=32, name='state_feature/dense2', activation='relu')(state_output_layer) # (None, ACTION_HORIZON, 32)
        state_fearure1 = tf.keras.layers.Dense(
                    units=32, name='state_feature/dense3', activation=None)(state_fearure1)
        state_fearure1 = tf.expand_dims(state_fearure1, axis=-2)
        state_fearure1 = tf.expand_dims(state_fearure1, axis=-2) # (None, ACTION_HORIZON, 1, 1, 32)
        
        # cnn_feature1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',
        #             kernel_initializer="he_normal",
        #             kernel_regularizer=regularizers.l2(1e-4), name='output_conv2d_0')(cnn_feature_layer2) # (None, 34, 60, 32)
        cnn_feature1 = tf.expand_dims(cnn_feature_layer, axis=1) # (None, 1, 34, 60, 32)
        
        cnn_feature1 = tf.keras.layers.add([cnn_feature1, state_fearure1]) # (None, ACTION_HORIZON, 34, 60, 32)

        output_conv = cnn_feature1

        # residual block
        output_conv2 = tf.keras.layers.BatchNormalization(name='output_batch_normalization_0')(output_conv)
        output_conv2 = tf.keras.layers.Activation('relu')(output_conv2)

        output_conv2 = tf.keras.layers.Conv2D(32, (3, 3), strides=[2,2], padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4), name='output_conv2d_1')(output_conv2)
        output_conv2 = tf.keras.layers.BatchNormalization(name='output_batch_normalization_1')(output_conv2)
        output_conv2 = tf.keras.layers.Activation('relu')(output_conv2)

        output_conv2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4), name='output_conv2d_2')(output_conv2)
        
        output_conv3 = tf.keras.layers.Conv2D(32, (1, 1), strides=[2,2], padding='same', name='output_conv2d_3')(output_conv)
        output_conv4 = tf.keras.layers.add([output_conv2, output_conv3])
        
        # another residual block
        output_conv5 = tf.keras.layers.BatchNormalization(name='output_batch_normalization_2')(output_conv4)
        output_conv5 = tf.keras.layers.Activation('relu')(output_conv5)

        output_conv5 = tf.keras.layers.Conv2D(64, (3, 3), strides=[2,2], padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4), name='output_conv2d_4')(output_conv5)
        output_conv5 = tf.keras.layers.BatchNormalization(name='output_batch_normalization_3')(output_conv5)
        output_conv5 = tf.keras.layers.Activation('relu')(output_conv5)

        output_conv5 = tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4), name='output_conv2d_5')(output_conv5)
        
        output_conv6 = tf.keras.layers.Conv2D(64, (1, 1), strides=[2,2], padding='same', name='output_conv2d_6')(output_conv4)
        output_conv7 = tf.keras.layers.add([output_conv5, output_conv6])
        
        # no batch norm at the end
        # output_conv7 = tf.keras.layers.BatchNormalization(name='output_batch_normalization_4')(output_conv7)
        output_conv7 = tf.keras.layers.Activation('relu')(output_conv7)

        # Final prediction
        x = tf.keras.layers.Reshape((ACTION_HORIZON, -1))(output_conv7)
        info_output_layer = tf.keras.layers.Dense(
            units=1, name='info_gain_output/dense0', activation='softplus')(x)

        # Input - Output
        inputs = [robot_state_input]
        inputs = inputs + [depth_image_submodel.inputs[0],
                        action_submodel.inputs[0]]

        outputs = [info_output_layer, pos_output_layer, yaw_output_layer]
        model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name='info_gain_net')
        return model

    def compile(self, optimizer, loss=None, metrics=None):
        super(TrainIPN, self).compile(optimizer, loss, metrics)
        self.loss_tracker = keras.metrics.Mean(name="loss") # already in self.metrics
        self.output1_mae_tracker = keras.metrics.Mean(name="output_1_mae")
        self.output1_mse_tracker = keras.metrics.Mean(name="output_1_mse")
        self.output2_mae_tracker = keras.metrics.Mean(name="output_2_mae")
        self.output2_mse_tracker = keras.metrics.Mean(name="output_2_mse")
        self.output3_mae_tracker = keras.metrics.Mean(name="output_3_mae")
        self.output3_mse_tracker = keras.metrics.Mean(name="output_3_mse")

    @tf.function
    def train_step(self, data):
        image, actions, robot_state, collision_label, info_label, pos_label, height, width, depth, action_horizon = data
        with tf.GradientTape() as tape:
            #velocity = robot_state[:,3:6]
            #quaternion = robot_state[:,9:13]
            velocity_x = robot_state[:, 3]
            velocity_x = tf.expand_dims(velocity_x, axis=1)
            velocity_y = robot_state[:, 4]
            velocity_y = tf.expand_dims(velocity_y, axis=1)
            velocity_z = robot_state[:, 5]
            velocity_z = tf.expand_dims(velocity_z, axis=1)
            yaw_rate = robot_state[:, 15]
            yaw_rate = tf.expand_dims(yaw_rate, axis=1)
            roll_angle = robot_state[:, 16]
            roll_angle = tf.expand_dims(roll_angle, axis=1)
            pitch_angle = robot_state[:, 17]
            pitch_angle = tf.expand_dims(pitch_angle, axis=1)
            robot_state = tf.concat(
                [velocity_x, velocity_y, velocity_z, yaw_rate, roll_angle, pitch_angle], 1)
            [info_pred, pos_pred, yaw_pred] = self.predictor_model(
                [robot_state, image, actions[:, :, 0]], training=True)  # Forward pass
            # (the loss function is configured in `compile()`)
            info_pred = tf.squeeze(info_pred)

            # create the mask
            mask = LossBuilder.create_mask(collision_label) # (batch, ACTION_HORIZON)
            mask_expand = tf.expand_dims(mask, axis=-1) # (batch, ACTION_HORIZON, 1)

            output1_mae = LossBuilder.mae_loss_with_mask(info_label, info_pred, mask)
            output1_mse = LossBuilder.mse_loss_with_mask(info_label, info_pred, mask)
            output2_mae = LossBuilder.mae_loss_with_mask(pos_label[:, :, 0:3], pos_pred, mask_expand)
            output2_mse = LossBuilder.mse_loss_with_mask(pos_label[:, :, 0:3], pos_pred, mask_expand)
            output3_mae = LossBuilder.mae_loss_with_mask(pos_label[:, :, 3], yaw_pred, mask)
            output3_mse = LossBuilder.mse_loss_with_mask(pos_label[:, :, 3], yaw_pred, mask)
            loss_raw = self.alpha1 * output1_mse + self.alpha2 * output2_mse + self.alpha3 * output3_mse
            loss = loss_raw + sum(self.losses)  # add regularization term 
        
        self.loss_tracker.update_state(loss_raw)
        self.output1_mae_tracker.update_state(output1_mae)
        self.output1_mse_tracker.update_state(output1_mse)
        self.output2_mae_tracker.update_state(output2_mae)
        self.output2_mse_tracker.update_state(output2_mse)
        self.output3_mae_tracker.update_state(output3_mae)
        self.output3_mse_tracker.update_state(output3_mse)

        # Compute gradients
        gradients = tape.gradient(loss, self.predictor_model.trainable_weights)

        # Update weights
        self.optimizer.apply_gradients(
            zip(gradients, self.predictor_model.trainable_weights))

        # Update metrics (includes the metric that tracks the loss)
        # self.compiled_metrics.update_state(
        #     [info_label, pos_label[:, :, 0:3], pos_label[:, :, 3]], [info_pred, pos_pred, yaw_pred])

        # Return a dict mapping metric names to current value
        result_dict = {m.name: m.result() for m in self.metrics}

        return result_dict

    @tf.function
    def test_step(self, data):
        # Unpack the data
        image, actions, robot_state, collision_label, info_label, pos_label, height, width, depth, action_horizon = data
        velocity_x = robot_state[:, 3]
        velocity_x = tf.expand_dims(velocity_x, axis=1)
        velocity_y = robot_state[:, 4]
        velocity_y = tf.expand_dims(velocity_y, axis=1)
        velocity_z = robot_state[:, 5]
        velocity_z = tf.expand_dims(velocity_z, axis=1)
        yaw_rate = robot_state[:, 15]
        yaw_rate = tf.expand_dims(yaw_rate, axis=1)
        roll_angle = robot_state[:, 16]
        roll_angle = tf.expand_dims(roll_angle, axis=1)
        pitch_angle = robot_state[:, 17]
        pitch_angle = tf.expand_dims(pitch_angle, axis=1)
        robot_state = tf.concat(
            [velocity_x, velocity_y, velocity_z, yaw_rate, roll_angle, pitch_angle], 1)
        # Compute predictions
        [info_pred, pos_pred, yaw_pred] = self.predictor_model(
            [robot_state, image, actions[:, :, 0]], training=False)  # Forward pass
        info_pred = tf.squeeze(info_pred)

        # create the mask
        mask = LossBuilder.create_mask(collision_label) # (batch, ACTION_HORIZON)
        mask_expand = tf.expand_dims(mask, axis=-1) # (batch, ACTION_HORIZON, 1)

        output1_mae = LossBuilder.mae_loss_with_mask(info_label, info_pred, mask)
        output1_mse = LossBuilder.mse_loss_with_mask(info_label, info_pred, mask)
        output2_mae = LossBuilder.mae_loss_with_mask(pos_label[:, :, 0:3], pos_pred, mask_expand)
        output2_mse = LossBuilder.mse_loss_with_mask(pos_label[:, :, 0:3], pos_pred, mask_expand)
        output3_mae = LossBuilder.mae_loss_with_mask(pos_label[:, :, 3], yaw_pred, mask)
        output3_mse = LossBuilder.mse_loss_with_mask(pos_label[:, :, 3], yaw_pred, mask)
        loss_raw = self.alpha1 * output1_mse + self.alpha2 * output2_mse + self.alpha3 * output3_mse # ignore regularization term  
        
        self.loss_tracker.update_state(loss_raw)
        self.output1_mae_tracker.update_state(output1_mae)
        self.output1_mse_tracker.update_state(output1_mse)
        self.output2_mae_tracker.update_state(output2_mae)
        self.output2_mse_tracker.update_state(output2_mse)
        self.output3_mae_tracker.update_state(output3_mae)
        self.output3_mse_tracker.update_state(output3_mse)
        
        # Update the metrics.
        # self.compiled_metrics.update_state(
        #     [info_label, pos_label[:, :, 0:3], pos_label[:, :, 3]], [info_pred, pos_pred, yaw_pred])
        
        # Return a dict mapping metric names to current value
        # Note that it will include the loss (tracked in self.metrics).
        result_dict = {m.name: m.result() for m in self.metrics}
        
        return result_dict

    def call(self, inputs):
        return self.predictor_model(inputs, training=False)

    def summary(self, line_length=None, positions=None, print_fn=None):
        return self.predictor_model.summary(line_length, positions, print_fn)

    def get_model(self):
        return self.predictor_model

class InferenceIPN():  # for inference, we need to run feature extractor and rnn parts separately
    def __init__(self, depth_image_shape):
        self.feature_extractor = NetworkBuilder.depth_image_cnn_info_small_new_v3(depth_image_shape)
        self.feature_outputs = self.feature_extractor.outputs[0]
        self.predictor_net = self.build_info_gain_predictor()
        self.model = tf.keras.Model(inputs = [self.predictor_net.inputs[0], self.feature_extractor.inputs[0], self.predictor_net.inputs[2]],
                                    outputs = self.predictor_net([self.predictor_net.inputs[0], self.feature_extractor.outputs[0], self.predictor_net.inputs[2]]), name='info_gain_net')

    def build_info_gain_predictor(self):
        # state input
        robot_state_input = tf.keras.Input(
            shape=STATE_INPUT_SHAPE, name="robot_state_input")
        robot_state_processed = tf.keras.layers.Dense(
            32, activation='relu', name='robot_state/dense0')(robot_state_input)
        robot_state_processed = tf.keras.layers.Dense(
            32, activation='relu', name='robot_state/dense1')(robot_state_processed)
        robot_state_processed = tf.keras.layers.Dense(
            32, activation=None, name='robot_state/dense2')(robot_state_processed)

        initial_state_h, initial_state_c = tf.split(robot_state_processed, 2, axis=1)

        # Action model
        action_submodel = NetworkBuilder.action_network(action_shape=(
            ACTION_HORIZON, ACTION_SHAPE_EVALUATE))    

        # 1D LSTM for robot's state
        recurrent_layer = tf.keras.layers.LSTM(16, name='recurrent_layer_robot_state', return_sequences=True)(
            inputs=action_submodel.outputs[0], initial_state=[initial_state_h, initial_state_c])

        state_output_layer = tf.keras.layers.Dense(
            units=32, name='state_feature/dense0', activation='relu')(recurrent_layer)
        state_output_layer = tf.keras.layers.Dense(
            units=32, name='state_feature/dense1', activation=None)(state_output_layer)

        cnn_feature_input = tf.keras.Input(
            shape=(34, 60, 32), name="cnn_feature_input")    
        # cnn_feature_layer = tf.expand_dims(cnn_feature_input, axis=1)
        # cnn_feature_layer = tf.tile(cnn_feature_layer, [1, ACTION_HORIZON, 1, 1, 1])

        # conc_layer = tf.keras.layers.concatenate(
        #         [state_output_layer, cnn_feature_layer], name="concatenate_features")

        # cnn_feature_layer2 = tf.keras.layers.BatchNormalization(name='batch_normalization_6')(cnn_feature_input)
        # cnn_feature_layer2 = tf.keras.layers.Activation('relu')(cnn_feature_layer2)

        # Combined with state feature
        state_fearure1 = tf.keras.layers.Dense(
                    units=32, name='state_feature/dense2', activation='relu')(state_output_layer) # (None, ACTION_HORIZON, 32)
        state_fearure1 = tf.keras.layers.Dense(
                    units=32, name='state_feature/dense3', activation=None)(state_fearure1)
        state_fearure1 = tf.expand_dims(state_fearure1, axis=-2)
        state_fearure1 = tf.expand_dims(state_fearure1, axis=-2) # (None, ACTION_HORIZON, 1, 1, 32)
        
        # cnn_feature1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',
        #             kernel_initializer="he_normal",
        #             kernel_regularizer=regularizers.l2(1e-4), name='output_conv2d_0')(cnn_feature_layer2) # (None, 34, 60, 32)
        cnn_feature1 = tf.expand_dims(cnn_feature_input, axis=1) # (None, 1, 34, 60, 32)
        
        cnn_feature1 = tf.keras.layers.add([cnn_feature1, state_fearure1]) # (None, ACTION_HORIZON, 34, 60, 32)

        output_conv = cnn_feature1

        # residual block
        output_conv2 = tf.keras.layers.BatchNormalization(name='output_batch_normalization_0')(output_conv)
        output_conv2 = tf.keras.layers.Activation('relu')(output_conv2)

        output_conv2 = tf.keras.layers.Conv2D(32, (3, 3), strides=[2,2], padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4), name='output_conv2d_1')(output_conv2)
        output_conv2 = tf.keras.layers.BatchNormalization(name='output_batch_normalization_1')(output_conv2)
        output_conv2 = tf.keras.layers.Activation('relu')(output_conv2)

        output_conv2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4), name='output_conv2d_2')(output_conv2)
        
        output_conv3 = tf.keras.layers.Conv2D(32, (1, 1), strides=[2,2], padding='same', name='output_conv2d_3')(output_conv)
        output_conv4 = tf.keras.layers.add([output_conv2, output_conv3])

        # another residual block
        output_conv5 = tf.keras.layers.BatchNormalization(name='output_batch_normalization_2')(output_conv4)
        output_conv5 = tf.keras.layers.Activation('relu')(output_conv5)

        output_conv5 = tf.keras.layers.Conv2D(64, (3, 3), strides=[2,2], padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4), name='output_conv2d_4')(output_conv5)
        output_conv5 = tf.keras.layers.BatchNormalization(name='output_batch_normalization_3')(output_conv5)
        output_conv5 = tf.keras.layers.Activation('relu')(output_conv5)

        output_conv5 = tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4), name='output_conv2d_5')(output_conv5)

        output_conv6 = tf.keras.layers.Conv2D(64, (1, 1), strides=[2,2], padding='same', name='output_conv2d_6')(output_conv4)
        output_conv7 = tf.keras.layers.add([output_conv5, output_conv6])

        # no batch norm at the end
        # output_conv7 = tf.keras.layers.BatchNormalization(name='output_batch_normalization_4')(output_conv7)
        output_conv7 = tf.keras.layers.Activation('relu')(output_conv7)

        # Final prediction
        x = tf.keras.layers.Reshape((ACTION_HORIZON, -1))(output_conv7)
        info_output_layer = tf.keras.layers.Dense(
            units=1, name='info_gain_output/dense0', activation='softplus')(x)

        # Input - Output
        inputs = [robot_state_input, cnn_feature_input, action_submodel.inputs[0]]
        outputs = [info_output_layer]
        # if predict_pos:
        #     outputs = outputs + [pos_output_layer]
        model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name='info_gain/predictor_net')
        return model

    def load_model(self, model):
        trainable_layers = ['conv2d_0', 'conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4', 'conv2d_5', 'conv2d_6', 'conv2d_7', 'conv2d_8', 'conv2d_9',
                            'batch_normalization_0', 'batch_normalization_1', 'batch_normalization_2', 'batch_normalization_3', 'batch_normalization_4', 'batch_normalization_5', 'batch_normalization_6',
                            'robot_state/dense0', 'robot_state/dense1', 'robot_state/dense2',
                            'action/dense0', 'action/dense1', 'recurrent_layer_robot_state',
                            'state_feature/dense0', 'state_feature/dense1', 'state_feature/dense2', 'state_feature/dense3',
                            'output_conv2d_0', 'output_conv2d_1', 'output_conv2d_2', 'output_conv2d_3', 'output_conv2d_4', 'output_conv2d_5', 'output_conv2d_6',
                            'output_batch_normalization_0', 'output_batch_normalization_1', 'output_batch_normalization_2', 
                            'info_gain_output/dense0',
                            'pos_output/dense0', 'pos_output/dense1']
        # for layer in model.layers:
        #     print('layer name:', layer.name)

        for layer in self.feature_extractor.layers:
            if layer.name in trainable_layers:
                layer.set_weights(model.get_layer(layer.name).get_weights())

        for layer in self.predictor_net.layers:
            if layer.name in trainable_layers:
                layer.set_weights(model.get_layer(layer.name).get_weights())

    def call_feature_extractor(self, inputs):
        return self.feature_extractor.predict(inputs)

    def call_predictor_net(self, inputs):
        return self.predictor_net.predict_on_batch(inputs)

    def summary(self):
        return self.model.summary()

    def get_model(self):
        return self.model

    def get_feature_extractor(self):
        return self.feature_extractor

    def get_predictor(self):
        return self.predictor_net

    def get_di_feature_size(self):
        return self.feature_outputs[0].shape[1:]

class DataLoader():
    @staticmethod
    def read_tfrecords_with_detection_mask(serialized_example):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'actions': tf.io.FixedLenFeature([], tf.string),
            'robot_state': tf.io.FixedLenFeature([], tf.string),
            'pca_state': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.string),
            'info_gain_t0_label': tf.io.FixedLenFeature([], tf.float32),
            'info_gain_label': tf.io.FixedLenFeature([], tf.string),
            'pos_label': tf.io.FixedLenFeature([], tf.string),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'action_horizon': tf.io.FixedLenFeature([], tf.int64)
        }
        example = tf.io.parse_single_example(
            serialized_example, feature_description)

        image = tf.cast(tf.io.parse_tensor(
            example['image'], out_type=tf.uint16), tf.float32) * tf.constant([0.001 * MAX_RANGE_INV, 0.001]) # scale the image to 0->1
        actions = tf.io.parse_tensor(example['actions'], out_type=tf.float32)
        robot_state = tf.io.parse_tensor(
            example['robot_state'], out_type=tf.float32)
        # pca_state = tf.io.parse_tensor(example['pca_state'], out_type=tf.float32)
        collision_label = tf.io.parse_tensor(example['label'], out_type=tf.float32)
        # info_label_t0 = example['info_gain_t0_label']
        info_label = tf.io.parse_tensor(
            example['info_gain_label'], out_type=tf.float32)
        info_label = info_label * 0.01 # try to scale the label to 0->1 (more or less)
        pos_label = tf.io.parse_tensor(example['pos_label'], out_type=tf.float32)
        height = example['height']
        width = example['width']
        depth = example['depth']
        action_horizon = example['action_horizon']

        return image, actions, robot_state, collision_label, info_label, pos_label, height, width, depth, action_horizon

    @staticmethod
    def read_tfrecords(serialized_example):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'actions': tf.io.FixedLenFeature([], tf.string),
            'robot_state': tf.io.FixedLenFeature([], tf.string),
            'pca_state': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.string),
            'info_gain_t0_label': tf.io.FixedLenFeature([], tf.float32),
            'info_gain_label': tf.io.FixedLenFeature([], tf.string),
            'pos_label': tf.io.FixedLenFeature([], tf.string),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'action_horizon': tf.io.FixedLenFeature([], tf.int64)
        }
        example = tf.io.parse_single_example(
            serialized_example, feature_description)

        image = tf.cast(tf.io.parse_tensor(
            example['image'], out_type=tf.uint16), tf.float32) * 0.001 * MAX_RANGE_INV # scale the image to 0->1
        actions = tf.io.parse_tensor(example['actions'], out_type=tf.float32)
        robot_state = tf.io.parse_tensor(
            example['robot_state'], out_type=tf.float32)
        # pca_state = tf.io.parse_tensor(example['pca_state'], out_type=tf.float32)
        collision_label = tf.io.parse_tensor(example['label'], out_type=tf.float32)
        # info_label_t0 = example['info_gain_t0_label']
        info_label = tf.io.parse_tensor(
            example['info_gain_label'], out_type=tf.float32)
        info_label = info_label * 0.01 # try to scale the label to 0->1 (more or less)
        pos_label = tf.io.parse_tensor(example['pos_label'], out_type=tf.float32)
        height = example['height']
        width = example['width']
        depth = example['depth']
        action_horizon = example['action_horizon']

        return image, actions, robot_state, collision_label, info_label, pos_label, height, width, depth, action_horizon

    @staticmethod
    def read_tfrecords_sevae(serialized_example):
        feature_description = {
            'latent_space': tf.io.FixedLenFeature([], tf.string),
            'actions': tf.io.FixedLenFeature([], tf.string),
            'robot_state': tf.io.FixedLenFeature([], tf.string),
            'pca_state': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.string),
            'info_gain_t0_label': tf.io.FixedLenFeature([], tf.float32),
            'info_gain_label': tf.io.FixedLenFeature([], tf.string),
            'pos_label': tf.io.FixedLenFeature([], tf.string),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'action_horizon': tf.io.FixedLenFeature([], tf.int64)
        }
        example = tf.io.parse_single_example(
            serialized_example, feature_description)

        latent_space = tf.io.parse_tensor(example['latent_space'], out_type=tf.float32)

        actions = tf.io.parse_tensor(example['actions'], out_type=tf.float32)
        robot_state = tf.io.parse_tensor(
            example['robot_state'], out_type=tf.float32)
        # pca_state = tf.io.parse_tensor(example['pca_state'], out_type=tf.float32)
        collision_label = tf.io.parse_tensor(example['label'], out_type=tf.float32)
        # info_label_t0 = example['info_gain_t0_label']
        info_label = tf.io.parse_tensor(
            example['info_gain_label'], out_type=tf.float32)
        info_label = info_label * 0.01 # try to scale the label to 0->1 (more or less)
        pos_label = tf.io.parse_tensor(example['pos_label'], out_type=tf.float32)
        height = example['height']
        width = example['width']
        depth = example['depth']
        action_horizon = example['action_horizon']

        return latent_space, actions, robot_state, collision_label, info_label, pos_label, height, width, depth, action_horizon

    @staticmethod
    def load_tfrecords(tfrecord_folders, training_type_str, is_shuffle_and_repeat=True, shuffle_buffer_size=5000, prefetch_buffer_size_multiplier=2, batch_size=32):
        print('Loading tfrecords...')
        # for tfrecord_folder in self.tfrecord_folders:
        #     logger.debug(tfrecord_folder)
        tfrecord_fnames = utilities.get_files_ending_with(
            tfrecord_folders, '.tfrecords')
        assert len(tfrecord_fnames) > 0
        if is_shuffle_and_repeat:
            np.random.shuffle(tfrecord_fnames)
        else:
            tfrecord_fnames = sorted(tfrecord_fnames)

        dataset = tf.data.TFRecordDataset(tfrecord_fnames)

        if training_type_str == "ORACLE":
            dataset = dataset.map(DataLoader.read_tfrecords,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)        
        elif training_type_str == "seVAE-ORACLE":
            dataset = dataset.map(DataLoader.read_tfrecords_sevae,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        elif training_type_str == "A-ORACLE":
            dataset = dataset.map(DataLoader.read_tfrecords_with_detection_mask,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if is_shuffle_and_repeat:
            #dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=shuffle_buffer_size))
            dataset = dataset.shuffle(
                buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)  # seed???
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(
            buffer_size=prefetch_buffer_size_multiplier * batch_size)

        # iterator = dataset.__iter__() # image, actions, label, height, width, depth, action_horizon = next(itr)

        return dataset  # iterator

if __name__ == "__main__":
    try:
        argv = FLAGS(sys.argv)  # parse flags
    except gflags.FlagsError:
        print('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
        sys.exit(1)

    training_type_dict = {0: "ORACLE",
                          1: "seVAE-ORACLE",
                          2: "A-ORACLE"}

    if FLAGS.training_type < 0:
        FLAGS.training_type = 0
    elif FLAGS.training_type > len(training_type_dict) - 1:
        FLAGS.training_type = len(training_type_dict) - 1

    training_type_str = training_type_dict[FLAGS.training_type]
    print(bcolors.OKGREEN + "Training type " + str(FLAGS.training_type) + ": " +
          training_type_str + bcolors.ENDC)

    # Create a callback that saves the model's weights
    if not os.path.exists(FLAGS.model_save_path):
        os.mkdir(FLAGS.model_save_path)
    filepath = "saved-model-{epoch:02d}.hdf5"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=FLAGS.model_save_path+'/'+filepath, save_best_only=False, mode='auto',
                                                     save_weights_only=False, verbose=1, save_freq='epoch')

    # load train data
    train_tf_folder = FLAGS.train_tf_folder
    train_dataset = DataLoader.load_tfrecords(
        train_tf_folder, training_type_str, batch_size=64)

    # load validation data
    validate_tf_folder = FLAGS.validate_tf_folder
    validate_dataset = DataLoader.load_tfrecords(
        validate_tf_folder, training_type_str, batch_size=64)

    # setup log folder
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = FLAGS.metrics_log_dir + '/' + current_time
    file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
    file_writer.set_as_default()

    # setup plot callback
    plot_epoch_flag = True
    if plot_epoch_flag:
        loss_acc_callback = LossAndAccuracyEpoch()
    else:
        loss_acc_callback = LossAndAccuracyBatch()
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir, profile_batch=0)

    # print data, visualize image
    # for image, actions, robot_state, collision_label, info_label, pos_label, height, width, depth, action_horizon in train_dataset:
        # img_idx = 0 # view first image in the batch
        # print('collision label:', collision_label[img_idx,...])
        # print('info_gain t0 label:', info_label_t0)
        # print('info_gain t0 label:', info_label_t0[img_idx])
        # print('info_gain label:', info_label[img_idx,...])
        # print('actions:', np.shape(actions))
        # print('robot_state:', robot_state[img_idx,...])
        # print('pos_label:', pos_label[img_idx,...])
        # io.imshow((image[img_idx,...,0].numpy() / MAX_RANGE * 255).astype('uint8'))
        # io.show()

                
    if training_type_str == "A-ORACLE":
        # predictor_model = build_info_gain_prediction_network(
        #     depth_image_shape=DI_WITH_MASK_SHAPE, predict_pos=True, predict_yaw=True, use_residual_at_output = True)
        custom_predictor_model = TrainIPN(depth_image_shape=DI_WITH_MASK_SHAPE)
        # predictor_model.summary()
    else: # ORACLE and seVAE-ORACLE 
        pos = 1
        neg = 1
        weight_for_0 = (1 / neg)*(pos + neg)/2.0 
        weight_for_1 = (1 / pos)*(pos + neg)/2.0
        print('Weight for class 0: {:.2f}'.format(weight_for_0))
        print('Weight for class 1: {:.2f}'.format(weight_for_1))
        if training_type_str == "ORACLE":
            custom_predictor_model = TrainCPN(depth_image_shape=DI_SHAPE, output_bias=np.log([pos/neg]))
        else: # seVAE-ORACLE
            custom_predictor_model = TrainCPNseVAE(depth_image_shape=DI_LATENT_SIZE, output_bias=np.log([pos/neg]))
    custom_predictor_model.summary()
    if training_type_str == "A-ORACLE":
        plot_model(custom_predictor_model.get_model(), to_file='info_gain_model_plot.png',
                show_shapes=True, show_layer_names=True)
    else:
        plot_model(custom_predictor_model.get_model(), to_file='collision_model_plot.png',
                show_shapes=True, show_layer_names=True)

    # create and evaluate the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

    if training_type_str == "A-ORACLE": # custom loss fcn
        custom_predictor_model.compile(optimizer=optimizer)
        eval_result = custom_predictor_model.evaluate(validate_dataset, use_multiprocessing=True, verbose=2)
        print(bcolors.OKBLUE + "Untrained model, eval result:" + str(eval_result) + bcolors.ENDC)
    else: # ORACLE and seVAE-ORACLE
        custom_predictor_model.compile(optimizer=optimizer) # metrics=tf.keras.metrics.AUC(name="auc")
        eval_result = custom_predictor_model.evaluate(validate_dataset, use_multiprocessing=True, verbose=2)
        print(bcolors.OKBLUE + "Untrained model, eval result:" + str(eval_result) + bcolors.ENDC)
        custom_predictor_model.reset_custom_metrics()    

    # train
    if training_type_str == "A-ORACLE":
        history = custom_predictor_model.fit(train_dataset, epochs=5000, validation_data=validate_dataset,
                                            callbacks=[tensorboard_callback, cp_callback], use_multiprocessing=True, verbose=2)
    else: # ORACLE and seVAE-ORACLE
        history = custom_predictor_model.fit(train_dataset, epochs=5000, validation_data=validate_dataset, 
                                            callbacks=[tensorboard_callback, loss_acc_callback, cp_callback], use_multiprocessing=True, verbose=2)