#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import h5py
import numpy as np
import os.path as osp
import os
import random

from tensorflow.contrib import layers
from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse
from skimage.io import imread, imsave


MAX_ITERS = 400 # maximum number of epochs -- will be much much higher than 100
BATCH_SIZE = 16 # batch size
NUM_TESTS = 100 # number of test cases
INPUT_COUNT = 20 # size of user input
THRESH = 0.1 # threshold for edge
BASE_PATH = "/mnt/disks/disk-dir/" # path in directory to persistent memory
SUMMARIES_PATH = "/home/samuelarnesen/jp-spring/summaries/"
SAVE_PATH = BASE_PATH + "models/"

# builds model architecture
def build_model():

	# placeholder values
	training_bool = tf.placeholder(dtype=tf.bool, name="training_bool")
	voxel_input = tf.placeholder(dtype=tf.float64, shape=[BATCH_SIZE, 32, 32, 32, 1], name="voxel_input") # input voxel grid
	ground_truth = tf.placeholder(dtype=tf.float64, shape=[BATCH_SIZE, 32, 32, 32, 1], name="ground_truth") # ground truth
	binary_mask = tf.placeholder(dtype=tf.float64, shape=[BATCH_SIZE, 32, 32, 32, 1], name="binary_mask") # areas to consider for loss

	# this downsamples from a two-channel 32x32x32 grid to a single length 320 vector
	ds_1 = tf.layers.conv3d(inputs=tf.concat([voxel_input, binary_mask], 4), filters=40, kernel_size=[4, 4, 4], strides=2, padding='same')
	ds_1 = tf.layers.batch_normalization(inputs=ds_1, momentum=0.9, training=training_bool)
	ds_2 = tf.layers.conv3d(inputs=ds_1, filters=80, kernel_size=[4, 4, 4], strides=2, padding='same')
	ds_2 = tf.layers.batch_normalization(inputs=ds_2, momentum=0.9, training=training_bool)
	ds_3 = tf.layers.conv3d(inputs=ds_2, filters=160, kernel_size=[4, 4, 4], strides=2, padding='same')
	ds_3 = tf.layers.batch_normalization(inputs=ds_3, momentum=0.9, training=training_bool)
	ds_4 = tf.layers.conv3d(inputs=ds_3, filters=320, kernel_size=[4, 4, 4], strides=2)

	# bottleneck layers
	fc_1 = tf.contrib.layers.fully_connected(inputs=ds_4, num_outputs=320)
	fc_2 = tf.contrib.layers.fully_connected(inputs=fc_1, num_outputs=320)

	# upsamples to get output 
	us_1 = tf.layers.conv3d_transpose(inputs=tf.concat([ds_4, fc_2], 4), filters=160, kernel_size=[4, 4, 4], strides=2)
	us_1 = tf.layers.batch_normalization(inputs=us_1, momentum=0.9, training=training_bool)
	us_2 = tf.layers.conv3d_transpose(inputs=tf.concat([ds_3, us_1], 4), filters=80, kernel_size=[4, 4, 4], strides=2, padding='same')
	us_2 = tf.layers.batch_normalization(inputs=us_2, momentum=0.9, training=training_bool)
	us_3 = tf.layers.conv3d_transpose(inputs=tf.concat([ds_2, us_2], 4), filters=40, kernel_size=[4, 4, 4], strides=2, padding='same')
	prediction = tf.layers.conv3d_transpose(inputs=tf.concat([ds_1, us_3], 4), filters=1, kernel_size=[4, 4, 4], strides=2, padding='same', name="prediction")

	# gets loss
	bool_mask = tf.cast((binary_mask - 1) / -2, tf.bool)
	masked_out = abs(tf.boolean_mask(prediction - ground_truth, bool_mask))
	loss = tf.reduce_mean(masked_out, name="loss")

	return voxel_input, ground_truth, binary_mask, training_bool, loss, prediction

# builds file list
def create_file_list(filelist_file):
	file_list = list()
	with open(filelist_file) as f:
		for line in f:
			file_list.append(line)

	return file_list

# fills in values for voxel_input, ground_truth, binary_mask 
def load_batch(file_list, include_user=False, prioritize_edge=False):

	# declares training data inputs
	voxel_input = np.zeros([BATCH_SIZE, 32, 32, 32], dtype=np.float64)
	ground_truth = np.zeros([BATCH_SIZE, 32, 32, 32], dtype=np.float64)
	binary_mask = np.zeros([BATCH_SIZE, 32, 32, 32], dtype=np.float64)
 
	# loads file
	file_index = random.randrange(len(file_list))
	h5_file = h5py.File(BASE_PATH + file_list[file_index][:len(file_list[file_index]) - 1], 'r')
	data = h5_file["data"]
	target = h5_file["target"]

	# gets starting pt 
	num_entries = np.shape(data)[0]
	starting_pt = random.randrange(0, num_entries - BATCH_SIZE)
	ending_pt = starting_pt + BATCH_SIZE

	# loads in data
	voxel_input[0:BATCH_SIZE, ...] = data[starting_pt:ending_pt, 0, ...]
	binary_mask[0:BATCH_SIZE, ...] = data[starting_pt:ending_pt, 1, ...]
	ground_truth[0:BATCH_SIZE, ...] = np.squeeze(target[starting_pt:ending_pt])
	
	# removes NaNs
	vi_nan_loc = np.isnan(voxel_input)
	bm_nan_loc = np.isnan(binary_mask)
	voxel_input[vi_nan_loc] = 0.0
	binary_mask[bm_nan_loc] = 0.0

	# removes infinities
	ninf = float('-inf')
	voxel_input[voxel_input == ninf] = 0.0

	# adds user input
	if include_user == True:
		for i in range(BATCH_SIZE):
			user_count = 0
			while user_count < INPUT_COUNT:
				x = random.randint(0, 31)
				y = random.randint(0, 31)
				z = random.randint(0, 31)
				if binary_mask[i][x][y][z] == -1 and (abs(ground_truth[i][x][y][z]) < THRESH or ~prioritize_edge):
					binary_mask[i][x][y][z] = 1
					voxel_input[i][x][y][z] = ground_truth[i][x][y][z]
					user_count += 1
	
	return np.expand_dims(voxel_input, -1), np.expand_dims(binary_mask, -1), np.expand_dims(ground_truth, -1),

# trains model and then tests
def train():
	# value to return
	overall_loss = 0
	avg_loss = -1

	# defines operations to build model
	voxel_input, ground_truth, binary_mask, training_bool, loss, prediction = build_model()
	train_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9).minimize(loss) # Adam optimizer
	saver_op = tf.train.Saver() # allows the saving and restoring of models

	# creates the list of files
	training_files = create_file_list(BASE_PATH + "h5_shapenet_dim32_sdf/train_shape_voxel_data_list.txt")
	test_files = create_file_list(BASE_PATH + "h5_shapenet_dim32_sdf/test_shape_voxel_data_list.txt")

	# defines summaries
	train_loss_summ = tf.summary.scalar('sui_training_loss', loss)
	test_loss_summ = tf.summary.scalar('sui_training_loss', loss)
	train_writer = tf.summary.FileWriter(SUMMARIES_PATH + "training/")
	test_writer = tf.summary.FileWriter(SUMMARIES_PATH + "testing/")

	# configures options to allow for growing memory usage 
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	# runs 
	with tf.device("/gpu:0"): # specifies particular GPU
		with tf.Session(config=config) as sess: # runs tf session

			# initializes global variables
			sess.run(tf.global_variables_initializer()) 
			
			# trains model for MAX_ITERS number of epochs
			print("BEGINNING TRAINING")
			feed_dict = dict()
			for i in range(MAX_ITERS):
				vi, bm, gt = load_batch(training_files, True, True)
				if i == 0:
					print("batch loaded!")
				feed_dict = {
                    voxel_input: vi,
                    ground_truth: gt,
                    binary_mask: bm,
                    training_bool: True
                }
				loss_val, summ, _ = sess.run([loss, train_loss_summ, train_op], feed_dict=feed_dict)
				train_writer.add_summary(summ, i)
				print(str(i) + "\t" + str(loss_val))

				# saves model
				saver_op.save(sess, SAVE_PATH + "sui_model.ckpt")

			# tests model 
			print("\nBEGINNING TESTING")
			BATCH_SIZE = 1
			for i in range(NUM_TESTS):
				vi, bm, gt = load_batch(test_files, True, True)
				feed_dict = {
                    voxel_input: vi,
                    ground_truth: gt,
                    binary_mask: bm,
                    training_bool: False
                }
				test_loss_val, test_summ = sess.run([loss, test_loss_summ], feed_dict=feed_dict)
				test_writer.add_summary(test_summ, i)
				print(str(i) + "\t" + str(test_loss_val))
				overall_loss += test_loss_val


		if NUM_TESTS > 0:
			avg_loss = overall_loss / NUM_TESTS
			print(avg_loss)

		return avg_loss


if __name__ == '__main__':
    train()


