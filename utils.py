import csv
import operator
import os
import numpy as np
import tensorflow as tf
from os import path

DATA_DIR_MOVIELENS_100K = 'data/ml-100k'
HOME_DIR = path.expanduser('~')
PROJECTS_DIR = path.join(HOME_DIR, 'Projects')

RANKSYS_DIR = path.join(PROJECTS_DIR, 'RankSys')
RANKSYS_EXAMPLES_DIR = path.join(RANKSYS_DIR, 'RankSys-examples')

NUM_FOLD = 5
NUM_GENRE = 18
THRESHOLD_ITEM = 50
THRESHOLD_USER = 20
VALID_PROPORTION = 0.05

NP_INT_DTYPE = np.int32
NP_FLOAT_DTYPE = np.float32
TF_INT_DTYPE = tf.int32
TF_FLOAT_DTYPE = tf.float32

class Dataset(object):
	def __init__(self, data_dir, num_fold=NUM_FOLD):
		user_filepath = path.join(data_dir, 'users.csv')
		item_filepath = path.join(data_dir, 'items.csv')

		self.num_fold = num_fold

		num_user = count_num_line(user_filepath)
		num_item = count_num_line(item_filepath)
		print('#users={}\t#items={}'.format(num_user, num_item))
		self.num_user, self.num_item = num_user, num_item

		datasets = []
		for fold in range(num_fold):
			train_filepath = path.join(data_dir, 'train{0:1d}.csv'.format(fold))
			train_data = load_triple(train_filepath)
			test_filepath = path.join(data_dir, 'test{0:1d}.csv'.format(fold))
			test_data = load_triple(test_filepath)
			valid_filepath = path.join(data_dir, 'valid{0:1d}.csv'.format(fold))
			valid_data = load_triple(valid_filepath)
			dataset = train_data, test_data, valid_data
			datasets.append(dataset)
		self.datasets = datasets

	def get_dataset(self, fold):
		return self.datasets[fold]

def get_cv_index(data_size, num_fold):
	quotient, remainder = divmod(data_size, num_fold)
	cv_sizes = []
	for i in range(remainder):
		cv_sizes.append(quotient + 1)
	for i in range(num_fold - remainder):
		cv_sizes.append(quotient)
	cv_index = []
	idx_start = 0
	for cv_size in cv_sizes:
		idx_end = idx_start + cv_size
		cv_index.append((idx_start, idx_end))
		idx_start = idx_end
	return cv_index

def count_num_line(filepath):
	num_line = 0
	with open(filepath, 'r') as fin:
		for line in fin.readlines():
			num_line += 1
	return num_line

def load_triple(filepath):
	alp_elems, bet_elems, gam_elems = [], [], []
	with open(filepath, 'r') as f:
		for line in f.readlines():
			tokens = line.split()
			alp_elem = int(tokens[0])
			bet_elem = int(tokens[1])
			gam_elem = float(tokens[2])
			alp_elems.append(alp_elem)
			bet_elems.append(bet_elem)
			gam_elems.append(gam_elem)
	alp_elems = np.asarray(alp_elems, dtype=NP_INT_DTYPE)
	bet_elems = np.asarray(bet_elems, dtype=NP_INT_DTYPE)
	gam_elems = np.asarray(gam_elems, dtype=NP_FLOAT_DTYPE)
	dataset = alp_elems, bet_elems, gam_elems
	return dataset

def np_build_indices(row_index, col_index):
	return np.concatenate((row_index[:, None], col_index[:, None]), axis=1)

def tf_build_indices(row_index, col_index):
	return tf.concat([tf.expand_dims(row_index, 1), tf.expand_dims(col_index, 1)], 1)

if __name__ == '__main__':
	# disable all debugging logs
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
