import pickle
import os
import numpy as np
import tensorflow as tf
from os import path

import data_movielens_100k
import utils

# disable all debugging logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

NUM_DEBUG_USER = 8

PERSONAL_NDCG_NAME = 'personal-ndcg'
ALPHA_NDCG_NAME = 'alpha-ndcg'

def get_ideal_filepath(filepath, measure_name):
	return '{}.{}.p'.format(filepath, measure_name)

class EvaluationMeasure(object):
	def __init__(self, sess, data_dir, cutoff=10, num_fold=utils.NUM_FOLD):
		user_filepath = path.join(data_dir, 'users.csv')
		item_filepath = path.join(data_dir, 'items.csv')
		feat_filepath = path.join(data_dir, 'feats.csv')
		rate_filepath = path.join(data_dir, 'rates.csv')
		alpha = 0.50
		
		self.sess = sess
		self.num_fold, self.cutoff = num_fold, cutoff

		num_user = utils.count_num_line(user_filepath)
		num_item = utils.count_num_line(item_filepath)
		num_genre = utils.NUM_GENRE
		print('#users={}\t#items={}'.format(num_user, num_item))
		self.num_user, self.num_item, self.num_genre, self.nonexist_item = num_user, num_item, num_genre, num_item

		items, feats, sparse_values = utils.load_triple(feat_filepath)
		output_shape = [num_item + 1, num_genre]
		sparse_indices = utils.np_build_indices(items, feats)
		self.feat_matrix = tf.sparse_to_dense(sparse_indices, output_shape, sparse_values, validate_indices=False)

		users, items, sparse_values = utils.load_triple(rate_filepath)
		output_shape = [num_user, num_item + 1]
		sparse_indices = utils.np_build_indices(users, items)
		self.rate_matrix = tf.subtract(tf.pow(2.0, tf.sparse_to_dense(sparse_indices, output_shape, sparse_values, validate_indices=False)), 1.0)
		
		# num_user, num_genre
		self.genre_preference_unnormalized = tf.reduce_sum(tf.multiply(
				tf.tile(tf.expand_dims(tf.slice(self.rate_matrix, [0, 0], [num_user, num_item]), 2), [1, 1, num_genre]),
				tf.tile(tf.expand_dims(tf.slice(self.feat_matrix, [0, 0], [num_item, num_genre]), 0), [num_user, 1, 1])), 1)
		# num_user, num_genre
		self.genre_preference = tf.divide(self.genre_preference_unnormalized, 
				tf.tile(tf.expand_dims(tf.reduce_sum(self.genre_preference_unnormalized, 1), 1), [1, num_genre]))
		self.genre_punishment = self.genre_preference
		# num_genre
		self.genre_generality = tf.divide(tf.reduce_sum(self.feat_matrix, 0), num_item)

		self.user = tf.placeholder(utils.TF_INT_DTYPE, shape=())
		self.item_list = tf.placeholder(utils.TF_INT_DTYPE, shape=(cutoff))
		
		self.user_rate = tf.gather_nd(self.rate_matrix, utils.tf_build_indices(tf.tile(tf.expand_dims(self.user, 0), [cutoff]), self.item_list))
		user_rate = self.user_rate

		genre_discount = tf.nn.embedding_lookup(self.genre_punishment, self.user)
		# cutoff, num_genre
		user_rec_feats = tf.multiply(tf.nn.embedding_lookup(self.feat_matrix, self.item_list), 
				tf.tile(tf.expand_dims(user_rate, 1), [1, num_genre]))
		# cutoff, num_genre
		user_rec_feats_shift = tf.concat((tf.zeros([1, num_genre]), tf.slice(user_rec_feats, [0, 0], [cutoff - 1, num_genre])), 0)
		mask_sparse_indices = tf.where(tf.not_equal(user_rec_feats_shift, tf.constant(0.0, dtype=tf.float32)))
		# cutoff, num_genre
		relevant_mask = tf.sparse_to_dense(mask_sparse_indices, [cutoff, num_genre], 1.0)
		# cutoff, num_genre
		irrelevant_mask = tf.sparse_to_dense(mask_sparse_indices, [cutoff, num_genre], 0.0, default_value=1.0)
		# cutoff, num_genre
		redundancy_punishment_personal = tf.cumprod(tf.add(irrelevant_mask, tf.multiply(relevant_mask, tf.tile(tf.expand_dims(genre_discount, 0), [cutoff, 1]))))
		# cutoff, num_genre
		redundancy_punishment_alpha = tf.cumprod(tf.add(irrelevant_mask, tf.multiply(relevant_mask, tf.multiply(tf.ones([cutoff, num_genre]), 1.0 - alpha))))
		# cutoff
		position_punishment = tf.divide(tf.log(tf.cast(tf.range(2, 2 + cutoff), dtype=utils.TF_FLOAT_DTYPE)),
				tf.log(tf.cast(tf.multiply(tf.ones([cutoff]), 2.0), dtype=utils.TF_FLOAT_DTYPE)))

		## no redundancy punishment, no position punishment
		# self.personal_dcg = tf.reduce_sum(tf.reduce_sum(user_rec_feats, 1))
		## no redundancy punishment, position punishment
		# self.personal_dcg = tf.reduce_sum(tf.divide(tf.reduce_sum(user_rec_feats, 1), position_punishment))
		## redundancy punishment, position punishment
		self.personal_dcg = tf.reduce_sum(tf.divide(tf.reduce_sum(tf.multiply(user_rec_feats, redundancy_punishment_personal), 1), position_punishment))
		## no redundancy punishment, no position punishment
		# self.personal_idcg = tf.reduce_sum(tf.reduce_sum(user_rec_feats, 1))
		## redundancy punishment, no position punishment
		self.personal_idcg = tf.reduce_sum(tf.reduce_sum(tf.multiply(user_rec_feats, redundancy_punishment_personal), 1))

		## no redundancy punishment, no position punishment
		# self.alpha_dcg = tf.reduce_sum(tf.reduce_sum(user_rec_feats, 1))
		## no redundancy punishment, position punishment
		# self.alpha_dcg = tf.reduce_sum(tf.divide(tf.reduce_sum(user_rec_feats, 1), position_punishment))
		## redundancy punishment, position punishment
		self.alpha_dcg = tf.reduce_sum(tf.divide(tf.reduce_sum(tf.multiply(user_rec_feats, redundancy_punishment_alpha), 1), position_punishment))
		## redundancy punishment, no position punishment
		self.alpha_idcg = tf.reduce_sum(tf.reduce_sum(tf.multiply(user_rec_feats, redundancy_punishment_alpha), 1))

	def compute_normalizing_constant(self, infile, personal_ndcg_filepath, alpha_ndcg_filepath, outfile=None):
		#personal_ndcg_filepath = get_ideal_filepath(infile, PERSONAL_NDCG_NAME)
		#alpha_ndcg_filepath = get_ideal_filepath(infile, ALPHA_NDCG_NAME)

		if (outfile == None) and path.exists(personal_ndcg_filepath) and path.exists(alpha_ndcg_filepath):
			personal_idcg_values = pickle.load(open(personal_ndcg_filepath, 'rb'))
			alpha_idcg_values = pickle.load(open(alpha_ndcg_filepath, 'rb'))
		else:
			num_fold, cutoff = self.num_fold, self.cutoff
			num_user, num_item, num_genre, nonexist_item = self.num_user, self.num_item, self.num_genre, self.nonexist_item
			triples = data_movielens_100k.load_triple(infile, threshold=0.0)
			user_triples = data_movielens_100k.group_by_user(triples)
			
			save_ideal = False
			if outfile != None:
				save_ideal = True
			if save_ideal:
				fout = open(outfile, 'w')

			personal_idcg_values = np.full((num_user), 0.0, dtype=utils.NP_FLOAT_DTYPE)
			alpha_idcg_values = np.full((num_user), 0.0, dtype=utils.NP_FLOAT_DTYPE)
			for user, user_triple in user_triples.iteritems():
				triples = user_triple.triples
				num_triple = len(triples)

				candidates = set()
				for _, item, _ in triples:
					candidates.add(item)
				ideal_list = np.full((cutoff), nonexist_item, dtype=utils.NP_INT_DTYPE)
				for i in range(min(cutoff, num_triple)):
					best_item = None
					best_personal_idcg = -np.inf
					for item in candidates:
						item_list = ideal_list
						item_list[i] = item
						personal_idcg, = self.sess.run([self.personal_idcg], feed_dict={self.user:user, self.item_list:item_list})
						if personal_idcg > best_personal_idcg:
							best_item = item
							best_personal_idcg = personal_idcg
					if save_ideal:
						fout.write('{}\t{}\t1.0\n'.format(user, best_item))
					ideal_list[i] = best_item
					candidates.remove(best_item)
					# if user == 705:
					# 	print('item={} best personal idcg={}'.format(best_item, best_personal_idcg))
				personal_idcg, = self.sess.run([self.personal_dcg], feed_dict={self.user:user, self.item_list:ideal_list})
				personal_idcg_values[user] = personal_idcg

				candidates = set()
				for _, item, _ in triples:
					candidates.add(item)
				ideal_list = np.full((cutoff), nonexist_item, dtype=utils.NP_INT_DTYPE)
				for i in range(min(cutoff, num_triple)):
					best_item = None
					best_alpha_idcg = -np.inf
					for item in sorted(list(candidates)):
						item_list = ideal_list
						item_list[i] = item
						alpha_idcg, = self.sess.run([self.alpha_idcg], feed_dict={self.user:user, self.item_list:item_list})
						if alpha_idcg > best_alpha_idcg:
							best_item = item
							best_alpha_idcg = alpha_idcg
					ideal_list[i] = best_item
					candidates.remove(best_item)
				alpha_idcg, = self.sess.run([self.alpha_dcg], feed_dict={self.user:user, self.item_list:ideal_list})
				alpha_idcg_values[user] = alpha_idcg
				if save_ideal:
					fout.close()
				pickle.dump(personal_idcg_values, open(personal_ndcg_filepath, 'wb'))
				pickle.dump(alpha_idcg_values, open(alpha_ndcg_filepath, 'wb'))

		self.personal_idcg_values = personal_idcg_values
		self.alpha_idcg_values = alpha_idcg_values

	def compute_normalized_score(self, filepath):
		num_fold, cutoff = self.num_fold, self.cutoff
		num_user, num_item, num_genre, nonexist_item = self.num_user, self.num_item, self.num_genre, self.nonexist_item
		recommendations = np.full((num_user, cutoff), nonexist_item, dtype=utils.NP_INT_DTYPE)
		triples = data_movielens_100k.load_triple(filepath, threshold=0.0)
		user_triples = data_movielens_100k.group_by_user(triples)
		for user, user_triple in user_triples.iteritems():
			triples = user_triple.triples
			num_triple = len(triples)
			for i in range(min(cutoff, num_triple)):
				recommendations[user, i] = triples[i][1]

		genre_preference = self.sess.run([self.genre_preference])[0]
		personal_ndcg_values = np.full((num_user), 0.0, dtype=utils.NP_FLOAT_DTYPE)
		alpha_ndcg_values = np.full((num_user), 0.0, dtype=utils.NP_FLOAT_DTYPE)
		for user in range(num_user):
			item_list = recommendations[user]
			personal_dcg, alpha_dcg = self.sess.run([self.personal_dcg, self.alpha_dcg], feed_dict={self.user:user, self.item_list:item_list})
			personal_ndcg_values[user] = personal_dcg / self.personal_idcg_values[user]
			alpha_ndcg_values[user] = alpha_dcg / self.alpha_idcg_values[user]
			# if user == 705:
			# 	print('user={} personal dcg={} personal idcg={}'.format(user, personal_dcg, self.personal_idcg_values[user]))

		self.personal_ndcg_values = personal_ndcg_values
		self.alpha_ndcg_values = alpha_ndcg_values

		return np.mean(personal_ndcg_values), np.mean(alpha_ndcg_values)

if __name__ == '__main__':
	print('evaluation measure')
	#data_dir = utils.DATA_DIR_MOVIELENS_100K
        data_dir = "../itime_rec/data/ml-100k/"
	#ranksys_examples_dir = utils.RANKSYS_EXAMPLES_DIR
	test_filepath = path.join(data_dir, 'test0.csv')
	ltr_filepath = './prediction/cv0.pred'
	#ideal_filepath = path.join(ranksys_examples_dir, 'recommendations/ml-100k/personal-ndcg0.csv')

	#config = tf.ConfigProto()
	#config.gpu_options.allow_growth = True
	with tf.Session() as sess:
		measure = EvaluationMeasure(sess, data_dir)
		# measure.compute_normalizing_constant(test_filepath, ideal_filepath)
		measure.compute_normalizing_constant(test_filepath)
		personal_ndcg, alpha_ndcg = measure.compute_normalized_score(ltr_filepath)
		print('personal_ndcg={0:.4f} alpha_ndcg={1:.4f}'.format(personal_ndcg, alpha_ndcg))
		
