import operator
import utils
import numpy as np
from os import path

class UserTriple(object):
	def __init__(self, user):
		self.user = user
		self.triples = []

	def append(self, triple):
		self.triples.append(triple)

	def randomize(self, seed=0):
		indices = np.arange(len(self.triples))
		np.random.seed(seed)
		np.random.shuffle(indices)
		self.indices = indices

	def gen_cv_index(self, num_fold=utils.NUM_FOLD):
		cv_index = utils.get_cv_index(len(self.triples), num_fold)
		self.cv_index = cv_index

	def get_train_triple(self, fold, num_fold=utils.NUM_FOLD):
		train_triples = []
		for f in range(num_fold):
			if f == fold:
				continue
			idx_start, idx_end = self.cv_index[f]
			for i in range(idx_start, idx_end):
				idx =  self.indices[i]
				train_triples.append(self.triples[idx])
		return train_triples

	def get_test_triple(self, fold):
		test_triples = []
		idx_start, idx_end = self.cv_index[fold]
		for i in range(idx_start, idx_end):
			idx =  self.indices[i]
			test_triples.append(self.triples[idx])
		return test_triples

	def get_valid_triple(self, fold):
		valid_triples = []
		idx_start, idx_end = self.cv_index[fold]
		num_test = idx_end - idx_start + 1
		num_valid = int(num_test * utils.VALID_PROPORTION)
		idx_end = idx_start + num_valid
		for i in range(idx_start, idx_end):
			idx = self.indices[i]
			valid_triples.append(self.triples[idx])
		return valid_triples

def load_triple(infile, threshold=2.0):
	triples = []
	with open(infile, 'r') as fin:
		for line in fin.readlines():
			fields = line.strip().split('\t')
			user = int(fields[0])
			item = int(fields[1])
			rate = float(fields[2]) - threshold
			if rate < 1.e-8:
				continue
			triple = user, item, rate
			triples.append(triple)
	return triples

def remove_triple(in_triples, threshold_user=0, threshold_item=0):
	user_count = {}
	item_count = {}
	for user, item, _ in in_triples:
		if user not in user_count:
			user_count[user] = 0
		user_count[user] += 1
		if item not in item_count:
			item_count[item] = 0
		item_count[item] += 1
	triples = []
	for user, item, rate in in_triples:
		if user_count[user] < threshold_user:
			continue
		if item_count[item] < threshold_item:
			continue
		triples.append((user, item, rate))
	num_removal = len(in_triples) - len(triples)
	return triples, num_removal

def zero_index(triples, position=0):
	elem_index = {}
	num_elem = 0
	for triple in triples:
		elem = triple[position]
		if elem in elem_index:
			continue
		elem_index[elem] = num_elem
		num_elem += 1
	return elem_index, num_elem

def reindex_triple(in_triples, user_index, item_index):
	rt_triples = []
	for user, item, rate in in_triples:
		triple = user_index[user], item_index[item], rate
		rt_triples.append(triple)
	return rt_triples

def group_by_user(triples):
	user_triples = {}
	for user, item, rate in triples:
		if user not in user_triples:
			user_triples[user] = UserTriple(user)
		user_triples[user].append((user, item, rate))
	return user_triples

def save_index(outfile, elem_index):
	sorted_index = sorted(elem_index.items(), key=operator.itemgetter(1))
	fout = open(outfile, 'w')
	for key, value in sorted_index:
		fout.write('{0}\t{1}\n'.format(value, key))
	fout.close()

def save_triple(fout, triples):
	triples = sorted(triples, key=operator.itemgetter(0, 1))
	for user, item, rate in triples:
		fout.write('{0}\t{1}\t{2:.1f}\n'.format(user, item, rate))

if __name__ == '__main__':
	data_dir =  utils.DATA_DIR_MOVIELENS_100K
	rate_infile = path.join(data_dir, 'ml-100k/u.data')
	item_infile = path.join(data_dir, 'ml-100k/u.item')
	user_outfile = path.join(data_dir, 'users.csv')
	item_outfile = path.join(data_dir, 'items.csv')
	feat_outfile = path.join(data_dir, 'feats.csv')

	triples = load_triple(rate_infile)
	print('{} triples before removal'.format(len(triples)))
	triples, num_removal = remove_triple(triples, threshold_item=utils.THRESHOLD_ITEM)
	triples, num_removal = remove_triple(triples, threshold_user=utils.THRESHOLD_USER)
	num_triple = len(triples)
	print('{} triples after removal'.format(num_triple))

	user_index, num_user = zero_index(triples)
	item_index, num_item = zero_index(triples, position=1)
	print('{} users and {} items in total'.format(num_user, num_item))
	save_index(user_outfile, user_index)
	save_index(item_outfile, item_index)

	triples = reindex_triple(triples, user_index, item_index)
	with open(path.join(data_dir, 'rates.csv'), 'w') as fout:
		save_triple(fout, triples)
	user_triples = group_by_user(triples)
	print('{} users after grouping triples'.format(len(user_triples)))

	with open(item_infile) as fin, \
			open(feat_outfile, 'w') as fout:
		for line in fin.readlines():
			fields = line.strip().split('|')
			item = int(fields[0])
			for i in range(-utils.NUM_GENRE, 0):
				indicator = int(fields[i])
				if indicator == 0:
					continue
				if item not in item_index:
					continue
				fout.write('{}\t{}\t1.0\n'.format(item_index[item], -i - 1))

	for user_triple in user_triples.values():
		user_triple.randomize()
		user_triple.gen_cv_index()

	for fold in range(utils.NUM_FOLD):
		train_filepath = path.join(data_dir, 'train%d.csv' % fold)
		test_filepath = path.join(data_dir, 'test%d.csv' % fold)
		valid_filepath = path.join(data_dir, 'valid%d.csv' % fold)
		ftrain = open(train_filepath, 'w')
		ftest = open(test_filepath, 'w')
		fvalid = open(valid_filepath, 'w')
		print('train={} test={} valid={}'.format(train_filepath, test_filepath, valid_filepath))
		for user_triple in user_triples.values():
			train_triples = user_triple.get_train_triple(fold)
			save_triple(ftrain, train_triples)
			test_triples = user_triple.get_test_triple(fold)
			save_triple(ftest, test_triples)
			valid_triples = user_triple.get_valid_triple(fold)
			save_triple(fvalid, valid_triples)
		fvalid.close()
		ftest.close()
		ftrain.close()
