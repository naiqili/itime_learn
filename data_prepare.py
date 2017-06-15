from config import Config
import numpy as np
import os
import tensorflow as tf

test_flag = False

if test_flag:
    conf = Config('test')
else:
    conf = Config()

# Build user-item-feature data
def build_uif(test_train, n_fold):
    uif = np.zeros((conf.user_size,
                    conf.item_size,
                    len(conf.recAlgos)), dtype=np.float16)
    for (recInd, recAlgo) in enumerate(conf.recAlgos):
        input_path = "%s%s_FalseFilter%s_%d.csv" % \
                     (conf.recPath, recAlgo, test_train, n_fold)
        with open(input_path) as f_in:
            for line in f_in:
                user, item, score = line.strip().split()
                user = int(user)
                item = int(item)
                if score == "NaN":
                    score = 0
                    #print recAlgo, user, item, score
                else:
                    score = float(score)
                if test_flag and (user >= conf.user_size or item >= conf.item_size):
                    continue
                uif[user, item, recInd] = score
    output_path = "%suif_%s_%d" % (conf.uifDir, test_train, n_fold)
    np.save(output_path, uif)

train_data_size = {}
valid_data_size = {}
train_data_user_size = {}
valid_data_user_size = {}
# Build item-user-rating data
def build_iur(test_train, n_fold):
    iur = np.zeros((conf.item_size,
                    conf.user_size), dtype=np.float16)
    input_path = "%s%s_%d.csv" % \
                 (conf.seed2048Path, test_train, n_fold)
    output_path = "%siur_%s_%d" % (conf.iurDir, test_train, n_fold)
    with open(input_path) as f_in:
        for line in f_in:
            user, item, rating = line.strip().split()
            user = int(user)
            item = int(item)
            rating = float(rating)
            if test_flag and (user >= conf.user_size or item >= conf.item_size):
                continue
            iur[item, user] = rating
    np.save(output_path, iur)
    

if not os.path.exists(conf.uifDir):
    os.makedirs(conf.uifDir)
if not os.path.exists(conf.iurDir):
    os.makedirs(conf.iurDir)
    
for cv in range(conf.n_folds):
    print "building training uif data for %d-th fold..." % cv
    build_uif("train", cv)
    print "building test uif data for %d-th fold..." % cv
    build_uif("test", cv)

    print "building training iur data for %d-th fold..." % cv
    build_iur("train", cv)
    print "building test iur data for %d-th fold..." % cv
    build_iur("test", cv)

if not os.path.exists(conf.datasizeDir):
    os.makedirs(conf.datasizeDir)
print 'Saving data size info...'

train_ds_path = '%strain_size.txt' % conf.datasizeDir
with open(train_ds_path, 'w') as f:
    for (cv, n) in train_data_size.items():
        f.write("%d %d\n" % (cv, n))

valid_ds_path = '%svalid_size.txt' % conf.datasizeDir
with open(valid_ds_path, 'w') as f:
    for (cv, n) in valid_data_size.items():
        f.write("%d %d\n" % (cv, n))
        
# build TFRecords

if not os.path.exists(conf.tfrecordDir):
    os.makedirs(conf.tfrecordDir)

def build_tfrecord(test_train, n_fold):
    input_path = "%spersonal-ndcg_%s%d.csv" % \
                 (conf.groundTruthPath, test_train, n_fold)
    output_path = "%s%s_%d.record" % \
                 (conf.tfrecordDir, test_train, n_fold)
    writer = tf.python_io.TFRecordWriter(output_path)
    data = {}
    cnt = 0
    with open(input_path) as f_in:
        for line in f_in:
            user, item, _ = line.strip().split()
            user = int(user)
            item = int(item)
            if test_flag and (user >= conf.user_size or item >= conf.item_size):
                continue
            if not user in data:
                data[user] = [item]
            else:
                data[user].append(item)
            cnt += 1
    for (user, item_list) in data.items():
        example = tf.train.Example(
            features = tf.train.Features(
                feature = {
                    'len': tf.train.Feature(int64_list=tf.train.Int64List(value=[len(item_list)])),
                    'user': tf.train.Feature(int64_list=tf.train.Int64List(value=[user])),
                    'item_list': tf.train.Feature(int64_list=tf.train.Int64List(value=item_list))
                    }
                )
            )
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()
    if test_train == 'test':
        valid_data_user_size[n_fold] = len(data)
        valid_data_size[n_fold] = cnt
    else:
        train_data_user_size[n_fold] = len(data)
        train_data_size[n_fold] = cnt

for cv in range(conf.n_folds):
    print "building training tfrecord for %d-th fold..." % cv
    build_tfrecord("train", cv)
    print "building test tfrecord for %d-th fold..." % cv
    build_tfrecord("test", cv)

if not os.path.exists(conf.datasizeDir):
    os.makedirs(conf.datasizeDir)
print 'Saving data size info...'

train_ds_path = '%strain_size.txt' % conf.datasizeDir
with open(train_ds_path, 'w') as f:
    for (cv, n) in train_data_size.items():
        f.write("%d %d\n" % (cv, n))

valid_ds_path = '%svalid_size.txt' % conf.datasizeDir
with open(valid_ds_path, 'w') as f:
    for (cv, n) in valid_data_size.items():
        f.write("%d %d\n" % (cv, n))

train_user_ds_path = '%strain_user_size.txt' % conf.datasizeDir
with open(train_user_ds_path, 'w') as f:
    for (cv, n) in train_data_user_size.items():
        f.write("%d %d\n" % (cv, n))

valid_user_ds_path = '%svalid_user_size.txt' % conf.datasizeDir
with open(valid_user_ds_path, 'w') as f:
    for (cv, n) in valid_data_user_size.items():
        f.write("%d %d\n" % (cv, n))
        
print "finish"
