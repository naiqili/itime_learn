import tensorflow as tf
from config import Config

def get_data(filename):
    # first construct a queue containing a list of filenames.
    # this lets a user split up there dataset in multiple files to keep
    # size down
    filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=None)
    # Unlike the TFRecordWriter, the TFRecordReader is symbolic
    reader = tf.TFRecordReader()
    # One can read a single serialized example from a filename
    # serialized_example is a Tensor of type string.
    _, serialized_example = reader.read(filename_queue)
    # The serialized example is converted back to actual values.
    # One needs to describe the format of the objects to be returned
    features = tf.parse_single_example(
        serialized_example,
        features={
            'len': tf.VarLenFeature(tf.int64),
            'user': tf.VarLenFeature(tf.int64),
            'item_list': tf.VarLenFeature(tf.int64)
        })
    # now return the converted data
    _l = tf.squeeze(tf.to_int32(tf.sparse_tensor_to_dense(features['len'])))
    _user = tf.squeeze(tf.to_int32(tf.sparse_tensor_to_dense(features['user'])))
    _item_list = tf.to_int32(tf.sparse_tensor_to_dense(features['item_list']))
    return tf.train.limit_epochs([_l, _user, _item_list])

def get_one_data(sess, l_ts, user_ts, item_list_ts):
    while True:
        _l, _user, _item_list = sess.run([l_ts, user_ts, item_list_ts])
        for k in range(_l):
            user = _user
            selected_items = _item_list[:k]
            all_items = _item_list
            groundtruth = _item_list[k]
            yield (user, selected_items, all_items, groundtruth)

if __name__=='__main__':
    conf = Config('test')
    l, user, item_list = get_data('%stest_0.record' % conf.tfrecordDir)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        one_data = get_one_data(sess, l, user, item_list)
        for _step in range(50):
            print _step
            (u, s, a, g) = one_data.next()
            print u
            print s
            print a
            print g

        coord.request_stop()
        coord.join(threads)
