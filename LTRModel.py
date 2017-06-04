import tensorflow as tf
import numpy as np

class LTRModel():
    def __init__(self, conf):
        self.conf = conf
        self.uif_mat = np.load(conf.uif_path)
        self.iur_mat = np.load(conf.iur_path)

    def add_variables(self, reuse=False):
        conf = self.conf
        with tf.variable_scope('Fixed', reuse=reuse):
            self.uif = tf.get_variable('uif',
                                       [conf.user_size+1,
                                        conf.item_size+1,
                                        len(conf.recAlgos)],
                                       initializer=tf.constant_initializer(self.uif_mat),
                                       trainable=False)
            self.iur = tf.get_variable('iur',
                                       [conf.item_size+1,
                                        conf.user_size+1],
                                       initializer=tf.constant_initializer(self.iur_mat),
                                       trainable=False)
        with tf.variable_scope('Weights', reuse=reuse):
            self.v1 = tf.get_variable('v1',
                                      [len(conf.recAlgos), 1])
            self.v2 = tf.get_variable('v2',
                                      [conf.z_size, 1])
            self.W_z = tf.get_variable('W_z',
                                       [conf.z_size,
                                        conf.user_size+1,
                                        conf.user_size+1])

    def build_model(self, len_ts, user_ts, item_list_ts):
        uif_u = self.uif[user_ts]
        score1 = tf.matmul(uif_u, self.v1)

        all_scores = tf.TensorArray(tf.float32, size=0, dynamic_size=True, \
            clear_after_read=False, infer_shape=False)
        mask = tf.sparse_to_dense(item_list_ts, [self.conf.item_size+1], 1.0, validate_indices=False) # vec of n
        all_scores = all_scores.write(0, tf.squeeze(score1) * mask) # vec of n

        def train_loop_body(i, item_list, score1, all_scores):
            selected_items = tf.slice(item_list, [0], [i])
            iur = self.iur
            se = tf.nn.embedding_lookup(iur, selected_items)
            se = tf.transpose(se)
            # see test/einsum_test.py
            iur_w = tf.einsum('nu,zud->znd', iur, self.W_z)
            iur_w_se = tf.einsum('znu,uk->znk', iur_w, se)
            mp_iur_w_se = tf.reduce_max(iur_w_se, axis=2) # z x n
            mp_iur_w_se = tf.transpose(mp_iur_w_se) # n x z
            score2 = tf.matmul(mp_iur_w_se, self.v2) # n x 1
            score_sum = tf.squeeze(score1 + score2) # vec of n
            mask = tf.sparse_to_dense(item_list, [self.conf.item_size+1], 1.0, validate_indices=False) # vec of n
            mask_score = mask * score_sum
            all_scores = all_scores.write(i, mask_score) # vec of n
            i = tf.add(i, 1)
            return i, item_list, score1, all_scores

        def test_loop_body(i, item_list, selected_items_arr, score1, all_scores):
            selected_items = selected_items_arr.stack() # vec of len(selected)
            iur = self.iur
            se = tf.nn.embedding_lookup(iur, selected_items)
            se = tf.transpose(se)
            # see test/einsum_test.py
            iur_w = tf.einsum('nu,zud->znd', iur, self.W_z)
            iur_w_se = tf.einsum('znu,uk->znk', iur_w, se)
            mp_iur_w_se = tf.reduce_max(iur_w_se, axis=2) # z x n
            mp_iur_w_se = tf.transpose(mp_iur_w_se) # n x z
            score2 = tf.matmul(mp_iur_w_se, self.v2) # n x 1
            score_sum = tf.squeeze(score1 + score2) # vec of n
            mask = tf.sparse_to_dense(item_list, [self.conf.item_size+1], 1.0, validate_indices=False) # vec of n
            mask_score = mask * score_sum
            all_scores = all_scores.write(i, mask_score) # vec of n
            _pred = tf.to_int32(tf.argmax(mask_score, axis=0))
            selected_items_arr = selected_items_arr.write(i, _pred)
            i = tf.add(i, 1)
            return i, item_list, selected_items_arr, score1, all_scores

        train_loop_cond = lambda i, item_list, score1, all_scores: \
            tf.less(i, len_ts)

        test_loop_cond = lambda i, item_list, selected, score1, all_scores: \
            tf.less(i, len_ts)

        if self.conf.is_training:
            _, _, _, all_scores = tf.while_loop(train_loop_cond, train_loop_body, \
                                                [1, item_list_ts, score1, all_scores])
        else:
            selected_items_arr = tf.TensorArray(tf.int32, size=0, dynamic_size=True, \
                                                clear_after_read=False, infer_shape=True)
            pred_0 = tf.to_int32(tf.argmax(tf.squeeze(score1), axis=0))
            selected_items_arr = selected_items_arr.write(0, pred_0)
            _, _,  _, _, all_scores = tf.while_loop(test_loop_cond, test_loop_body, \
                                                    [1, item_list_ts, selected_items_arr, score1, all_scores])

        all_scores_ts = all_scores.stack() # len x n, len is len(item_list)
        self.logits = all_scores_ts
        self.pred = tf.squeeze(tf.argmax(self.logits, 1))
        self.all_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=self.logits, labels=item_list_ts)
        self.sum_loss = tf.reduce_sum(self.all_losses)
        self.loss_summary = tf.summary.scalar('Sum_Loss', self.sum_loss)
        if self.conf.is_training:
            self.train_op = tf.train.AdamOptimizer(self.conf.lr).minimize(self.sum_loss)
