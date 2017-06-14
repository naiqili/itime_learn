import tensorflow as tf
import numpy as np

class SimpleEmbedModel():
    def __init__(self, conf):
        self.conf = conf
        self.uif_mat = np.load(conf.uif_path)
        self.iur_mat = np.load(conf.iur_path)

    def add_variables(self, reuse=False):
        conf = self.conf
        with tf.variable_scope('Fixed', reuse=reuse):
            self.uif = tf.get_variable('uif',
                                       [conf.user_size,
                                        conf.item_size,
                                        len(conf.recAlgos)],
                                       initializer=tf.constant_initializer(self.uif_mat),
                                       trainable=False)
            self.iur = tf.get_variable('iur',
                                       [conf.item_size,
                                        conf.user_size],
                                       initializer=tf.constant_initializer(self.iur_mat),
                                       trainable=False)
        with tf.variable_scope('Weights', reuse=reuse):
            self.v1 = tf.get_variable('v1',
                                      [len(conf.recAlgos), 1])
            self.v2 = tf.get_variable('v2',
                                      [conf.z_size, 1])
            self.W_z = tf.get_variable('W_z',
                                       [conf.z_size,
                                        conf.embed_size,
                                        conf.embed_size])
        with tf.variable_scope('Embeddings', reuse=reuse):
            self.embed = tf.get_variable('embed',
                                         [conf.user_size, conf.embed_size])

    def build_model(self, len_ts, user_ts, item_list_ts):
        uif_u = self.uif[user_ts]
        score1 = tf.matmul(uif_u, self.v1)

        def loop_body(i, item_list, selected_items_arr, score1, all_losses):
            def fn_i0(): # (choices, score_sum) when i = 0
                return (item_list, tf.squeeze(score1))
            def fn_not_i0():  # (choices, score_sum) when i != 0
                if self.conf.is_training:
                    selected_items = tf.slice(item_list, [0], [i])
                else:
                    selected_items = selected_items_arr.stack() # vec of len(selected)
                    selected_items = tf.reshape(selected_items, [i])
                iur = self.iur
                iur_embed = tf.matmul(iur, self.embed)
                se = tf.nn.embedding_lookup(iur, selected_items)
                se_embed = tf.matmul(se, self.embed)
                se_embed = tf.transpose(se_embed)                
                # see test/einsum_test.py
                iur_w = tf.einsum('nu,zud->znd', iur_embed, self.W_z)
                iur_w_se = tf.einsum('znu,uk->znk', iur_w, se_embed)
                mp_iur_w_se = tf.reduce_max(iur_w_se, axis=2) # z x n
                mp_iur_w_se = tf.transpose(mp_iur_w_se) # n x z
                score2 = tf.matmul(mp_iur_w_se, self.v2) # n x 1
                score_sum = tf.squeeze(score1 + score2) # vec of n
                choices = tf.reshape(tf.sparse_tensor_to_dense(tf.sets.set_difference([item_list], [selected_items])), [-1]) # vec of remaining choices
                return (choices, score_sum)

            choices, score_sum = tf.cond(tf.equal(i, 0),
                                         lambda: fn_i0(),
                                         lambda: fn_not_i0())
            
            eff_score = tf.gather(score_sum, choices, validate_indices=False) # vec of choices
            _argmax = tf.argmax(eff_score, axis=0)
            _pred = tf.gather(choices, _argmax, validate_indices=False)
            _loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=score_sum, labels=item_list[i])
            all_losses = all_losses.write(i, _loss) # vec of n
            selected_items_arr = selected_items_arr.write(i, _pred)
            i = tf.add(i, 1)
            return i, item_list, selected_items_arr, score1, all_losses

        loop_cond = lambda i, item_list, selected, score1, all_scores: \
            tf.less(i, len_ts)

        all_losses = tf.TensorArray(tf.float32, size=0, dynamic_size=True, \
                                    clear_after_read=True, infer_shape=True)
        selected_items_arr = tf.TensorArray(tf.int32, size=0, dynamic_size=True, \
                                                clear_after_read=False, infer_shape=True)
        _, _,  selected_items_arr, _, all_losses = tf.while_loop(loop_cond, loop_body, \
                                                [0, item_list_ts, selected_items_arr, score1, all_losses])

        self.all_losses = all_losses.stack() # vec of n
        self.reset1 = all_losses.close()
        self.reset2 = selected_items_arr.close()
        #self.sum_loss = tf.reduce_sum(self.all_losses)
        self.sum_loss = tf.reduce_mean(self.all_losses)
        self.pred = selected_items_arr.stack()
        self.loss_summary = tf.summary.scalar('Sum_Loss', self.sum_loss)
        if self.conf.is_training:
            self.train_op = tf.train.AdamOptimizer(self.conf.lr).minimize(self.sum_loss)
