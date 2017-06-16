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
                                        conf.user_size+conf.feat_size],
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
                                         [conf.user_size+conf.feat_size, conf.embed_size])

        self.ph_selected_items = tf.placeholder(tf.int32, shape=(None,))
        self.ph_all_items = tf.placeholder(tf.int32, shape=(None,))
        self.ph_groundtruth = tf.placeholder(tf.int32, shape=[])
        self.ph_user = tf.placeholder(tf.int32, shape=[])

    def build_model(self):
        uif_u = self.uif[self.ph_user]
        score1 = tf.matmul(uif_u, self.v1)

        def fn_i0(): # (choices, score_sum) when i = 0
            return (self.ph_all_items, tf.squeeze(score1))
        def fn_not_i0():  # (choices, score_sum) when i != 0
            selected_items = self.ph_selected_items
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
            choices = tf.reshape(tf.sparse_tensor_to_dense(tf.sets.set_difference([self.ph_all_items], [selected_items])), [-1]) # vec of remaining choices
            return (choices, score_sum)

        i = tf.shape(self.ph_selected_items)[0]
        choices, score_sum = tf.cond(tf.equal(i, 0),
                                     lambda: fn_i0(),
                                     lambda: fn_not_i0())
            
        eff_score = tf.gather(score_sum, choices, validate_indices=False) # vec of choices
        _argmax = tf.argmax(eff_score, axis=0)
        _pred = tf.gather(choices, _argmax, validate_indices=False)
        _loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=score_sum, labels=self.ph_groundtruth)

        self.loss = _loss
        self.pred = _pred
        self.loss_summary = tf.summary.scalar('Loss', self.loss)
        if self.conf.is_training:
            self.train_op = tf.train.AdamOptimizer(self.conf.lr).minimize(self.loss)
