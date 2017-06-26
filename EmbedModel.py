import tensorflow as tf
import numpy as np

class EmbedModel():
    def __init__(self, conf):
        self.conf = conf
        self.uif_mat = np.load(conf.uif_path)
        self.embed_user_mat = np.load("%sembed_user.npy" % conf.embedPath)
        self.embed_item_mat = np.load("%sembed_item.npy" % conf.embedPath)
        self.context_user_mat = np.load("%scontext_user.npy" % conf.embedPath)
        self.context_item_mat = np.load("%scontext_item.npy" % conf.embedPath)
        self.feat_mat = np.load("%sfeat_mat.npy" % conf.featMatDir)

    def add_variables(self, reuse=False):
        conf = self.conf
        with tf.variable_scope('Fixed', reuse=reuse):
            self.uif = tf.get_variable('uif',
                                       [conf.user_size,
                                        conf.item_size,
                                        len(conf.recAlgos)],
                                       initializer=tf.constant_initializer(self.uif_mat),
                                       trainable=False)
            self.embed_user = tf.get_variable('embed_user',
                                              [conf.user_size,
                                               conf.embed_size],
                                              initializer=tf.constant_initializer(self.embed_user_mat),
                                              trainable=False)
            self.embed_item = tf.get_variable('embed_item',
                                              [conf.item_size,
                                               conf.embed_size],
                                              initializer=tf.constant_initializer(self.embed_item_mat),
                                              trainable=False)
            self.context_user = tf.get_variable('context_user',
                                                [conf.user_size,
                                                 conf.embed_size],
                                                initializer=tf.constant_initializer(self.context_user_mat),
                                                trainable=False)
            self.context_item = tf.get_variable('context_item',
                                                [conf.item_size,
                                                 conf.embed_size],
                                                initializer=tf.constant_initializer(self.context_item_mat),
                                                trainable=False)
            self.feat_embed = tf.get_variable('feat',
                                                [conf.item_size,
                                                 conf.feat_size],
                                                initializer=tf.constant_initializer(self.feat_mat),
                                                trainable=False)
            if self.conf.drop_embed:
                self.embed_user = tf.contrib.layers.dropout(self.embed_user, self.conf.keep_prob, is_training=self.conf.is_training)
                self.embed_item = tf.contrib.layers.dropout(self.embed_item, self.conf.keep_prob, is_training=self.conf.is_training)
                self.context_user = tf.contrib.layers.dropout(self.context_user, self.conf.keep_prob, is_training=self.conf.is_training)
                self.context_item = tf.contrib.layers.dropout(self.context_item, self.conf.keep_prob, is_training=self.conf.is_training)
            
            self.item_joint_embed = tf.concat([self.embed_item, self.context_item], 1)
            self.user_joint_embed = tf.concat([self.embed_user, self.context_user], 1)
            self.item_feat_joint_embed = tf.concat([self.item_joint_embed, self.feat_embed], 1)
            
        with tf.variable_scope('Weights', reuse=reuse):
            self.v1 = tf.get_variable('v1',
                                      [len(conf.recAlgos), 1])
            self.v2 = tf.get_variable('v2',
                                      [conf.z_size, 1])
            self.W_z = tf.get_variable('W_z',
                                       [conf.z_size,
                                        2*conf.embed_size+conf.feat_size,
                                        2*conf.embed_size+conf.feat_size])
            self.W_rel = tf.get_variable('W_rel',
                                         [2*conf.embed_size,
                                          2*conf.embed_size])

        self.ph_selected_items = tf.placeholder(tf.int32, shape=(None,))
        self.ph_all_items = tf.placeholder(tf.int32, shape=(None,))
        self.ph_groundtruth = tf.placeholder(tf.int32, shape=[])
        self.ph_user = tf.placeholder(tf.int32, shape=[])

    def build_model(self):
        uif_u = self.uif[self.ph_user]
        if self.conf.drop_matrix:
            uif_u = tf.contrib.layers.dropout(uif_u, self.conf.keep_prob, is_training=self.conf.is_training) # Add dropout layer
        rel_score1 = tf.matmul(uif_u, self.v1)
        user_embed_u = tf.expand_dims(tf.nn.embedding_lookup(self.user_joint_embed, self.ph_user), 1)
        rel_score2 = tf.matmul(tf.matmul(self.item_joint_embed, self.W_rel), user_embed_u)
        rel_score = rel_score1 + rel_score2

        def fn_i0(): # (choices, score_sum) when i = 0
            return (self.ph_all_items, tf.squeeze(rel_score))
        def fn_not_i0():  # (choices, score_sum) when i != 0
            selected_items = self.ph_selected_items
            iur = self.item_feat_joint_embed
            if self.conf.drop_matrix:
                iur = tf.contrib.layers.dropout(iur, self.conf.keep_prob, is_training=self.conf.is_training) # Add dropout layer
            se = tf.nn.embedding_lookup(iur, selected_items)
            se = tf.transpose(se)
            # see test/einsum_test.py
            iur_w = tf.einsum('nu,zud->znd', iur, self.W_z)
            iur_w_se = tf.einsum('znu,uk->znk', iur_w, se)
            mp_iur_w_se = tf.reduce_max(iur_w_se, axis=2) # z x n
            mp_iur_w_se = tf.transpose(mp_iur_w_se) # n x z
            mp_iur_w_se = tf.tanh(mp_iur_w_se)
            div_score = tf.matmul(mp_iur_w_se, self.v2) # n x 1
            score_sum = tf.squeeze(rel_score + div_score) # vec of n
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
