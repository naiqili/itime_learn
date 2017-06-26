import matplotlib

import tensorflow as tf
import numpy as np
import logging
import cPickle
import os, shutil, random, datetime

from config import Config
from tfrecord_reader import get_data, get_one_data
from EmbedModel import EmbedModel
from evaluation_measure import *

matplotlib.use('Agg')
#np.set_printoptions(threshold=np.nan)

import pylab

flags = tf.flags

flags.DEFINE_string("config_name", "default", "Config to be used.")
flags.DEFINE_string("model_name", "Embed", "Model name.")
flags.DEFINE_boolean("load_model", False, "Whether load the best model.")
flags.DEFINE_integer("cv", 0, "Which cross validation set to be used.")
flags.DEFINE_integer("z_size", 5, "Z size.")
flags.DEFINE_integer("embed_size", 10, "Embedding size.")
flags.DEFINE_float("keep_prob", 0.5, "Dropout rate.")

FLAGS = flags.FLAGS

logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.DEBUG,
                    format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s")

def get_dir(cv, z_size, embed_size, cur_time, keep_prob):
    time_str = cur_time.strftime('%b-%d-%y %H:%M:%S')
    _dir = "%s%s_cv%d_z%d_drop%f_%s/" % (Config(FLAGS.config_name).log_dir, FLAGS.model_name, cv, z_size, keep_prob, time_str)
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    return _dir

def report_figure(valid_history, train_history, fig_path):
    train_x, train_y = zip(*train_history)
    valid_x, valid_y = zip(*valid_history)
    try:
        pylab.figure()
        pylab.title("Loss")
        pylab.plot(train_x, train_y, 'r', label='train loss')
        pylab.plot(valid_x, valid_y, 'b', label='valid loss')
        pylab.legend()
        pylab.savefig(fig_path + 'figure.png')
        pylab.close()
    except:
        pass

def train(cv, z_size, embed_size, keep_prob):
    cur_time = datetime.datetime.now()
    logger.addHandler(logging.FileHandler(get_dir(cv, z_size, embed_size, cur_time, keep_prob) + 'log.txt'))
    
    logger.debug("Start training for the %d-th fold..." % cv)
    training_config = Config(FLAGS.config_name)
    training_config.is_training = True
    training_config.z_size = z_size # overwrite
    training_config.embed_size = embed_size
    training_config.keep_prob = keep_prob
    training_config.uif_path = "%suif_train_%d.npy" % (training_config.uifDir, cv)
    training_config.record_path = "%strain_%d.record" % (training_config.tfrecordDir, cv)
    train_md = EmbedModel(training_config)
    train_md.add_variables(reuse=False)

    valid_config = Config(FLAGS.config_name)
    valid_config.is_training = False
    valid_config.z_size = z_size # overwrite
    valid_config.embed_size = embed_size
    valid_config.keep_prob = 1.0
    valid_config.uif_path = "%suif_train_%d.npy" % (valid_config.uifDir, cv)
    valid_config.record_path = "%stest_%d.record" % (valid_config.tfrecordDir, cv)
    valid_md = EmbedModel(valid_config)
    valid_md.add_variables(reuse=True)

    # tts means Training input TenSor
    len_tts, user_tts, item_list_tts = get_data(filename=training_config.record_path)
    train_md.build_model()

    # vts means Training input TenSor
    len_vts, user_vts, item_list_vts = get_data(filename=valid_config.record_path)
    valid_md.build_model()

    def predict(suffix_str):
        # start making prediction
        logger.debug('start making prediction...')
        logger.debug('suffix: %s' % suffix_str)
        results = {}
        for i in xrange(valid_config.valid_data_user_size[cv]):
            _len, _user, _all_items = sess.run([len_vts, user_vts, item_list_vts])
            _selected_items = []
            results[_user] = []
            for k in range(_len):
                feed_dict = {valid_md.ph_user: _user, \
                             valid_md.ph_selected_items: _selected_items, \
                             valid_md.ph_all_items: _all_items}
                valid_pred = \
                        sess.run(valid_md.pred, feed_dict=feed_dict)
                results[_user].append(valid_pred)
                _selected_items.append(valid_pred)
            #logger.debug('user_id: %d' % _user)                
            #logger.debug('items: %s' % str(results[_user]))

        # Save to file
        logger.debug("start making predictions for %d-th fold" % cv)
        pred_path = get_dir(cv, z_size, embed_size, cur_time, keep_prob) + 'pred/' 
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)
        with open(pred_path + ("cv%d_%s.pred" % (cv, suffix_str)), 'w') as f:
            for (user_id, item_list) in results.items():
                for i in range(len(item_list)):
                    f.write("%d\t%d\t1.0\n" % (user_id, item_list[i]))
        logger.debug("Predictions saved.")
    
    _patience = training_config.patience
    best_valid_loss = 10000000.0
    best_valid_acc = 0.0
    valid_history = []
    train_history = []

    mean_valid_loss_hd = tf.placeholder(tf.float32)

    with tf.Session() as sess:
        measure = EvaluationMeasure(sess, training_config.seed2048Path)
        logger.debug("computing normalizing constant... this may take some time...")
        measure.compute_normalizing_constant(training_config.seed2048Path + 'test%d.csv' % cv, training_config.ndcgConstantPath + 'p_ndcg_const%d' % cv, training_config.ndcgConstantPath + 'a_ndcg_const%d' % cv)
        logger.debug("done")
        if FLAGS.load_model:
            tf.train.Saver().restore(sess, training_config.bestmodel_dir)
        else:
            tf.global_variables_initializer().run()

        train_log_path = get_dir(cv, z_size, embed_size, cur_time, keep_prob) + 'train/'
        if not os.path.exists(train_log_path):
            os.makedirs(train_log_path)
        train_writer = tf.summary.FileWriter(train_log_path)

        valid_log_path = get_dir(cv, z_size, embed_size, cur_time,keep_prob) + 'valid/' 
        if not os.path.exists(valid_log_path):
            os.makedirs(valid_log_path)
        valid_writer = tf.summary.FileWriter(valid_log_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        train_one_data = get_one_data(sess, len_tts, user_tts, item_list_tts)
        valid_one_data = get_one_data(sess, len_vts, user_vts, item_list_vts)
        
        batch_train_history = []
        for _step in xrange(training_config.max_step):
            if _patience == 0:
                break
            (_user, _selected_items, _all_items, _gt) = \
                train_one_data.next()
            feed_dict = {train_md.ph_user: _user, \
                         train_md.ph_selected_items: _selected_items, \
                         train_md.ph_all_items: _all_items, \
                         train_md.ph_groundtruth: _gt}
            train_loss, train_summary, _ = sess.run([train_md.loss, train_md.loss_summary, train_md.train_op], feed_dict=feed_dict)
            train_history.append([_step, train_loss])
            train_writer.add_summary(train_summary, _step)
            train_writer.flush()
            batch_train_history.append(train_loss)
            
            if _step % training_config.train_freq == 0:
                logger.debug("Step: %d Training: Loss: %f" % (_step, train_loss))

            if _step != 0 and _step % valid_config.valid_freq == 0:
                logger.debug("Start validation")
                valid_losses = []

                for i in xrange(valid_config.valid_data_size[cv]):
                #for i in xrange(100):
                    (_user, _selected_items, _all_items, _gt) = \
                        train_one_data.next()
                    feed_dict = {valid_md.ph_user: _user, \
                                 valid_md.ph_selected_items: _selected_items, \
                                 valid_md.ph_all_items: _all_items, \
                                 valid_md.ph_groundtruth: _gt}
                    valid_loss, valid_pred = \
                        sess.run([valid_md.loss, valid_md.pred], feed_dict=feed_dict)
                    valid_losses.append(valid_loss)
                    #logger.debug("Validation loss %f" % valid_loss)

                mean_loss = np.mean(valid_losses)
                logger.debug('Validation: finish')
                
                logger.debug('Step: %d Validation: mean loss: %f' % (_step, mean_loss))

                valid_history.append([_step, mean_loss])
                batch_train_loss = np.mean(batch_train_history)
                logger.debug('Step: %d Training batch loss: %f' % (_step, batch_train_loss))
                batch_train_history = []
                
                report_figure(valid_history, train_history, get_dir(cv, z_size, embed_size, cur_time, keep_prob))
                logger.debug('Figure saved')

                valid_loss_summary = tf.Summary(value=[
                        tf.Summary.Value(tag="mean_valid_loss", simple_value=mean_loss)])
                
                valid_writer.add_summary(valid_loss_summary, _step)
                
                predict("step%d" % _step) # make prediction

                logger.debug("Start evaluation...")
                ltr_filepath = get_dir(cv, z_size, embed_size, cur_time, keep_prob) + 'pred/cv%d_step%d.pred' % (cv, _step)
                
                personal_ndcg, alpha_ndcg = measure.compute_normalized_score(ltr_filepath)
                logger.debug('personal ndcg: %.4f, alpha ndcg: %.4f' % (personal_ndcg, alpha_ndcg))
                pndcg_summary = tf.Summary(value=[
                        tf.Summary.Value(tag="personal_ndcg", simple_value=personal_ndcg)])
                andcg_summary = tf.Summary(value=[
                        tf.Summary.Value(tag="alpha_ndcg", simple_value=alpha_ndcg)])
                valid_writer.add_summary(pndcg_summary, _step)
                valid_writer.add_summary(andcg_summary, _step)
                valid_writer.flush()
                
                if mean_loss < best_valid_loss:
                    best_valid_loss = mean_loss
                    _patience = training_config.patience
                    saver = tf.train.Saver()
                    saver.save(sess, training_config.bestmodel_dir)
                    logger.debug('Better model saved')
                else:                    
                    _patience -= 1
                    logger.debug('Not improved. Patience: %d' % _patience)
                logger.debug('Best loss: %f' % (best_valid_loss))
        train_writer.close()
        valid_writer.close()
        
        coord.request_stop()
        coord.join(threads)
    
if __name__=='__main__':
    train(FLAGS.cv, FLAGS.z_size, FLAGS.embed_size, FLAGS.keep_prob)
