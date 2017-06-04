import matplotlib

import tensorflow as tf
import numpy as np
import logging
import cPickle
import os, shutil, random

from config import Config
from tfrecord_reader import get_data
from LTRModel import LTRModel

matplotlib.use('Agg')
#np.set_printoptions(threshold=np.nan)

import pylab

flags = tf.flags

flags.DEFINE_string("config_name", "default", "Config to be used.")
flags.DEFINE_boolean("load_model", False, "Whether load the best model.")

FLAGS = flags.FLAGS

logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.DEBUG,
                    format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s")
logger.addHandler(logging.FileHandler(Config(FLAGS.config_name).log_dir+'log.txt'))

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

def train():
    training_config = Config(FLAGS.config_name)
    training_config.is_training = True
    training_config.uif_path = training_config.train_uif_path
    training_config.iur_path = training_config.train_iur_path
    training_config.record_path = training_config.train_record_path
    train_md = LTRModel(training_config)
    train_md.add_variables(reuse=False)

    valid_config = Config(FLAGS.config_name)
    valid_config.is_training = False
    valid_config.uif_path = valid_config.valid_uif_path
    valid_config.iur_path = valid_config.valid_iur_path
    valid_config.record_path = valid_config.valid_record_path
    valid_md = LTRModel(valid_config)
    valid_md.add_variables(reuse=True)

    # tts means Training input TenSor
    len_tts, user_tts, item_list_tts = get_data(filename=training_config.record_path)
    train_md.build_model(len_tts, user_tts, item_list_tts)

    # vts means Training input TenSor
    len_vts, user_vts, item_list_vts = get_data(filename=valid_config.record_path)
    valid_md.build_model(len_vts, user_vts, item_list_vts)
    
    _patience = training_config.patience
    best_valid_loss = 10000000.0
    best_valid_acc = 0.0
    valid_history = []
    train_history = []

    with tf.Session() as sess:
        if FLAGS.load_model:
            tf.train.Saver().restore(sess, training_config.bestmodel_dir)
        else:
            tf.global_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        batch_train_history = []
        for _step in xrange(training_config.max_step):
            if _patience == 0:
                break
            loss_ts = train_md.sum_loss
            train_op = train_md.train_op
            train_loss, _ = sess.run([loss_ts, train_op])
            batch_train_history.append(train_loss)
            
            if _step % training_config.train_freq == 0:
                logger.debug("Step: %d Training: Loss: %f" % (_step, train_loss))

            if _step != 0 and _step % valid_config.valid_freq == 0:
                logger.debug("Start validation")
                valid_losses = []

                for i in xrange(valid_config.valid_data_size):
                    valid_loss, valid_pred = \
                        sess.run([valid_md.sum_loss, valid_md.pred])
                    valid_losses.append(valid_loss)
                    #logger.debug("Validation loss %f" % valid_loss)

                mean_loss = np.mean(valid_losses)
                logger.debug('Validation: finish')
                
                logger.debug('Step: %d Validation: mean loss: %f' % (_step, mean_loss))

                valid_history.append([_step, mean_loss])
                batch_train_loss = np.mean(batch_train_history)
                logger.debug('Step: %d Training batch loss: %f' % (_step, batch_train_loss))
                batch_train_history = []
                train_history.append([_step, batch_train_loss])
                report_figure(valid_history, train_history, training_config.fig_path)
                logger.debug('Figure saved')

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
        coord.request_stop()
        coord.join(threads)
    
if __name__=='__main__':
    train()
