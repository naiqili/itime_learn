import matplotlib

import tensorflow as tf
import numpy as np
import logging
import cPickle
import os, shutil, random, datetime

from config import Config
from tfrecord_reader import get_data
from SimpleEmbedModel import SimpleEmbedModel

matplotlib.use('Agg')
#np.set_printoptions(threshold=np.nan)

import pylab

flags = tf.flags

flags.DEFINE_string("config_name", "default", "Config to be used.")
flags.DEFINE_string("model_name", "SimpleEmbed", "Model name.")
flags.DEFINE_boolean("load_model", False, "Whether load the best model.")
flags.DEFINE_integer("cv", 0, "Which cross validation set to be used.")
flags.DEFINE_integer("z_size", 5, "Z size.")
flags.DEFINE_integer("embed_size", 10, "Embedding size.")

FLAGS = flags.FLAGS

logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.DEBUG,
                    format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s")

def get_dir(cv, z_size, embed_size, cur_time):
    time_str = cur_time.strftime('%b-%d-%y %H:%M:%S')
    _dir = "%s%s_cv%d_z%d_%s/" % (Config(FLAGS.config_name).log_dir, FLAGS.model_name, cv, z_size, time_str)
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

def train(cv, z_size, embed_size):
    cur_time = datetime.datetime.now()
    logger.addHandler(logging.FileHandler(get_dir(cv, z_size, embed_size, cur_time) + 'log.txt'))
    
    logger.debug("Start training for the %d-th fold..." % cv)
    training_config = Config(FLAGS.config_name)
    training_config.is_training = True
    training_config.z_size = z_size # overwrite
    training_config.embed_size = embed_size
    training_config.uif_path = "%suif_train_%d.npy" % (training_config.uifDir, cv)
    training_config.iur_path = "%siur_train_%d.npy" % (training_config.iurDir, cv)
    training_config.record_path = "%strain_%d.record" % (training_config.tfrecordDir, cv)
    train_md = SimpleEmbedModel(training_config)
    train_md.add_variables(reuse=False)

    valid_config = Config(FLAGS.config_name)
    valid_config.is_training = False
    valid_config.z_size = z_size # overwrite
    valid_config.embed_size = embed_size
    valid_config.uif_path = "%suif_test_%d.npy" % (valid_config.uifDir, cv)
    valid_config.iur_path = "%siur_test_%d.npy" % (valid_config.iurDir, cv)
    valid_config.record_path = "%stest_%d.record" % (valid_config.tfrecordDir, cv)
    valid_md = SimpleEmbedModel(valid_config)
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

    mean_valid_loss_hd = tf.placeholder(tf.float32)

    with tf.Session() as sess:
        if FLAGS.load_model:
            tf.train.Saver().restore(sess, training_config.bestmodel_dir)
        else:
            tf.global_variables_initializer().run()

        train_log_path = get_dir(cv, z_size, embed_size, cur_time) + 'train/'
        if not os.path.exists(train_log_path):
            os.makedirs(train_log_path)
        train_writer = tf.summary.FileWriter(train_log_path)

        valid_log_path = get_dir(cv, z_size, embed_size, cur_time) + 'valid/' 
        if not os.path.exists(valid_log_path):
            os.makedirs(valid_log_path)
        valid_writer = tf.summary.FileWriter(valid_log_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        batch_train_history = []
        for _step in xrange(training_config.max_step):
            if _patience == 0:
                break
            loss_ts = train_md.sum_loss
            train_op = train_md.train_op
            train_loss, train_summary, _ = sess.run([loss_ts, train_md.loss_summary, train_op])
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
                    valid_loss, valid_pred = \
                        sess.run([valid_md.sum_loss, valid_md.pred])
                    valid_losses.append(valid_loss)
                    logger.debug("Validation loss %f" % valid_loss)

                mean_loss = np.mean(valid_losses)
                logger.debug('Validation: finish')
                
                logger.debug('Step: %d Validation: mean loss: %f' % (_step, mean_loss))

                valid_history.append([_step, mean_loss])
                batch_train_loss = np.mean(batch_train_history)
                logger.debug('Step: %d Training batch loss: %f' % (_step, batch_train_loss))
                batch_train_history = []
                
                report_figure(valid_history, train_history, get_dir(cv, z_size, embed_size, cur_time))
                logger.debug('Figure saved')

                summary = tf.Summary(value=[
                        tf.Summary.Value(tag="mean_valid_loss", simple_value=mean_loss),
                    ])
                
                valid_writer.add_summary(summary, _step)
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
        
        # start making prediction
        results = {}
        for i in xrange(valid_config.valid_data_size[cv]):
            user_id, valid_pred = sess.run([user_vts, valid_md.pred])
            results[user_id] = valid_pred
            logger.debug('user_id: %d' % user_id)                
            logger.debug('items: %s' % str(valid_pred))

        # Save to file
        logger.debug("start making predictions for %d-th fold" % cv)
        pred_path = training_config.pred_path
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)
        with open(pred_path + ("cv%d.pred" % cv), 'w') as f:
            for (user_id, item_list) in results.items():
                if isinstance(item_list, np.ndarray):
                    for i in range(len(item_list)):
                        f.write("%d\t%d\t1.0\n" % (user_id, item_list[i]))
                else: # only one item
                    f.write("%d\t%d\t1.0\n" % (user_id, item_list))
        logger.debug("Predictions saved.")
        
        coord.request_stop()
        coord.join(threads)
    
if __name__=='__main__':
    train(FLAGS.cv, FLAGS.z_size, FLAGS.embed_size)
