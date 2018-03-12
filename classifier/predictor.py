import numpy as np
import tensorflow as tf
from classifier.flags import FLAGS


class Predictor(object):

    def __init__(self, classifier):
        self.classifier = classifier
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

    def predict(self, x):
        with tf.Session(config=self.config) as sess:
            saver = tf.train.Saver()
            print("Restoring Variables from Checkpoint for rnn model.")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
            start = 0
            number_of_training_data = len(x)
            batch_size = FLAGS.batch_size
            curr_epoch = sess.run(self.classifier.epoch_step)
            print("Load model from epoch %d" % (curr_epoch,))
            result = np.array([])
            while start < number_of_training_data:
                end = min(start + batch_size, number_of_training_data)
                predict = sess.run(
                    [self.classifier.predictions],
                    feed_dict={self.classifier.input_x: x[start:end],
                               self.classifier.dropout_keep_prob: 1.0}
                )
                result = np.concatenate((result, predict[0]))
                start += batch_size
        return result
