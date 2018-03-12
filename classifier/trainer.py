import os

import tensorflow as tf

from classifier.flags import FLAGS


class Trainer(object):
    def __init__(self, classifier, embeddings):
        self.classifier = classifier
        self.embeddings = embeddings
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

    def train(self, x, y):
        print('Start training....')
        valid_portion = int(0.8 * len(x))
        train_x = x[:valid_portion]
        train_y = y[:valid_portion]
        test_x = x[valid_portion + 1:]
        test_y = y[valid_portion + 1:]
        with tf.Session(config=self.config) as sess:
            # Initialize Save
            saver = tf.train.Saver()
            if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):
                print("Restoring Variables from Checkpoint for rnn model.")
                saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
            else:
                print('Initializing Variables')
                sess.run(tf.global_variables_initializer())
                if FLAGS.use_embedding:  # load pre-trained word embedding
                    self.assign_pre_trained_word_embedding(sess)
            curr_epoch = sess.run(self.classifier.epoch_step)
            # feed data & training
            number_of_training_data = len(train_x)
            batch_size = FLAGS.batch_size
            for epoch in range(curr_epoch, FLAGS.num_epochs):
                loss, acc, counter = 0.0, 0.0, 0
                start = 0
                while start < number_of_training_data:
                    end = min(start + batch_size, number_of_training_data)
                    curr_loss, curr_acc, _ = sess.run(
                        [self.classifier.loss_val, self.classifier.accuracy, self.classifier.train_op],
                        feed_dict={self.classifier.input_x: train_x[start:end],
                                   self.classifier.input_y: train_y[start:end],
                                   self.classifier.dropout_keep_prob: 1.0}
                    )
                    loss, counter, acc = loss + curr_loss, counter + 1, acc + curr_acc
                    start += batch_size

                print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.3f" % (
                    epoch, counter, loss / float(counter),
                    acc / float(counter)))
                # epoch increment
                sess.run(self.classifier.epoch_increment)
                # 4.validation
                if epoch % FLAGS.validate_every == 0:
                    eval_loss, eval_acc = self.do_eval(sess, test_x, test_y, batch_size)
                    print("Epoch %d Validation Loss:%.3f\tValidation Accuracy: %.3f" % (epoch, eval_loss, eval_acc))

            self.do_eval(sess, test_x, test_y, batch_size)
            self.checkpoint(sess, saver, FLAGS.num_epochs)

    def do_eval(self, sess, eval_x, eval_y, batch_size):
        number_examples = len(eval_x)
        eval_loss, eval_acc, eval_counter = 0.0, 0.0, 0
        start = 0
        while start < number_examples:
            end = min(start + batch_size, number_examples)
            curr_eval_loss, logits, curr_eval_acc = sess.run(
                [self.classifier.loss_val, self.classifier.logits, self.classifier.accuracy],
                feed_dict={self.classifier.input_x: eval_x[start:end],
                           self.classifier.input_y: eval_y[start:end],
                           self.classifier.dropout_keep_prob: 1})
            eval_loss, eval_acc, eval_counter = eval_loss + curr_eval_loss, eval_acc + curr_eval_acc, eval_counter + 1
            start += batch_size
        eval_loss /= float(eval_counter)
        eval_acc /= float(eval_counter)
        return eval_loss, eval_acc

    @staticmethod
    def checkpoint(sess, saver, epoch):
        # save model to checkpoint
        save_path = FLAGS.ckpt_dir + "model.ckpt." + str(epoch)
        if not os.path.exists(FLAGS.ckpt_dir):
            os.mkdir(FLAGS.ckpt_dir)
        saver.save(sess, save_path, global_step=epoch)

    def assign_pre_trained_word_embedding(self, sess):
        word_embedding = tf.constant(self.embeddings, dtype=tf.float32)
        t_assign_embedding = tf.assign(self.classifier.Embedding, word_embedding)
        sess.run(t_assign_embedding)


if __name__ == "__main__":
    tf.app.run()
