import tensorflow as tf

# configuration
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float("learning_rate", 0.1, "learning rate")
tf.app.flags.DEFINE_integer("batch_size", 512, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("decay_steps", 120, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 0.9, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_string("ckpt_dir", "./data/checkpoint/", "checkpoint location for the model")
tf.app.flags.DEFINE_boolean("is_training", True, "is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs", 200, "embedding size")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")
tf.app.flags.DEFINE_boolean("use_embedding", True, "whether to use embedding or not.")
tf.app.flags.DEFINE_string("traning_data_path", "/home/xul/xul/9_ZhihuCup/test_twoCNN_zhihu.txt",
                           "path of traning data.")
tf.app.flags.DEFINE_string("word2vec_model_path", "./data/GoogleNews-vectors-negative300.bin.gz",
                           "word2vec's vocabulary and vectors")
