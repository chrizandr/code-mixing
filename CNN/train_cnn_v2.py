import tensorflow as tf
import numpy as np
import os
import time
import json
import sys
import pickle
import datetime
import data_helpers
from model import CNNModel
from w2v import Word2VecModel, GloveVecModel
from data import build_vocab, load_sentences
from tensorflow.contrib import learn

# Parameters
# ==================================================
embeddings = ["/home/upendra/WORD_EMBEDDINGS/GoogleNews-vectors-negative300.bin", "/home/upendra/WORD_EMBEDDINGS/glove.twitter.27B.100d.txt", "/home/upendra/WORD_EMBEDDINGS/codemixed_w2vec.txt"]
models = [Word2VecModel, GloveVecModel, GloveVecModel]
# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .2, "Percentage of the training data to use for validation")
tf.flags.DEFINE_float("test_sample_percentage", .2, "Percentage of test data to use for testing")
tf.flags.DEFINE_string("data_file", "../Data/CODEMIXED_TRAIN_DATA.json", "Data source for the negative data.")
tf.flags.DEFINE_string("test_file", "../Data/CODEMIXED_TEST_DATA.json", "Data source for the negative data.")
#tf.flags.DEFINE_string("glove_embed_file", "/home/upendra/WORD_EMBEDDINGS/GoogleNews-vectors-negative300.bin", "Pretrained embeddings")
#tf.flags.DEFINE_string("glove_embed_file", "/home/upendra/WORD_EMBEDDINGS/glove.twitter.27B/glove.twitter.27B.100d.txt", "Pretrained embeddings")
tf.flags.DEFINE_string("glove_embed_file", embeddings[int(sys.argv[1])], "Pretrained embeddings")
tf.flags.DEFINE_boolean('clean', True, 'Do cleaning')
tf.flags.DEFINE_boolean('chars', False, 'Use characters instead of words')
tf.flags.DEFINE_boolean('static', False, 'Fix pre-trained embeddings weights')

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("conv_filter_sizes", "[[3,4,5],[3,]]", "Comma-separated filter sizes (default: '[[3,4,5],[3,]]')")
tf.flags.DEFINE_string("pool_filter_sizes", "[[2,2,2],[2,]]", "Comma-separated filter sizes (default: '[[2,2,2],[2,]]')")
tf.flags.DEFINE_string("n_filters", "[50,100]", "Number of filters per filter size (default: '[50,100]')")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("unif",0.25,"range of randomly initialized vectors")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

conv_filter_sizes = [3, 2]
pool_filter_sizes = [[2,],[2,]]
n_filters = [50,100]

# Data Preparation
# ==================================================
vocab = build_vocab([FLAGS.data_file, FLAGS.test_file], FLAGS.clean, FLAGS.chars)
unif = 0 if FLAGS.static else FLAGS.unif
#w2vModel = Word2VecModel(FLAGS.glove_embed_file, vocab, unif)
w2vModel = models[int(sys.argv[1])](FLAGS.glove_embed_file, vocab, unif)

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(FLAGS.data_file)

print(len(x_text))

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
max_document_length = 100
# vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
# x = np.array(list(vocab_processor.fit_transform(x_text)))
dataset = load_sentences([(y1, x) for x, y1 in zip(x_text, y)], w2vModel.vocab, FLAGS.clean, FLAGS.chars, max_document_length)
x_train = dataset.x
print(dataset.x.shape)
y_train = dataset.y


x_text_test, y_test = data_helpers.load_data_and_labels(FLAGS.test_file)
dataset2 = load_sentences([(y1, x) for x, y1 in zip(x_text_test, y_test)], w2vModel.vocab, FLAGS.clean, FLAGS.chars, max_document_length)
x_test = dataset2.x
print(dataset2.x.shape)
y_test = dataset2.y

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y_train)))
x_train = x_train[shuffle_indices]
y_train = y_train[shuffle_indices]


# Split train/test set
# TODO: This is very crude, should use cross-validation

# Get test data
# test_sample_index = -1 * int(FLAGS.test_sample_percentage * float(len(y)))
# x_train, x_test = x_shuffled[:test_sample_index], x_shuffled[test_sample_index:]
# y_train, y_test = y_shuffled[:test_sample_index], y_shuffled[test_sample_index:]
# print("Train/Test split: {:d}/{:d}".format(len(y_train), len(y_test)))

# Get validation data
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y_train)))
x_dev = x_train[dev_sample_index:]
x_train = x_train[:dev_sample_index]
y_dev = y_train[dev_sample_index:]
y_train = y_train[:dev_sample_index]
# print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = CNNModel(
            sentence_length=x_train.shape[1],
            n_classes=y_train.shape[1],
            w2vec=w2vModel,
            conv_filter_sizes=conv_filter_sizes,
            pool_filter_sizes=pool_filter_sizes,
            n_filters=n_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        pickle.dump(vocab, open(os.path.join(out_dir, "vocab.pkl"), "wb"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.X: x_batch,
              cnn.y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)
            return loss, accuracy

        def dev_step(X, labels, batch_size=50):
            """
            Evaluates model on a dev set
            """
            batch_num = int(len(X) / batch_size)
            validation_accuracy = 0
            prev = 0
            predicted_vector = []
            for i in range(batch_num):
                x_dev_batch = X[prev:prev+batch_size]
                y_dev_batch = labels[prev:prev+batch_size]
                prev += batch_size
                pred_list, test_acc = sess.run(
                    [cnn.predictions, cnn.accuracy],
                    feed_dict={
                        cnn.X: x_dev_batch,
                        cnn.y: y_dev_batch,
                        cnn.dropout_keep_prob: 1.0})
                validation_accuracy += test_acc
                predicted_vector.extend(list(pred_list))
            validation_accuracy /= batch_num
            print("validation accuracy %g" % validation_accuracy)

        def test_step(X, labels, batch_size=50):
            """
            Evaluates model on a test set
            """
            batch_num = int(len(X) / batch_size)
            testing_accuracy = 0
            prev = 0
            predicted_vector = []
            for i in range(batch_num+1):
                x_dev_batch = X[prev:prev+batch_size]
                y_dev_batch = labels[prev:prev+batch_size]
                prev += batch_size
                pred_list, test_acc = sess.run(
                    [cnn.predictions, cnn.accuracy],
                    feed_dict={
                        cnn.X: x_dev_batch,
                        cnn.y: y_dev_batch,
                        cnn.dropout_keep_prob: 1.0})
                testing_accuracy += test_acc
                predicted_vector.extend(list(pred_list))

                if i == (batch_num-1):
                    batch_size = int(len(X) % batch_size)

            testing_accuracy /= batch_num
            pickle.dump(predicted_vector, open("{}".format(sys.argv[2]), "wb"))
            print("Testing accuracy %g" % testing_accuracy)

        def batch_iter(data, batch_size, num_epochs, shuffle=True):
            """
            Generates a batch iterator for a dataset.
            """
            data = np.array(data)
            data_size = len(data)
            num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
            for epoch in range(num_epochs):
                # Shuffle the data at each epoch
                if shuffle:
                    shuffle_indices = np.random.permutation(np.arange(data_size))
                    shuffled_data = data[shuffle_indices]
                else:
                    shuffled_data = data
                start_progress_log(epoch, num_batches_per_epoch)
                epcomplete = False
                for batch_num in range(num_batches_per_epoch):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size)
                    x_batch, y_batch = zip(*shuffled_data[start_index:end_index])
                    train_loss, train_acc = train_step(x_batch, y_batch)
                    update_progress_log(batch_num, epoch, num_batches_per_epoch, train_loss, train_acc)
                    current_step = tf.train.global_step(sess, global_step)
                print("\nValidating ...")
                dev_step(x_dev, y_dev)
                print("")

        def start_progress_log(epoch, num_batches_per_epoch):
            printProgressBar(0,num_batches_per_epoch,prefix='Epoch {}: [{}/{}]'.format(epoch, 0,num_batches_per_epoch), suffix='Complete', length=50)

        def update_progress_log(batch_num, epoch, num_batches_per_epoch, train_loss, train_acc):
            printProgressBar(
                batch_num,
                num_batches_per_epoch,
                prefix='Epoch {}: [{}/{}]'.format(epoch, batch_num + 1,num_batches_per_epoch),
                suffix='loss: {}, acc: {}, Complete'.format(train_loss, train_acc),
                length=50)

        def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '>'):
            """
            Call in a loop to create terminal progress bar
            @params:
                iteration   - Required  : current iteration (Int)
                total       - Required  : total iterations (Int)
                prefix      - Optional  : prefix string (Str)
                suffix      - Optional  : suffix string (Str)
                decimals    - Optional  : positive number of decimals in percent complete (Int)
                length      - Optional  : character length of bar (Int)
                fill        - Optional  : bar fill character (Str)
            """
            percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
            filledLength = int(length * iteration // total)
            bar = fill * filledLength + '-' * (length - filledLength)
            print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
            # Print New Line on Complete
            if iteration == total:
                print()

        # Training loop
        batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, int(FLAGS.num_epochs))
        """
        # Training loop. For each batch...
        for batch, epcomplete in batches:
            x_batch, y_batch = zip(*batch)
            # x_dev_batch, y_dev_batch = zip(*dev_batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if epcomplete:
                print("\nValidating ...")
                dev_step(x_dev, y_dev)
                # dev_step(x_dev_batch, y_dev_batch, writer=dev_summary_writer)
                print("")
            #if current_step % FLAGS.checkpoint_every == 0:
            #    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            #    print("Saved model checkpoint to {}\n".format(path))
        """
        print("\nTesting ...")
        test_step(x_test, y_test)
