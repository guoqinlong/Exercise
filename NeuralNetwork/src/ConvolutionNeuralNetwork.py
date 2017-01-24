import os
from datetime import time
import tensorflow as tf
from tensorflow.tensorboard.tensorboard import FLAGS
import datetime
import data_helpers

class TextCNN(object):
    """
    A CNN for text classification.
    Used an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W')
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expand = tf.expand_dims(self.embedded_chars, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            # Convolution layer
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
            conv = tf.nn.conv2d(
                input=self.embedded_chars_expand,
                filter=W,
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='conv'
            )
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name='h')
            # Max-pooling over the output
            pooled = tf.nn.max_pool(
                    value=h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pooled'
            )
            pooled_outputs.append(pooled)
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Add scores and predictions
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(value=0.1, shape=[num_classes]), name='b')
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses)

        # Calculate Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,"float"), name="accuracy")


def train_step(x_batch, y_batch):
    """
    A single training step
    """
    feed_dict = {
        cnn.input_x: x_batch,
        cnn.input_y: y_batch,
        cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
    }
    _, step, summaries, loss, accuracy = sess.run([train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)
    time_str = datetime.datetime.isoformat()
    print("{}: step {}, loss {:g}, acc {:g}").format(time_str, step, loss, accuracy)
    train_summary_writer.add_summary(summaries, step)


def dev_step(x_batch, y_batch, writer=None):
    """
    Evaluates model on a dev set
    """
    feed_dict = {
        cnn.input_x : x_batch,
        cnn.input_y : y_batch,
        cnn.dropout_keep_prob : 1.0
    }
    step, summaries, loss, accuracy = sess.run([global_step, dev_summary_op, cnn.loss, cnn.accuracy], feed_dict)
    time_str = datetime.datetime().isoformat()
    print("{}: step {}, loss {:g}, acc {:g}").format(time_str, step, loss, accuracy)
    if writer:
        writer.add_summary(summaries, step)


if __name__ == '__main__':
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement
        )
        sess = tf.Session(session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=2,
                vocab_size=len(vocabulary),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=map(int, FLAGS.filter_sizes.split(",")),
                num_filters=FLAGS.num_filters)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-4)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "run", timestamp))
            print("Writing to {} \n".format(out_dir))

            # Summary for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.acccuracy)

            # Train summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summary", "train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summary", "dev")
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)

            # Checkpointing
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            # Tensorflow assumes this directory already exists so we need to create it.
            if not os.path.exists(checkpoint_prefix):
                os.makedirs(checkpoint_prefix)
            saver = tf.train.Saver(tf.all_variables())

            #Initialize all variables.
            sess.run(tf.initialize_all_variables())

            # Generate batches
            batches = data_helpers.batch_iter(
                zip(x_train, y_train),
                FLAGS.batch_size,
                FLAGS.num_epochs
            )

            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluate")
                    dev_step(x_batch, y_batch, dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


