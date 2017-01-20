import tensorflow as tf
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
            W = tf.variables(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W')
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expand = tf.expand_dims(self.embedded_chars, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            # Convolution layer
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.variables(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
            b = tf.variables(tf.constant(0.1, shape=[num_filters]), name='b')
            conv = tf.conv2d(
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
            W = tf.Variable(tf.truncate_normal([num_filters_total, num_classes], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(value=0.1, shape=[num_classes]), name='b')
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, name='predictions')
            


if __name__ == '__main__':
    a = tf.placeholder(tf.int16)
    b = tf.placeholder(tf.int16)
    with tf.device('/cpu:0'), tf.Session() as sess:
        add = tf.add(a, b)
        mul = tf.mul(a, b)
        print(sess.run(add, feed_dict={a: 3, b: 2}))
        print(sess.run(mul, feed_dict={a: 4, b: 3}))
