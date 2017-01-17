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


if __name__ == '__main__':
    a = tf.placeholder(tf.int16)
    b = tf.placeholder(tf.int16)
    add = tf.add(a, b)
    mul = tf.mul(a, b)
    with tf.Session() as sess:
        print(sess.run(add, feed_dict={a: 3, b: 2}))
        print(sess.run(mul, feed_dict={a: 4, b: 3}))
