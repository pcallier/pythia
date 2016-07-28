import tensorflow as tf

class MemNet:

    def regression(self, input_tensor, input_size, output_size):
        """Return tensors for multinomial regression
        of embedding input on some label space"""

        weight_dims = tf.pack([tf.shape(input_tensor)[1], output_size])
        regression_weights = tf.random_normal(weight_dims)
        regression_biases = tf.Variable(tf.zeros([output_size]))

        regression_outputs = tf.nn.softmax(tf.matmul(input_tensor, regression_weights) + regression_biases)
        return regression_outputs

def test_regression():
    memnet = MemNet()
    input_tensor = tf.random_normal([10,100])
    regr = memnet.regression(input_tensor, 100, 2)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    result = sess.run(regr)
    assert result.shape == (10, 2)
