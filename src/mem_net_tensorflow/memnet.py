import tensorflow as tf

class MemNet: 

    def memnet (vocab_size=10, embedding_size=5, labels_size=2):
        self.data = tf.placeholder(tf.float32, [vocab_size, embedding_size])
        self.labels = tf.placeholder(tf.float32, [labels_size])
        self.network = self.regression(self.data, labels_size)
        return self.network

    def embedding():
        pass

    def regression(self, input_tensor, output_size):
        """Return tensors for multinomial regression
        of embedding input on some label space"""

        weight_dims = tf.pack([tf.shape(input_tensor)[1], output_size])
        regression_weights = tf.random_normal(weight_dims)
        regression_biases = tf.Variable(tf.zeros([output_size]))

        regression_outputs = tf.nn.softmax(tf.matmul(input_tensor, regression_weights) + regression_biases)
        return regression_outputs

    def train(train_data, train_labels, batch_size):
        loss = tf.reduce_mean(-tf.reduce_sum(train_labels * tf.log(self.network)))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        for i in range(1000):
            train_step.run(feed_dict={self.data: train_data, self.labels: train_labels})

def test_regression():
    memnet = MemNet()
    input_tensor = tf.random_normal([10,100])
    regr = memnet.regression(input_tensor, 2)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    result = sess.run(regr)
    assert result.shape == (10, 2)



