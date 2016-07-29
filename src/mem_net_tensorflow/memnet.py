import sys
import logging
import tensorflow as tf
import numpy as np
from sklearn.metrics import log_loss

logging.basicConfig()

class MemNet: 

    def memnet(self, vocab_size=10, embedding_size=5, labels_size=2):
        self.data = tf.placeholder(tf.float32, [vocab_size, embedding_size])
        self.labels = tf.placeholder(tf.float32, [labels_size])
        self.network = self.regression(self.data, labels_size)
        return self.network

    def embedding(self):
        pass

    def regression(self, input_tensor, output_size):
        """Return tensors for multinomial regression
        of embedding input on some label space"""

        weight_dims = tf.pack([tf.shape(input_tensor)[1], output_size])
        regression_weights = tf.random_normal(weight_dims)
        regression_biases = tf.Variable(tf.zeros([output_size]))

        regression_outputs = tf.nn.softmax(tf.matmul(input_tensor, regression_weights) + regression_biases)
        return regression_outputs

    def train(self, train_data, train_labels, batch_size=32):
        logger = logging.getLogger(__name__)
        losses=[]
        loss = tf.reduce_mean(-tf.reduce_sum(train_labels * tf.log(self.network)))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        data_to_feed = {self.data: train_data, self.labels: train_labels}
        for i in range(10):
            train_step.run(feed_dict=data_to_feed)
            if i % 10 == 0:
                result = self.network.eval(feed_dict=data_to_feed)
                logger.debug("Run {}".format(i))
            losses.append(log_loss(train_labels.reshape(1,-1), result))
                
        return np.mean(losses)

def test_regression():
    memnet = MemNet()
    input_tensor = tf.random_normal([10,100])
    regr = memnet.regression(input_tensor, 2)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    result = sess.run(regr)
    assert result.shape == (10, 2)

def test_net():
    memnet = MemNet()
    train_data = np.random.random((10, 100))
    # easy one-hot set: is row sum above 50 (0, 1) or below (1, 0)?
    train_labels = np.sum(train_data, axis=1) > 50.0
    train_labels = np.array([[0, 1] if over_50 else [1, 0] for over_50 in train_labels])
    
    memnet.memnet(1, 100, 2)
    
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    mean_losses=[]
    for epoch_idx in range(500):
        with sess.as_default():
            obs_idx = np.random.choice(range(train_data.shape[0]))
            mean_loss = memnet.train(train_data[obs_idx].reshape(1,-1), train_labels[obs_idx])
            mean_losses.append(mean_loss)
            print("Mean loss:", np.mean(mean_losses))
    

