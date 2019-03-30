import math
import tensorflow as tf
import numpy as np

class EmbeddingTrainer(object):

    def __init__(self, learning_rate,
                 vocab_size,
                 word2idx,
                 use_nce=True,
                 embedding_size=200,
                 batch_size=128,
                 window_size=5,
                 num_skips=3,
                 num_negative=64):
        """
        A class to train a word2vec model using the skip-gram model approach. This implementation uses either
        the softmax output layer with a node for each word in the vocabulary, or noise contranstive estimation (nce);
        it is recommended to use nce, as the softmax approach is extremely slow.

        :param learning_rate: float; learning rate for gradient descent
        :param vocab_size: int; number of words in the vocabulary
        :param word2idx: dict; mappings between words and their vocabulary index
        :param use_nce: Boolean; uses nce if True (recomended), uses softmax if False
        :param embedding_size: int; size of embeddings to train
        :param batch_size: int; number of samples in a minibatch
        :param window_size: int; size of the skip-gram context window
        :param num_skips: int; number of words to sample from the skip-gram context window
        :param num_negative: int; number of negative context words to sample (only relevant for nce)
        """

        # Get the instance attributes
        self.learning_rate = learning_rate
        self.vocab_size = vocab_size
        self.word2idx = word2idx
        self.use_nce = use_nce
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.window_size = window_size
        self.num_skips = num_skips
        self.num_negative = num_negative

        # Set up the graph
        self.add_placeholders()
        self.initialize_weights()
        self.add_loss_op()
        self.sess = tf.Session()

    def add_placeholders(self):
        """
        Adds the placeholders to the graph. Defines self.inputs and self.context, for the
        center and context words respectively
        """

        self.inputs = tf.placeholder(shape=[self.batch_size], dtype=tf.int32, name='X')
        self.context = tf.placeholder(shape=[self.batch_size, 1], dtype=tf.int32, name='y')

    def initialize_weights(self):
        """
        Initialize the weights for the two layers. Defines the weights and biases of the hideen layer,
        self.W_out and self.b_out. Defines self.logits, the linear activations of the hidden layer,
        self.embeddings, the embedding matrix, and self.embedd, the embedding lookup.
        """
        # Embdding layer
        self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1, 1))
        self.embedd = tf.nn.embedding_lookup(self.embeddings, self.inputs)

        # Output layer - one node for each word in vocab
        self.W_out = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size],
                                                     stddev=1.0 / math.sqrt(self.embedding_size)))
        self.b_out = tf.Variable(tf.zeros([self.vocab_size]))

        self.logits = tf.add(tf.matmul(self.embedd, tf.transpose(self.W_out)), self.b_out)

    def add_loss_op(self):
        """
        Sets up loss with softmax_cross_entropy, defines self.loss and self.optimizer
        """
        context_onehot = tf.one_hot(self.context, self.vocab_size)
        if self.use_nce:
            self.loss = tf.reduce_mean(tf.nn.nce_loss(
                weights=self.W_out,
                biases=self.b_out,
                labels=self.context,
                inputs=self.embedd,
                num_sampled=self.num_negative,
                num_classes=self.vocab_size))
        else:
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits,
                labels=context_onehot))
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    def train(self, batches, num_epochs=10, verbose=True):
        """
        Trains the model.

        :param batches: list; should be an iterable containing tuples of the form (input, context)
        :param num_epochs: int; number of passes over the training set
        :param verbose: Boolean; if true, prints loss after each epoch
        """
        # Inititalize
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # Train
        for epoch in range(num_epochs):
            epoch_loss = 0
            for input_batch, context_batch in batches:
                feed_dict = {
                    self.inputs: input_batch,
                    self.context: np.array(context_batch).reshape(-1, 1)
                }
                _, loss = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
                epoch_loss += loss
            if verbose:
                print("Loss after epoch {}: {}".format(epoch, epoch_loss/len(batches)))

    def get_embedding_array(self):
        embeddings = self.sess.run(self.embeddings)
        return embeddings
