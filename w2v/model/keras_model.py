import numpy as np
from keras.layers import Input, Embedding, Reshape, Dense, dot
from keras.models import Model
from keras.preprocessing.sequence import make_sampling_table, skipgrams
from keras.optimizers import RMSprop

class EmbeddingTranier(object):
    """
    A class to train the word2vec using the skip-gram approach in keras with negative sampling.
    """

    def __init__(self, vocab_size, embedding_size, window_size=3):
        """
        Constructor method for embedding trainer.

        :param vocab_size: int; the number of words in the vocabulary
        :param embedding_size: int; the size of embeddings to train
        :param window_size: int; size of the skip-gram context window
        """
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.model = None
        self.embeddings = None


    def get_skips(self, docs):
        """
        Formats the data and generates negative samples.

        :param docs: list; a list of documents; each document is a list of sentences;
        a sentence is a list of tokens (strings)
        :return: tuple; contains the center and context words, and the corresponding labels
        """
        sampling_table = make_sampling_table(self.vocab_size)
        center_words, context_words, labels = [], [], []
        for doc in docs:
            tokens = [token for sent in doc for token in sent]
            pairs, labels_ = skipgrams(tokens,
                                        self.vocab_size,
                                        window_size=self.window_size,
                                        sampling_table=sampling_table)
            try:
                center, context = zip(*pairs)
            except ValueError:
                continue
            center_words += center
            context_words += context
            labels += labels_

        return center_words, context_words, labels

    def w2v_model(self, learning_rate):
        """
        Generates the neural architecture for the word2vec skip-gram model

        :return: keras.models.Model(); the word2vec model
        """

        # Add the input and embedding layers
        input_center = Input((1,))
        input_context = Input((1,))
        self.embeddings = Embedding(self.vocab_size, self.embedding_size, input_length=1, name="Embeddings")

        # Get the center and context embeddings
        center = self.embeddings(input_center)
        center = Reshape((self.embedding_size, 1))(center)
        context = self.embeddings(input_context)
        context = Reshape((self.embedding_size, 1))(context)

        # Calculate the linear activations
        # dot_product = Concatenate([center, context], mode="dot", dot_axes=1)
        dot_product = dot([center, context], axes=1, normalize=False)
        dot_product = Reshape((1,))(dot_product)

        # Sigmoid activations
        output = Dense(1, activation="sigmoid")(dot_product)

        # Define the model
        model = Model(input=[input_center, input_context], output=output)
        optimizer = RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
        model.compile(loss="binary_crossentropy", optimizer=optimizer)

        return model

    def train(self, docs, num_batches=2000, learning_rate=0.001, verbose=True):
        """
        Optimizes the model on the training data

        :param docs: list; a sequence of documents; each document is a list of sentences;
        a sentence is a list of tokens (strings)
        :param num_batches: int; the number of (center, context) pairs to use in training
        :param verbose: Boolean; if true, prints the loss druing training
        """

        # Get the data and the model
        center_words, context_words, labels = self.get_skips(docs)
        self.model = self.w2v_model(learning_rate)

        # Randomly sample pair/label
        loss = []
        for batch in range(num_batches):

            idx = np.random.randint(0, len(center_words)-1)
            center_word = np.array([center_words[idx]])
            context_word = np.array([context_words[idx]])
            label = np.array([labels[idx]])
            loss += [self.model.train_on_batch([center_word, context_word], label)]

            # Print the loss every 1000 batches
            if len(loss) >= 1000 and verbose:
                print(batch, sum(loss)/1000)
                loss = []

    def get_embedding_array(self):
        """
        Gets the word embeddings

        :return: array; the trained word embeddings
        """
        return self.embeddings.get_weights()[0]
