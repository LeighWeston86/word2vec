import numpy as np
from w2v.data_utils import get_data
from keras.layers import Input, Embedding, Reshape, Concatenate, Dense, dot
from keras.models import Model
from keras.preprocessing.sequence import make_sampling_table, skipgrams

class EmbeddingTranier(object):

    def __init__(self, vocab_size, embedding_size, window_size=3):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.model = None


    def get_skips(self, docs):
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

    def w2v_model(self):

        # Add the input and embedding layers
        input_center = Input((1,))
        input_context = Input((1,))
        embedding = Embedding(self.vocab_size, self.embedding_size, input_length=1, name="Embeddings")

        # Get the center and context embeddings
        center = embedding(input_center)
        center = Reshape((self.embedding_size, 1))(center)
        context = embedding(input_context)
        context = Reshape((self.embedding_size, 1))(context)

        # Calculate the linear activations
        # dot_product = Concatenate([center, context], mode="dot", dot_axes=1)
        dot_product = dot([center, context], axes=1, normalize=False)
        dot_product = Reshape((1,))(dot_product)

        # Sigmoid activations
        output = Dense(1, activation="sigmoid")(dot_product)

        # Define the model
        model = Model(input=[input_center, input_context], output=output)
        model.compile(loss="binary_crossentropy", optimizer="rmsprop")

        return model

    def train(self, docs, num_batches=200000, verbose=True):

        # Get the data and the model
        center_words, context_words, labels = self.get_skips(docs)
        self.model = self.w2v_model()

        # Randomly sample pair/label
        loss = []
        for batch in range(num_batches):

            idx = np.random.randint(0, len(center_words)-1)
            center_word = np.array([center_words[idx]])
            context_word = np.array([context_words[idx]])
            label = np.array([labels[idx]])
            loss += [self.model.train_on_batch([center_word, context_word], label)]

            # Print the loss every 1000 batches
            if batch % 1000 == 0 and verbose:
                print(batch, sum(loss)/1000)
                loss = []

if __name__ == '__main__':
    data, word2idx, idx2word = get_data(num_docs=100000, get_minibatches=False)
    embedd = EmbeddingTranier(len(word2idx), 200)
    embedd.train(data)





