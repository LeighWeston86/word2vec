import nltk
import os
import pandas as pd
import numpy as np
from operator import itemgetter

def get_data(num_docs=10000, batch_size=128, data_path=None, get_minibatches=True):
    """
    Gets the word2vec training data.

    :param num_docs: int; number of documents to use in training
    :param batch_size: int; size for the minibatches during training
    :param data_path: string; path to training data
    :param get_minibatches: Boolean; if true, partitions training data into minibatches
    :return: tuple; batches, word2idx, idx2word
    """
    if not data_path:
        local_dir = os.path.dirname(__file__)
        data_path = os.path.join(local_dir, "../data/train_data.csv")
    df = pd.read_csv(data_path)
    reviews = list(df["Description"])[:num_docs]
    processed = pre_process(reviews)
    word2idx, idx2word, vocab = get_lookups(processed)
    encoded = [[[word2idx[token] if token in vocab else word2idx["UNK"] for token in sent]
               for sent in doc] for doc in processed]
    if not get_minibatches:
        return encoded,  word2idx, idx2word
    else:
        batches = get_batches(encoded, batch_size)
        return batches, word2idx, idx2word

def pre_process(docs):
    """
    Performs preprocessing: sentence/word tokenization, lowering.

    :param docs: list; a list containing each document as a string
    :return: list; processed documents
    """
    processed_docs = []
    for doc in docs:
        sents = nltk.sent_tokenize(doc)
        tokenized_sents = [nltk.word_tokenize(sent) for sent in sents]
        processed = [[token.lower() for token in sent] for sent in tokenized_sents]
        processed_docs.append(processed)
    return processed_docs

def get_lookups(docs, min_freq=3):
    """
    Gets the dictionaries mapping words onto their index in the embedding array.

    :param docs: list; each document is a list of sentences, each sentence is a list of tokens;
    each token is a string representing a word
    :param min_freq: int; minimum word freaquency to be included in vocab
    :return: tuple; (word2idx, idx2word, vocab)
    """

    words = [word for doc in docs for sent in doc for word in sent]
    fd = nltk.FreqDist(words)
    vocab = sorted([(word, freq) for word, freq in fd.items() if freq >= min_freq],
                   key=itemgetter(1),
                   reverse=True)
    vocab = [word for word, _ in vocab]
    vocab.insert(0, "UNK")
    word2idx = {word:idx for idx, word in enumerate(vocab)}
    idx2word = {idx:word for idx, word in enumerate(vocab)}
    return word2idx, idx2word, vocab

def get_batches(docs, batch_size=128, window_size=5, num_skips=3):
    """
    Partition the training data into minibatches.

    :param docs: list; each document is a list of sentences, each sentence is a list of tokens;
    each token is an integer representing a word in the vocabulary
    :param batch_size: int; size of the minibatches
    :param window_size: int; size of the skip-gram context window
    :param num_skips: int; number of context words to sample
    :return: list; minibatches
    """

    batches = []
    center_batch = []
    context_batch = []
    for doc in docs:
        for sent in doc:
            for idx, token in enumerate(sent):
                context_words = [sent[pos] for pos in range(idx-window_size, idx+window_size+1)
                          if pos != idx and pos < len(sent) and pos >= 0]
                if not context_words:
                    continue
                num_to_sample = num_skips if len(context_words) >= num_skips else len(context_words)
                context_words = np.random.choice(context_words, size=num_to_sample)
                for word in context_words:
                    center_batch.append(token)
                    context_batch.append(word)
                    if len(center_batch) >= batch_size:
                        batches.append((center_batch, context_batch))
                        center_batch = []
                        context_batch = []
    return batches

def get_analogies(vocab):
    """
    Load the analogies for assessing the word vectors.

    :param vocab: set; unique words in the vocabulary
    :return: list; all of the analogies found in the vocabulary
    """

    analogies_path = os.path.join(os.path.dirname(__file__), "../data/analogies.txt")
    with open(analogies_path, "r") as f:
        analogies = [line.strip().split() for line in f if line.strip().split()[0] != ":"]

    in_vocab = []
    for analogy in analogies:
        if all(token in vocab for token in analogy):
            in_vocab.append(analogy)
    return in_vocab

def assess(embeddings, word2idx, idx2word, analogies):
    """
    Assess the trained word vectors against the analogy test set.

    :param embeddings: numpy.ndarray; trained word embeddings
    :param word2idx: dict; maps word onto embedding array index
    :param idx2word: dict; maps embedding array index onto word
    :param vocab: set; all unique words in the vocabulary
    :param analogies: list; contains the analogies; each analogy is a list of length 4
    :return: float; accuracy of the word vectors on the analogy set
    """

    result = []
    for analogy in analogies:
        embedding_0 = embeddings[word2idx[analogy[0]]]
        embedding_1 = embeddings[word2idx[analogy[1]]]
        embedding_2 = embeddings[word2idx[analogy[2]]]
        new_embedding = embedding_1 - embedding_0 + embedding_2
        closest = closest_word(new_embedding, embeddings, idx2word)
        result += [1] if closest == analogy[3] else [0]

    return sum(result)/len(result)

def closest_word(word_embedding, embeddings, idx2word):
    """
    Finds the closest word to a given word vector

    :param word_embedding: np.array; a one dimensional word vector
    :param embeddings: np.ndarray; the word embedding array
    :param idx2word: dict; maps embedding array index onto word
    :return: string; the closest word
    """
    dot = np.dot(word_embedding, embeddings.T)
    cosine = dot/(np.linalg.norm(word_embedding)*np.linalg.norm(embeddings, axis=1))
    closest = np.argsort(cosine)[-1]
    return idx2word[closest]
