import nltk
import os
import pandas as pd
import numpy as np

def get_data(num_docs=10000, batch_size=128, data_path=None):
    if not data_path:
        local_dir = os.path.dirname(__file__)
        data_path = os.path.join(local_dir, 'data/hotel-reviews.csv')
    df = pd.read_csv(data_path)
    reviews = list(df['Description'])[:num_docs]
    processed = pre_process(reviews)
    word2idx, idx2word, vocab = get_lookups(processed)
    encoded = [[[word2idx[token] if token in vocab else word2idx['UNK'] for token in sent]
               for sent in doc] for doc in processed]
    batches = get_batches(encoded, batch_size)
    return batches, word2idx, idx2word

def pre_process(docs):
    processed_docs = []
    for doc in docs:
        sents = nltk.sent_tokenize(doc)
        tokenized_sents = [nltk.word_tokenize(sent) for sent in sents]
        processed = [[token.lower() for token in sent] for sent in tokenized_sents]
        processed_docs.append(processed)
    return processed_docs

def get_lookups(docs, min_freq=3):
    words = [word for doc in docs for sent in doc for word in sent]
    fd = nltk.FreqDist(words)
    vocab = {word for word in words if fd[word] >= min_freq}
    vocab.add('UNK')
    word2idx = {word:idx for idx, word in enumerate(vocab)}
    idx2word = {idx:word for idx, word in enumerate(vocab)}
    return word2idx, idx2word, vocab

def get_batches(docs, batch_size=128, window_size=5, num_skips=3):
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


def get_analogies(): pass







