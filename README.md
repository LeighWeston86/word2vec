## Word2vec with tensorflow and keras using the skip-gram approach.

### Usage

##### Load the data

w2v/data contains a small data set of hotel reviews for training. To get production embeddings replace train_data.csv with a larger text corpus.

```python
from w2v.model.utils import get_data
train_data, word2idx, idx2word = get_data()
```

##### Train the model

```python
from w2v.model.tf_model import EmbeddingTrainer
embedd = EmbeddingTrainer(0.5, len(word2idx), word2idx)
embedd.train(train_data)
result = embedd.get_embedding_array()
```

##### Assess

After training, save the embeddings and vocabulary. The wordvectors can be assessed against the analogy set as follows:

```python
from w2v.model.utils import assess, get_analogies
vocab = list(word2idx)
get_analogies(vocab)
accuracy = assess(embeddings, word2idx, idx2word, analogies)
```
