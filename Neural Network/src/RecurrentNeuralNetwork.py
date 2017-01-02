# coding=utf-8
import csv
import itertools
import nltk
import numpy as np


def softmax():
    pass


class RNNNumpy:

    def __init__(self, word_dim, hidden_dim, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(- np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(- np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(- np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

    def forward_propagation(self, x):
        # The total number of time steps
        T = len(x)
        # During forward propagation, we save all hidden states in s because we need them later.
        # We add one additional element for the initial hidden layer, which we set to 0
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)

        #The output at each time step. Again, we save them for later.
        o = np.zeros((T, self.hidden_dim))

        #For each time step
        for t in np.arange(T):
            # Note that we are indexing U by x[t], This is the same as multiplying U with a one-hot vector.
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t - 1]))
            o[t] = softmax(self.V.dot(s[t]))

        return [o, s]

    def predict(self, x):
        # Perform forward propagation, return the index with max value.
        o, s = self.forward_propagation(x)
        return np.argmax(o, 1)

if __name__ == '__main__':
    vocabulary_size = 8000
    unknown_token = 'UNKNOWN_TOKEN'
    sentence_start_token = 'SENTENCE_START'
    sentence_end_token = 'SENTENCE_END'

    print "Reading CSV file..."
    with open('data/reddit-comments-2015-08.csv', 'rb') as f:
        reader = csv.reader(f, skipinitialspace=True)
        reader.next()
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
        sentences = ['%s %s %s' % (sentence_start_token, x, sentence_end_token) for x in sentences]

    print "Parsed %d sentences" % (len(sentences))

    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print "Found %d unique word items" % (len(word_freq.items()))

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    print "using vocabulary size %d." % vocabulary_size
    print "The least frequent word in our vocublary is '%s' and appeared %d times" % (vocab[-1][0], vocab[-1][1])

    # Replace all words not in our vocabulary with unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

    print "\nExample sentence: '%s'" % sentences[0]
    print "\nExample sentence after Pre-processing '%s'" % tokenized_sentences[0]

    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
