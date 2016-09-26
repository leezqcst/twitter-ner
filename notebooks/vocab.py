""" A convenience vocabulary wrapper """
from collections import Counter
import numpy as np
import numpy.random as npr

class Vocab():
    def __init__(self, all_tokens=None, min_count=5):
        self.min_count=min_count
        self.count_index = Counter()
        self._vocab2idx = {'<PAD>':0,
                           '<UNK>':1}
        self._idx2vocab = {0:'<PAD>',
                           1:'<UNK>'}
        self.vocabset = set(self._vocab2idx.keys())
        self.idxset = set(self._idx2vocab.keys())
        
        if all_tokens:
            self.use(all_tokens)

        self._n = sum( count for count in self.count_index.values() if count >= self.min_count)
        self._v = sum( 1 for count in self.count_index.values() if count >= self.min_count)

        self.make_sampling_table()
        
    @property
    def n(self):
        return self._n    

    @property
    def v(self):
        return self._v

    @property
    def pad(self):
        return '<PAD>'
    
    @property
    def ipad(self):
        return 0
    
    def idx(self, token):
        if token in self.vocabset:
            return self._vocab2idx[token]
        else:
            return self._vocab2idx['<UNK>']
        
    def token(self, idx):
        if idx in self.idxset:
            return self._idx2vocab[idx]
        else:
            return self._idx2vocab['<UNK>']
    
    def use(self, tokens):
        self.count_index = Counter()
        self.add(tokens)        
    
    def add(self, tokens):
        for token in tokens:
            self.count_index[token] += 1
        self._vocab2idx = {'<UNK>':0}
        self._vocab2idx.update({token:i+1 for i, (token, count) in enumerate(self.count_index.most_common())
                                if count >= self.min_count})
        self._idx2vocab = {i:token for token, i in self._vocab2idx.items()}
        self.vocabset = set(self._vocab2idx.keys())
        self.idxset = set(self._idx2vocab.keys())
        self._n = sum( count for count in self.count_index.values() if count >= self.min_count)
        self._v = sum( 1 for count in self.count_index.values() if count >= self.min_count)
        
    def count(self, token):
        return self.count_index[token]

    def make_sampling_table(self, power_scalar=.75):
        # from 0 to V-1, get the frequency
        self.vocab_distribution = np.array([ (self.count_index[self._idx2vocab[idx]]/float(self._n))**power_scalar
                                    for idx in range(len(self.idxset))])
        self.vocab_distribution /= np.sum(self.vocab_distribution)

    def sample(self, sample_shape):
        # sample a tensor of indices
        # by walking up the CDF
        # setting each position to the index
        # of the word which is the closest
        # word with that CDF
        sums = np.zeros(sample_shape)
        rands = npr.uniform(size=sample_shape)
        idxs = np.zeros(sample_shape)
        for i in range(len(self.vocab_distribution)):
            sums += self.vocab_distribution[i]
            idxs[sums <= rands] = i
        return idxs.astype(np.int)