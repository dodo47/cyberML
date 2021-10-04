import numpy as np
from copy import deepcopy

class batch_provider:
    def __init__(self, data, batchsize, num_negSamples = 2, seed = 1231245):
        '''
        Helper class to provide data in batches with negative examples.

        data: Training data triples
        batchsize: size of the mini-batches
        num_negSamples: number of neg. samples.
        seed: random seed for neg. sample generation
        '''
        self.data = deepcopy(data)
        self.num_nodes = np.max([np.max(data[:,0]), np.max(data[:,2])])

        np.random.seed(seed)
        np.random.shuffle(self.data)

        self.batchsize = batchsize
        self.number_minibatches = int(len(self.data)/batchsize)
        self.current_minibatch = 0

        self.num_negSamples = num_negSamples

    def next_batch(self):
        '''
        Return the next mini-batch.
        Data triples are shuffled after each epoch.
        '''
        i = self.current_minibatch
        di = self.batchsize
        mbatch = deepcopy(self.data[i*di:(i+1)*di])
        self.current_minibatch += 1
        if self.current_minibatch == self.number_minibatches:
            np.random.shuffle(self.data)
            self.current_minibatch = 0
        if self.num_negSamples > 0:
            subj, pred, obj, labels = self.apply_neg_examples(list(mbatch[:,0]), list(mbatch[:,1]), list(mbatch[:,2]))
            return subj, pred, obj, labels
        else:
            return mbatch[:,0], mbatch[:,1], mbatch[:,2]

    def apply_neg_examples(self, subj, pred, obj):
        '''
        Generate neg. samples for a mini-batch.
        Both subject and object neg. samples are generated.
        '''
        vsize = len(subj)
        labels = np.array([1 for i in range(vsize)] + [-1 for i in range(self.num_negSamples*2*vsize)])
        neg_subj = list(np.random.randint(self.num_nodes, size = self.num_negSamples*vsize))
        neg_obj = list(np.random.randint(self.num_nodes, size = self.num_negSamples*vsize))
        return np.concatenate([subj, neg_subj, subj*self.num_negSamples]), np.concatenate([pred*(2*self.num_negSamples+1)]), np.concatenate([obj, obj*self.num_negSamples, neg_obj]), labels
