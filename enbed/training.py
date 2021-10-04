import torch
import time
import numpy as np

def train_energy(optimizer, batcher, model, samples, steps):
    '''
    Training function for energy-based models.

    optimizer: pytorch optimizer
    batcher: batcher object providing mini-batches
    model: energy-based scorer
    samples: number of samples for the free-running phase
    steps: number of training steps
    '''
    starttime = time.time()
    for k in range(steps):
        if k%100 == 0:
            estimate = (time.time()-starttime)/(k+1)*(steps-k)/60.
            print('ETA {}min'.format(np.round(estimate,2)), end =  '\r')
        optimizer.zero_grad()
        databatch = batcher.next_batch()

        loss = model.cost(databatch[0], databatch[1], databatch[2], samples)
        loss.backward()

        optimizer.step()

def train_RESCAL(optimizer, batcher, model, steps, whichloss = 'KL'):
    '''
    Training function for RESCAL.

    optimizer: pytorch optimizer
    batcher: batcher object providing mini-batches
    model: RESCAL scorer
    steps: number of training steps
    whichloss: loss function, either mean square error (MSE) or Kullback-Leibler (KL)
    '''
    if whichloss == 'MSE':
        lossf = torch.nn.MSELoss()
    elif whichloss == 'KL':
        lossf = torch.nn.KLDivLoss()

    starttime = time.time()
    for k in range(steps):
        if k%100 == 0:
            estimate = (time.time()-starttime)/(k+1)*(steps-k)/60.
            print('ETA {}min'.format(np.round(estimate,2)), end =  '\r')
        optimizer.zero_grad()
        databatch = batcher.next_batch()

        prediction = model.score(databatch[0], databatch[1], databatch[2])
        targets = (torch.tensor(databatch[-1])>0)*1.

        if whichloss == 'KL':
            loss = lossf(F.log_softmax(prediction), targets)
        else:
            loss = lossf(prediction, targets)
        loss.backward()

        optimizer.step()
