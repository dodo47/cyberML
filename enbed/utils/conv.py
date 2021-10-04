import numpy as np

def get_rel2id(datapath):
    '''
    Convenience function returning a dictionary that turns relation names into ids.
    '''
    file = open('{}/{}'.format(datapath, 'relation_ids.del'))
    content = file.readlines()
    ids = []
    nodenames = []
    for i in range(len(content)):
        a = content[i].split('\t')
        ids.append(int(a[0]))
        nodenames.append(a[-1][:-1])
    rel2id = dict(zip(nodenames, ids))

    return rel2id

def get_id2rel(datapath):
    '''
    Convenience function returning a dictionary that turns ids into relation names.
    '''
    rel2id = get_rel2id(datapath)
    id2rel = dict(zip(rel2id.values(), rel2id.keys()))

    return id2rel

def get_ent2id(datapath):
    '''
    Convenience function returning a dictionary that turns entity names into ids.
    '''
    file = open('{}/{}'.format(datapath, 'entity_ids.del'))
    content = file.readlines()
    ids = []
    nodenames = []
    for i in range(len(content)):
        a = content[i].split('\t')
        ids.append(int(a[0]))
        nodenames.append(a[-1][:-1])
    ent2id = dict(zip(nodenames, ids))

    return ent2id

def get_id2ent(datapath):
    '''
    Convenience function returning a dictionary that turns ids into entity names.
    '''
    ent2id = get_ent2id(datapath)
    id2ent = dict(zip(ent2id.values(), ent2id.keys()))

    return id2ent

def load_data(datapath):
    '''
    Load the data from datapath.

    datapath: path to the data folder

    Output
    training triples, number of nodes, number of relations
    '''
    train_data = np.array(np.loadtxt('{}/{}'.format(datapath, 'train.del')), dtype=int)

    num_nodes = np.max([np.max(train_data[:,0]), np.max(train_data[:,2])])+1
    num_predicates = np.max(train_data[:,1])+1

    return train_data, num_nodes, num_predicates

def load_test(testpath, testcase):
    '''
    Load test data from testpath.

    testpath: path to the test folder
    testcase: benchmark case to use (variables_access, ssh, https, scan, credential_use)

    Output
    list of triples as ids, list of triples with names, list of labels
    labels are:
    0 - highly suspicious
    1 - suspicious
    2 - unexpected
    3 - expected
    4 - observed during training
    '''
    trip_data = np.loadtxt('{}/{}.del'.format(testpath, testcase))
    labels = np.array(trip_data[:,-1:], dtype=int).flatten()
    triples = np.array(trip_data[:,:-1], dtype=int)

    named_triples = open('{}/{}.txt'.format(testpath, testcase))
    read_data = named_triples.read()
    triples_named = []
    for lines in read_data.split('\n')[:-1]:
        triples_named.append(lines.split('\t')[:-1])
    triples_named = np.array(triples_named)
    named_triples.close()

    return triples, triples_named, labels
