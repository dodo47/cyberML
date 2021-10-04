import torch
import numpy as np
import matplotlib.pyplot as plt
from enbed.utils.scorer import RESCAL_score
from enbed.utils.scorer import DistMult_score

class RESCAL_Scorer:
    def __init__(self, num_nodes, num_predicates, dim, seed = 1231245):
        '''
        Implementation of the RESCAL graph embedding model (Nickel et al., 2011).

        dim: embedding dimension
        num_nodes: number of nodes in the graph
        num_predicates: number of relation types in the graph
        '''
        self.dim = dim
        self.num_nodes = num_nodes
        self.num_predicates = num_predicates

        # embeddings
        torch.manual_seed(seed)
        self.entities = torch.nn.Embedding(num_nodes, dim)
        self.predicates = torch.nn.Embedding(num_predicates, dim*dim)

    def score(self, subj, pred, obj):
        '''
        Score a list of triple [[s0, p0, o0], [s1, p1, o1],...]

        subj, pred and obj are lists [s0, s1, ...], [p0, p1, ...], [o0, o1, ...]
        '''
        s_emb = self.entities(torch.tensor(subj).long())
        o_emb = self.entities(torch.tensor(obj).long())
        p_emb = self.predicates(torch.tensor(pred).long())

        return RESCAL_score(s_emb, o_emb, p_emb.view(-1, self.dim, self.dim))

    def prob(self, subj, pred, obj):
        '''
        Apply sigmoid to score.
        '''
        return torch.sigmoid(self.score(subj, pred, obj))

    def save(self, savepath, appdix = ''):
        '''
        Save and visualize embeddings.
        '''
        pred_embs = self.predicates.weight.data.detach().numpy()
        ent_embs = self.entities.weight.data.detach().numpy()
        np.save('{}/predicate_embeddings_{}.npy'.format(savepath, appdix), pred_embs)
        np.save('{}/entity_embeddings_{}.npy'.format(savepath, appdix), ent_embs)

        plt.close()
        for j in range(50):
            plt.vlines(ent_embs[j], j+0.1, (j+1)-0.1)
        plt.savefig('{}/entity_embeddings_{}.png'.format(savepath, appdix))

        plt.close()
        for j in range(len(pred_embs)):
            plt.vlines(pred_embs[j], j+0.1, (j+1)-0.1)
        plt.savefig('{}/predicate_embeddings_{}.png'.format(savepath, appdix))


class Energy_Scorer(RESCAL_Scorer):
    def __init__(self, num_nodes, num_predicates, dim, seed = 1231245):
        '''
        Energy-based model for calculating embeddings. The cost is obtained using stochastic sampling.
        '''
        super().__init__(num_nodes, num_predicates, dim, seed)
        np.random.seed(seed)

    def cost(self, subj, pred, obj, num_samples, burnin=0):
        '''
        Cost function using sampling to maximize data likelihood.
        '''
        nbatch = len(subj)
        pscore = self.score(subj, pred, obj)

        total_score = 0
        old_score = pscore
        for k in range(num_samples+burnin):
            spo = np.random.randint(3, size=nbatch)
            smask = (spo == 0)
            omask = (spo == 2)
            pmask = (spo == 1)

            new_subj = ~smask*subj + smask*(np.random.random(nbatch)*self.num_nodes)
            new_obj = ~omask*obj + omask*(np.random.random(nbatch)*self.num_nodes)
            new_pred = ~pmask*pred + pmask*(np.random.random(nbatch)*self.num_predicates)

            proposal_score = self.score(new_subj, new_pred, new_obj)

            filters = 1.*(torch.rand(nbatch) <= torch.exp(proposal_score-old_score))
            old_score = proposal_score*filters + old_score*(1-filters)

            filters = filters.detach().numpy()
            subj = np.array(new_subj*filters + subj*(1-filters), dtype=int)
            obj = np.array(new_obj*filters + obj*(1-filters), dtype=int)
            pred = np.array(new_pred*filters + pred*(1-filters), dtype=int)

            if k >= burnin:
                total_score += old_score.sum()

        cost = -pscore.sum() + 1./num_samples*total_score
        return cost

class Energy_Diag_Scorer(Energy_Scorer):
    def __init__(self, num_nodes, num_predicates, dim, seed = 1231245):
        '''
        Energy-based model for calculating embeddings, with diagonally-constrained relation matrices.
        Similar to DistMult (Yang et al., 2014).
        '''
        super().__init__(num_nodes, num_predicates, dim, seed)
        self.predicates = torch.nn.Embedding(num_predicates, dim)

    def score(self, subj, pred, obj):
        s_emb = self.entities(torch.tensor(subj).long())
        o_emb = self.entities(torch.tensor(obj).long())
        p_emb = self.predicates(torch.tensor(pred).long())

        return DistMult_score(s_emb, o_emb, p_emb)
