import torch
from torch.nn import functional as F

def DistMult_score(s_emb, o_emb, p_emb):
    '''
    DistMult triple score function (Yang et al., 2014).
    '''
    return (s_emb*p_emb*o_emb).sum(-1)

def RESCAL_score(s_emb, o_emb, p_emb):
    '''
    RESCAL triple score function (Nickel et al., 2011).
    '''
    return (s_emb.unsqueeze(-1)*torch.matmul(p_emb, o_emb.unsqueeze(-1))).sum(-2).T[0]
