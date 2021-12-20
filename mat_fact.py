import torch
import numpy as np
import scipy as sp
import scipy.sparse, scipy.io
import scipy.sparse.linalg

def compute_pmi_inf(adj, rank_approx=None):
    lap, deg_sqrt = sp.sparse.csgraph.laplacian(adj, normed=True, return_diag=True)
    iden = sp.sparse.identity(adj.shape[0])
    vol = adj.sum()
    ss_probs_invsqrt = np.sqrt(vol) / deg_sqrt # inverse square root of stationary probabilities
    lap_pinv = np.linalg.pinv(lap, hermitian=True)
    return 1. + ss_probs_invsqrt[:,np.newaxis] * np.array(lap_pinv - iden) * ss_probs_invsqrt[np.newaxis,:]

def compute_log_ramp(pmi_inf, T=3, thresh=np.finfo(float).eps):
    pmi_inf_trans = T * np.log(np.maximum(thresh, 1. + pmi_inf / T))
    return pmi_inf_trans

def compute_mat_embed(mat, dims=128):
    w, v = sp.sparse.linalg.eigsh(mat, k=dims)
    return np.sqrt(np.abs(w))[np.newaxis,:] * v

def test():
    a = torch.randint(2, (20, 20))
    a = (a + a.t()).clamp(max=1)
    adj = a.numpy()

    pmi_inf = compute_pmi_inf(adj)
    pmi_inf_trans = compute_log_ramp(pmi_inf, T=10)
    emb = compute_mat_embed(pmi_inf_trans, 4)

    print(emb.shape)

if __name__ == "__main__":
    test()
