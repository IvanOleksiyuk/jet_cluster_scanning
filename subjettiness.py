import numpy as np
from pyjet import cluster


def subjettinesses(jet, n=3):
    constis = jet.constituents_array()
    sub_sequence = cluster(constis, R=1.0, p=1)
    n_constis = len(constis)
    taus=[]
    for nsubj in range(1, n+1):
        if n_constis <= nsubj: 
            taus.append(np.nan)
            print("jet has less than", nsubj+1, "const")
            continue
        subjets = sub_sequence.exclusive_jets(nsubj)
        
        dRs = np.zeros((n_constis, nsubj))
        for j, subj in enumerate(subjets):
            dRs[:, j] = np.sqrt(
                (constis['eta'] - subj.eta) ** 2 +
                ((constis['phi'] - subj.phi + np.pi) % (2 * np.pi) - np.pi) ** 2)
            
        dRs = np.min(dRs, axis=1)
        tau = 1. / jet.pt * np.sum(constis['pT'] * dRs)
        taus.append(tau)
    return taus