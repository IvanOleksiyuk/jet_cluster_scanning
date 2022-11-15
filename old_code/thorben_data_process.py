import pandas as pd
import numpy as np
from pyjet import cluster
import time, h5py

def generator(filename, chunksize=100000,total_size=1100000):
    i = 0
    while True:
        yield pd.read_hdf(filename,start=i*chunksize, stop=(i+1)*chunksize)
        
        i+=1
        if (i+1)*chunksize > total_size:
            break

def extract_events_jets():
    start = time.time()
    NEVENTS = 110000
    CHUNKSIZE = 5000
    FILENAME = 'events_anomalydetection.h5'
    
    events = np.zeros((NEVENTS, 2, 200, 4), dtype=np.float32)
    jet_info = np.zeros((NEVENTS, 2, 4))
    
    labels = np.ones(NEVENTS, dtype=np.int8) * (-1)
    n_constituents = np.zeros((NEVENTS, 2,))
    n_tot = 0
    for ind1, data in enumerate(generator(FILENAME, chunksize=CHUNKSIZE, total_size=NEVENTS)):
        n_tot += data.shape[0]
        jets = np.reshape(data.to_numpy(dtype=np.float32)[:, :-1], (CHUNKSIZE, 700, 3))
        
        labels[ind1 * CHUNKSIZE: (ind1 + 1) * CHUNKSIZE] = data.to_numpy()[:, -1]
        print(ind1, labels[:(ind1 + 1) * CHUNKSIZE].sum())
        jets = np.append(jets, np.zeros((CHUNKSIZE, 700, 1)), axis=-1)
        
        for n_event in range(len(jets)):
            current_jet = jets[n_event][jets[n_event, :, 0]!=0]
            
            particles = []
            for part in current_jet:
                particles.append(tuple(part[i] for i in range(4)))
                
            tmp_jet = np.array(particles, dtype=np.dtype([('pT', 'f8'), ('eta', 'f8'), ('phi', 'f8'), ('mass', 'f8')]))
            
            sequence = cluster(tmp_jet, R=1.0, p=-1)
            clustered = sequence.inclusive_jets(ptmin=20)
            
            leading2 = np.argsort([-x.pt for x in clustered])[:2]
            assert np.array_equal(leading2, np.array([0, 1])), f'Whats poppin {leading2}'
            n = 0
            for ind2, jet in enumerate(clustered[:2]):
                constis = []
                for constit in jet:
                    constis.append((constit.e, constit.px, constit.py, constit.pz))
                    
                # Store pT ordered list of constituents
                constis = np.array(constis)
                args = np.argsort(-constis[:, 0])
                constis = constis[args]
                events[ind1*CHUNKSIZE+n_event, ind2, :min(200, len(constis))] = constis[:200]
                
                # Store jet mass
                jet_info[ind1*CHUNKSIZE+n_event, ind2, 0] = jet.mass
                
                # Get subjettiness for n = 1, 2, 3
                constis = jet.constituents_array()
                sub_sequence = cluster(constis, R=1.0, p=1)
                n_constis = len(constis)
                for nsubj in [1, 2, 3]:
                    if n_constis <= nsubj: continue
                    subjets = sub_sequence.exclusive_jets(nsubj)
                    
                    dRs = np.zeros((n_constis, nsubj))
                    for j, subj in enumerate(subjets):
                        dRs[:, j] = np.sqrt(
                            (constis['eta'] - subj.eta) ** 2 +
                            ((constis['phi'] - subj.phi + np.pi) % (2 * np.pi) - np.pi) ** 2)
                        
                    dRs = np.min(dRs, axis=1)
                    tau = 1. / jet.pt * np.sum(constis['pT'] * dRs)
                    jet_info[ind1*CHUNKSIZE+n_event, ind2, nsubj] = tau
                    
    """
    events with shape (NEVENTS, 2, 200, 4), second dimension giving leading
        and subleading jet, last dimension e, px, py, pz
    jet_info with shape (NEVENTS, 2, 4) giving the mass, tau1, tau2, tau3 for
        both jets
    labels with shape (NEVENTS,) giving 1 for signal, zero for background events
    """
    with h5py.File('tmp.h5', 'w') as f:
        f.create_dataset('events', data=events, dtype='float32')
        f.create_dataset('jet_info', data=jet_info, dtype='float32')
        f.create_dataset('labels', data=labels, dtype='int')
        
        
if __name__ == '__main__':
    extract_events_jets()
    pass