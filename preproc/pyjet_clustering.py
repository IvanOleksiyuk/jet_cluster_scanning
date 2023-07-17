import os
import sys
#Make sure the path to the root directory of the project is in the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pyjet import cluster
import h5py
from utils.config_utils import Config

def generator(filename, chunksize=100000, total_size=1100000):
    i = 0
    while True:
        yield pd.read_hdf(filename,start=i*chunksize, stop=(i+1)*chunksize)
        
        i+=1
        if (i+1)*chunksize > total_size:
            break

def extract_events_jets(
        infilepath='events_anomalydetection_v2.h5', 
        outfilepath='tmp.h5',
        CHUNKSIZE=5000, 
        NEVENTS=110000):
    """
    Script that extracts the leading and subleading jet from each event and
    stores them in a new file.

    events with shape (NEVENTS, 2, 200, 4), second dimension giving leading
        and subleading jet, last dimension e, px, py, pz
    jet_info with shape (NEVENTS, 2, 4) giving the mass, tau1, tau2, tau3 for
        both jets
    labels with shape (NEVENTS,) giving 1 for signal, zero for background events
    """
    events = np.zeros((NEVENTS, 2, 200, 4), dtype=np.float32)
    jet_info = np.zeros((NEVENTS, 2, 4))
    
    labels = np.ones(NEVENTS, dtype=np.int8) * (-1)
    #n_constituents = np.zeros((NEVENTS, 2,))
    m_jj = np.zeros(NEVENTS)
    n_tot = 0
    for ind1, data in enumerate(generator(infilepath, chunksize=CHUNKSIZE, total_size=NEVENTS)):
        n_tot += data.shape[0]
        jets = np.reshape(data.to_numpy(dtype=np.float32)[:, :-1], (CHUNKSIZE, 700, 3))
        
        labels[ind1 * CHUNKSIZE: (ind1 + 1) * CHUNKSIZE] = data.to_numpy()[:, -1]
        print("in all chuncks from 0 to", ind1, "found anomalous lables", labels[:(ind1 + 1) * CHUNKSIZE].sum())
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
            assert np.array_equal(leading2, np.array([0, 1])), f'Something wrong {leading2}'
            for ind2, jet in enumerate(clustered[:2]):
                constis = []
                for constit in jet:
                    constis.append((constit.e, constit.px, constit.py, constit.pz))
                    
                # Store pT ordered list of constituents
                constis = np.array(constis)
                args = np.argsort(-constis[:, 0])
                constis = constis[args]
                if len(constis) > 200:
                    print("found a jet with more than 200 constituents")
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
            # invariant mass of the two jets
            m_jj[ind1*CHUNKSIZE+n_event] = ((clustered[0].e + clustered[1].e)**2 - (clustered[0].px + clustered[1].px)**2 - (clustered[0].py + clustered[1].py)**2 - (clustered[0].pz + clustered[1].pz)**2)**0.5

    with h5py.File(outfilepath, 'w') as f:
        f.create_dataset('events', data=events, dtype='float32')
        f.create_dataset('jet_info', data=jet_info, dtype='float32')
        f.create_dataset('labels', data=labels, dtype='int')
        f.create_dataset('m_jj', data=m_jj, dtype='float32')
        
if __name__ == '__main__':
    cfg = Config("config/path.yaml")
    extract_events_jets(infilepath = cfg.get("data_directory")+cfg.get("initial_file"),
                        outfilepath = cfg.get("data_directory")+cfg.get("clustered_jets_file"),
                        NEVENTS = 1100000)