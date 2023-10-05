import os 
import pickle

def adj_wr_correlation(species, species_wr, num_samples):
    if species != 'shrimp':
        with open(os.path.join('sentience_estimates', '{}_psent_hv1_model.p'.format(species)), 'rb') as f_s:
            species_psent = list(pickle.load(f_s))
    else:
        with open(os.path.join('sentience_estimates', 'shrimp_assumed_psent.p'), 'rb') as f_s:
            species_psent = list(pickle.load(f_s))

    species_adj_wr = []

    for i in range(num_samples):
        psent_i = species_psent[i]
        wr_i = species_wr[i]
        adj_wr_i = max(psent_i*wr_i, 0)
        species_adj_wr.append(adj_wr_i)
    
    return species_adj_wr