import os
import pickle
import random
import csv
import pandas as pd
from scipy import stats

## Generates the simulated data for each species' sentience proxies

def sent_simulate_fn(SPECIES,UNKNOWN_PROB,WEIGHT_NO,HC_WEIGHT,N_SCENARIOS,VERBOSE,CSV,SAVE,PATH,update_every):

    judgments = pd.read_csv(os.path.join('input_data', 'Sentience Judgments.csv'))
    
    hc_csv = os.path.join('input_data', 'Sentience High-Confidence Proxies.csv')
    hc_proxies = set()
    with open(hc_csv, newline='') as f:
        reader = csv.reader(f)
        hc_proxies_lists = list(reader)
    for item in hc_proxies_lists:
        hc_proxies.add(item[0])
    
    if WEIGHT_NO == "Yes":
        judgment_prob_map = {'likely no': {'lower': 0, 'upper': 0.25},
                        'lean no': {'lower': 0.25, 'upper': 0.50},
                        'lean yes': {'lower': 0.50, 'upper': 0.75},
                        'likely yes': {'lower': 0.75, 'upper': 1.00},
                        'unknown': UNKNOWN_PROB, 'yes': 1, 'na': 0}
    else:
        judgment_prob_map = {'likely no': {'lower': 0, 'upper': 0},
                        'lean no': {'lower': 0, 'upper': 0},
                        'lean yes': {'lower': 0.50, 'upper': 0.75},
                        'likely yes': {'lower': 0.75, 'upper': 1.00},
                        'unknown': UNKNOWN_PROB, 'yes': 1, 'na': 0}
    
    species_scores = judgments[["proxies", SPECIES]]
    
    simulated_probs = {}
    simulated_scores = {}
    for proxy in species_scores["proxies"].to_list():
        simulated_probs[proxy] = []
        simulated_scores[proxy] = []
    
    for s in range(N_SCENARIOS):
        if s % update_every == 0:
            if VERBOSE:
                print('-')
                print('### SCENARIO {} ###'.format(s + 1))
            else:
                print('... Completed {}/{}'.format(s + 1, N_SCENARIOS))

        #Set up for iteration over proxies                
        bern_probs = []
        proxies_arr = []
        num_proxies = len(species_scores)
        judgements = species_scores[SPECIES]
        
        for ii in range(0,num_proxies):
            proxy = species_scores.proxies[ii]
    
            judgment = judgements[ii]
            judgment = judgment.lower()
    
    
            if judgment in {'unknown', 'yes', 'na'}:
                proxy_prob = judgment_prob_map[judgment]
            else:
                lower_prob = judgment_prob_map[judgment]['lower']
                upper_prob = judgment_prob_map[judgment]['upper']
                proxy_prob = random.uniform(lower_prob, upper_prob)
                
            bern_probs.append(proxy_prob)
            proxies_arr.append(proxy)

        #Obtain bernoulli draws         
        flips = stats.bernoulli.rvs(bern_probs,size=len(bern_probs))
        
        #Store bernoulli draw results        
        for ii in range(0,num_proxies):

            proxy = proxies_arr[ii]  
            has_proxy = flips[ii]
            if proxy in hc_proxies:
                score = HC_WEIGHT*has_proxy
            else:
                score = has_proxy
            simulated_probs[proxy].append(proxy_prob)
            simulated_scores[proxy].append(score)
            
    if SAVE:
        print('... Saving 1/1')
        pickle.dump(simulated_scores, open('{}simulated_scores.p'.format(PATH), 'wb'))
        
    return simulated_scores