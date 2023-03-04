# Rethink Priorities Moral Weights Computation - All-In-One Version:
#
# Updated by Matt Romer from the original RP code
# 

################## Setup ##################
import os
import csv
import pandas as pd
import warnings
import pickle
import copy
from sent_simulate_fn_file import *
from wr_simulate_fn_file import *
import matplotlib.pyplot as plt
import test_simulations
import s_model
import sim_utils
import wr_model
import unittest
import numpy as np


################## Simulations ##################

WR_SPECIES = ['pigs', 'chickens', 'carp', 'salmon', 'octopuses', 'shrimp', 'crabs', 'crayfish', 'bees', 'bsf', 'silkworms']

SENT_SPECIES = ['bees', 'cockroaches', 'fruit_flies', 'ants', \
            'c_elegans', 'crabs', 'crayfish', 'earthworms', \
            'sea_hares',  'spiders', 'octopuses', 'chickens', \
            'cows', 'sometimes_operates', 'bsf', \
            'carp', 'salmon', 'silkworms', 'pigs']

wr_default_unknowns = {'pigs': 0, 'chickens': 0, 'carp': 0, 'salmon': 0, 
                            'octopuses': 0, 'shrimp': 0, 'crabs': 0, 'crayfish': 0, 'bees': 0,
                            'bsf': 0, 'silkworms': 0}

sent_default_unknowns = {'bees': 0, 'cockroaches': 0, 'fruit_flies': 0, 'ants':0, \
            'c_elegans': 0, 'crabs': 0, 'crayfish': 0, 'earthworms': 0, \
            'sea_hares': 0, 'spiders': 0, 'octopuses': 0, 'chickens': 0, \
            'cows': 0, 'sometimes_operates': 0, 'bsf': 0, \
            'carp': 0, 'salmon': 0, 'silkworms': 0, 'pigs': 0}

## Sentience 
print("For the PROBABILITY OF SENTIENCE...")
s_unknowns   = copy.deepcopy(sent_default_unknowns)
s_weight_nos = "Yes"
s_hc_weight  = 5

S_PARAMS = {'N_SCENARIOS': 10000, 'UPDATE_EVERY': 1000, "WEIGHT_NOS": s_weight_nos, "HC_WEIGHT": s_hc_weight}

## Welfare Ranges 
print("For the WELFARE RANGES...")
wr_unknowns   = copy.deepcopy(wr_default_unknowns)
wr_weight_nos = "Yes"
wr_hc_weight  = 5

WR_PARAMS = {'N_SCENARIOS': 10000, 'UPDATE_EVERY': 1000, "WEIGHT_NOS": wr_weight_nos, "HC_WEIGHT": wr_hc_weight}



def simulate_scores(species, params, s_or_wr):
    print("### {} ###".format(s_or_wr.upper()))
    print('...Simulate all Scores for {}'.format(species))

    params['species'] = species

    if s_or_wr == "sentience":
        params['path'] = "sent_{}".format(species)
        params['unknown_prob'] = s_unknowns[species]
        sent_simulate_fn(params['species'],params['unknown_prob'],params['WEIGHT_NOS'],\
                         params['HC_WEIGHT'],params['N_SCENARIOS'],False,\
                        'output_data/scores_'+params['path'],True,'output_data/'+params['path']+"_",\
                         params['UPDATE_EVERY'])
    else:
        params['path'] = "wr_{}".format(species)
        params['unknown_prob'] = wr_unknowns[species]
        wr_simulate_fn(params['species'],params['unknown_prob'],params['WEIGHT_NOS'],\
                         params['HC_WEIGHT'],params['N_SCENARIOS'],False,\
                         'output_data/scores_'+params['path'],True,'output_data/'+params['path']+"_",\
                         params['UPDATE_EVERY'])



sim_utils.clear_make_dir('output_data')

pickle.dump(s_unknowns, open(os.path.join('input_data', 'Sentience Unknown Probabilities.p'), 'wb'))
pickle.dump(S_PARAMS, open(os.path.join('input_data', 'Sentience Parameters.p'), 'wb'))
pickle.dump(wr_unknowns, open(os.path.join('input_data', 'Welfare Range Unknown Probabilities.p'), 'wb'))
pickle.dump(WR_PARAMS, open(os.path.join('input_data', 'Welfare Range Parameters.p'), 'wb'))

for species in SENT_SPECIES:
    simulate_scores(species, S_PARAMS, "sentience")

for species in WR_SPECIES:
    simulate_scores(species, S_PARAMS, "welfare ranges")
    
################## Sentience Computation ################## 

### Import Data from Simulations

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (18.5 * 0.65, 10.5 * 0.65)

SCENARIO_RANGES = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]

params = pickle.load(open(os.path.join('input_data', "Sentience Parameters.p"), 'rb'))
NUM_SCENARIOS = params['N_SCENARIOS']
HC_WEIGHT = params['HC_WEIGHT']
WEIGHT_NOS = params['WEIGHT_NOS']

SPECIES = ['bees', 'cockroaches', 'fruit_flies', 'ants', \
            'c_elegans', 'crabs', 'crayfish', 'earthworms', \
            'sea_hares', 'spiders', 'octopuses', 'chickens', \
            'cows', 'sometimes_operates', 'bsf', \
            'carp', 'salmon', 'silkworms', 'pigs']
    
    
# import simulated scores
bee_scores = pickle.load(open('{}_simulated_scores.p'.format(os.path.join('output_data', "sent_bees")), 'rb'))
cockroach_scores = pickle.load(open('{}_simulated_scores.p'.format(os.path.join('output_data', "sent_cockroaches")), 'rb'))
fruit_fly_scores = pickle.load(open('{}_simulated_scores.p'.format(os.path.join('output_data', "sent_fruit_flies")), 'rb'))
ants_scores = pickle.load(open('{}_simulated_scores.p'.format(os.path.join('output_data', "sent_ants")), 'rb'))
c_elegans_scores = pickle.load(open('{}_simulated_scores.p'.format(os.path.join('output_data', "sent_c_elegans")), 'rb'))
crab_scores = pickle.load(open('{}_simulated_scores.p'.format(os.path.join('output_data', "sent_crabs")), 'rb'))
crayfish_scores = pickle.load(open('{}_simulated_scores.p'.format(os.path.join('output_data', "sent_crayfish")), 'rb'))
carp_scores = pickle.load(open('{}_simulated_scores.p'.format(os.path.join('output_data', "sent_carp")), 'rb'))
earthworm_scores = pickle.load(open('{}_simulated_scores.p'.format(os.path.join('output_data', "sent_earthworms")), 'rb'))
sea_hare_scores = pickle.load(open('{}_simulated_scores.p'.format(os.path.join('output_data', "sent_sea_hares")), 'rb'))
spiders_scores = pickle.load(open('{}_simulated_scores.p'.format(os.path.join('output_data', "sent_spiders")), 'rb'))
octopus_scores = pickle.load(open('{}_simulated_scores.p'.format(os.path.join('output_data', "sent_octopuses")), 'rb'))
chicken_scores = pickle.load(open('{}_simulated_scores.p'.format(os.path.join('output_data', "sent_chickens")), 'rb'))
cow_scores = pickle.load(open('{}_simulated_scores.p'.format(os.path.join('output_data', "sent_cows")), 'rb'))
sometimes_operates_scores = pickle.load(open('{}_simulated_scores.p'.format(os.path.join('output_data', "sent_sometimes_operates")), 'rb'))
bsf_scores = pickle.load(open('{}_simulated_scores.p'.format(os.path.join('output_data', "sent_bsf")), 'rb'))
salmon_scores = pickle.load(open('{}_simulated_scores.p'.format(os.path.join('output_data', "sent_salmon")), 'rb'))
silkworm_scores = pickle.load(open('{}_simulated_scores.p'.format(os.path.join('output_data', "sent_silkworms")), 'rb'))
pig_scores = pickle.load(open('{}_simulated_scores.p'.format(os.path.join('output_data', "sent_pigs")), 'rb'))

unknown_probabilities = pickle.load(open(os.path.join('input_data', "Sentience Unknown Probabilities.p"), 'rb'))


sim_utils.clear_make_dir('sentience_estimates')


data = {'bees': {'Scores': bee_scores, 'Unknown Prob': unknown_probabilities['bees']}, 
        'cockroaches': {'Scores': cockroach_scores, 'Unknown Prob': unknown_probabilities['cockroaches']}, 
        'fruit_flies': {'Scores': fruit_fly_scores, 'Unknown Prob': unknown_probabilities['fruit_flies']}, 
        'ants': {'Scores': ants_scores, 'Unknown Prob': unknown_probabilities['ants']},
        'c_elegans': {'Scores': c_elegans_scores, 'Unknown Prob': unknown_probabilities['c_elegans']}, 
        'crabs': {'Scores': crab_scores, 'Unknown Prob': unknown_probabilities['crabs']}, 
        'crayfish': {'Scores': crayfish_scores, 'Unknown Prob': unknown_probabilities['crayfish']}, 
        'earthworms': {'Scores': earthworm_scores, 'Unknown Prob': unknown_probabilities['earthworms']},
        'sea_hares': {'Scores': sea_hare_scores, 'Unknown Prob': unknown_probabilities['sea_hares']}, 
        'spiders': {'Scores': spiders_scores, 'Unknown Prob': unknown_probabilities['spiders']},
        'octopuses': {'Scores': octopus_scores, 'Unknown Prob': unknown_probabilities['octopuses']},
        'chickens': {'Scores': chicken_scores, 'Unknown Prob': unknown_probabilities['chickens']},
        'cows': {'Scores': cow_scores, 'Unknown Prob': unknown_probabilities['cows']},
        'sometimes_operates': {'Scores': sometimes_operates_scores, 'Unknown Prob': unknown_probabilities['sometimes_operates']},
        'bsf': {'Scores': bsf_scores, 'Unknown Prob': unknown_probabilities['bsf']},
        'carp': {'Scores': carp_scores, 'Unknown Prob': unknown_probabilities['carp']},
        'salmon': {'Scores': salmon_scores, 'Unknown Prob': unknown_probabilities['salmon']},
        'silkworms': {'Scores': silkworm_scores, 'Unknown Prob': unknown_probabilities['silkworms']},
        'pigs': {'Scores': pig_scores, 'Unknown Prob': unknown_probabilities['pigs']},
        }

print("### Testing to make sure score generation works ###")
result = test_simulations.test_sentience_scores(data, HC_WEIGHT, SPECIES)
print(result)

### Make Subsets of Proxies for Each Model

# import proxies lists for each model
model_proxies_df = pd.read_csv(os.path.join('input_data', 'Sentience Model Proxies.csv'))

# simple scoring proxies
ss_proxies_lst = model_proxies_df['simple scoring'].dropna().values.tolist()
ss_proxies = set(ss_proxies_lst)


# first high-value proxies list
hv1_proxies_lst = model_proxies_df['#1_high value proxies'].dropna().values.tolist()
hv1_proxies = set(hv1_proxies_lst)

# pleasure and pain centric proxies
pp_proxies_lst = model_proxies_df['#1 pleasure and pain centric model'].dropna().values.tolist()
pp_proxies = set(pp_proxies_lst)

# Anna's high-value proxies
hva_proxies_lst = model_proxies_df['high value proxies_anna'].dropna().values.tolist()
hva_proxies = set(hva_proxies_lst)

# Martina's high-value proxies
hvm_proxies_lst = model_proxies_df['high value proxies_martina'].dropna().values.tolist()
hvm_proxies = set(hvm_proxies_lst)

### Parameters for these Simulations

print("For all the models below, the assumptions are that...")

if WEIGHT_NOS:
    print("'Likely no's and 'Lean no's are given probabilities between [0,0.25) and [0.25,0.5) of being true.")
else:
    print("'Likely no's and 'Lean no's are given no probability of being true.")

print("Proxies we're higly confident matter for sentience are given", HC_WEIGHT, "x the weight of other proxies.")

unknowns_df = s_model.unknown_probs_df(SPECIES,data)

# Priors-Based Scoring

SENT_SPECIES = ['bees', 'cockroaches', 'fruit_flies', 'ants', 'c_elegans', 'crabs', 'crayfish', \
        'earthworms', 'sea_hares', 'spiders', 'octopuses', 'chickens', 'cows', 'bsf', \
        'carp', 'salmon', 'silkworms', 'pigs']

species_caps = ["Bees", "Cockroaches", "Fruit Flies", "Ants", \
                "C. elegans", "Crabs", "Crayfish", "Earthworms", \
                    "Sea Hares", "Spiders", "Octopuses", "Chickens", \
                            "Cows", "Black Soldier Flies", "Carp", "Salmon", "Silkworms", "Pigs"]

d_prob_map = {'very probably yes': {'lower': 0.9, 'upper': 1.0}, 
                           'probably yes': {'lower': 0.65, 'upper': 0.9},
                           'possibly yes': {'lower': 0.5, 'upper': 0.65},
                           'possibly no': {'lower': 0.35, 'upper': 0.5},
                           'probably no': {'lower': 0.1, 'upper': 0.35},
                           'very probably no': {'lower': 0.0, 'upper': 0.1}}

d_judgments = {'bees': 'probably yes', 'cockroaches': 'possibly yes', 'fruit_flies': 'probably yes', 
                'ants': 'possibly yes', 'c_elegans': 'probably no', 'crayfish': 'probably yes', 'crabs': 'probably yes',
                'earthworms': 'probably no', 'sea_hares': 'possibly no', 'spiders': 'possibly yes', 
                'octopuses': 'very probably yes', 'chickens': 'very probably yes', 'cows': 'very probably yes', 
                # this row was made up using similar results from like animals
                'bsf': 'probably yes', 'carp': 'probably yes', 'salmon': 'probably yes', 'silkworms': 'probably no', 'pigs': 'very probably yes'}

daniela_priors = {}
for species in SENT_SPECIES:
        daniela_priors[species] = {'dist_type': 'normal', 'lower': d_prob_map[d_judgments[species]]['lower'], 
                                        'upper': d_prob_map[d_judgments[species]]['upper'], 'lclip': 0, 'rclip': 1}

marcus_priors = {'bees': {'dist_type': 'lognormal', 'lower': 0.02, 
                'upper': 0.6, 'lclip': 0, 'rclip': 1}, 
        'cockroaches': {'dist_type': 'lognormal', 'lower': 0.01, 
                'upper': 0.4, 'lclip': 0, 'rclip': 1}, 
        'fruit_flies': {'dist_type': 'lognormal', 'lower': 0.04, 
                'upper': 0.55, 'lclip': 0, 'rclip': 1}, 
        'ants': {'dist_type': 'lognormal', 'lower': 0.02, 
                'upper': 0.6, 'lclip': 0, 'rclip': 1},
        'c_elegans': {'dist_type': 'lognormal', 'lower': 0.001, 
                'upper': 0.01, 'lclip': 0, 'rclip': 0.01}, 
        'crabs': {'dist_type': 'lognormal', 'lower': 0.05, 
                'upper': 0.6, 'lclip': 0, 'rclip': 1}, 
        'crayfish': {'dist_type': 'lognormal', 'lower': 0.05, 
                'upper': 0.6, 'lclip': 0, 'rclip': 1}, 
        'earthworms': {'dist_type': 'lognormal', 'lower': 0.001, 
                'upper': 0.2, 'lclip': 0, 'rclip': 1},
        'sea_hares': {'dist_type': 'lognormal', 'lower': 0.001, 
                'upper': 0.04, 'lclip': 0, 'rclip': 1}, 
        'spiders': {'dist_type': 'lognormal', 'lower': 0.01, 
                'upper': 0.4, 'lclip': 0, 'rclip': 1},
        'octopuses': {'dist_type': 'normal', 'lower': 0.3, 
                'upper': 0.9, 'lclip': 0, 'rclip': 1},
        'chickens': {'dist_type': 'normal', 'lower': 0.5, 
                'upper': 0.9, 'lclip': 0, 'rclip': 1},
        'cows': {'dist_type': 'normal', 'lower': 0.6, 
                'upper': 0.9, 'lclip': 0, 'rclip': 1},
        # put this as as same as fruit flies
        'bsf': {'dist_type': 'normal', 'lower': 0.04, 
                'upper': 0.55, 'lclip': 0, 'rclip': 1},
        # put this as same as crabs
        'carp': {'dist_type': 'lognormal', 'lower': 0.05, 
                'upper': 0.6, 'lclip': 0, 'rclip': 1},
        'salmon': {'dist_type': 'lognormal', 'lower': 0.05, 
                'upper': 0.6, 'lclip': 0, 'rclip': 1},
        # put this as same as earthworms
        'silkworms': {'dist_type': 'lognormal', 'lower': 0.001, 
                'upper': 0.2, 'lclip': 0, 'rclip': 1},
        # put this as same as cows
        'pigs': {'dist_type': 'normal', 'lower': 0.6, 
                'upper': 0.9, 'lclip': 0, 'rclip': 1},
        
        }

peter_priors = {'bees': {'dist_type': 'normal', 'lower': 0.36, 'upper': 0.44, 'lclip': 0, 'rclip': 1}, 
        'cockroaches': {'dist_type': 'normal', 'lower': 0.18, 'upper': 0.22, 'lclip': 0, 'rclip': 1}, 
        'fruit_flies': {'dist_type': 'normal', 'lower': 0.27, 'upper': 0.33, 'lclip': 0, 'rclip': 1}, 
        'ants': {'dist_type': 'normal', 'lower': 0.225, 'upper': 0.275, 'lclip': 0, 'rclip': 1},
        'c_elegans': {'dist_type': 'lognormal', 'lower': 0.0001, 'upper': 0.02, 'lclip': 0.0001, 'rclip': 1}, 
        'crabs': {'dist_type': 'normal', 'lower': 0.27, 'upper': 0.33, 'lclip': 0, 'rclip': 1}, 
        'crayfish': {'dist_type': 'normal', 'lower': 0.27, 'upper': 0.33, 'lclip': 0, 'rclip': 1}, 
        'earthworms': {'dist_type': 'normal', 'lower': 0.045, 'upper': 0.055, 'lclip': 0, 'rclip': 1},
        'sea_hares': {'dist_type': 'normal', 'lower': 0.045, 'upper': 0.055, 'lclip': 0, 'rclip': 1}, 
        'spiders': {'dist_type': 'normal', 'lower': 0.225, 'upper': 0.275, 'lclip': 0, 'rclip': 1},
        'octopuses': {'dist_type': 'normal', 'lower': 0.63, 'upper': 0.77, 'lclip': 0, 'rclip': 1},
        'chickens': {'dist_type': 'normal', 'lower': 0.72, 'upper': 0.88, 'lclip': 0, 'rclip': 1},
        'cows': {'dist_type': 'normal', 'lower': 0.765, 'upper': 0.935, 'lclip': 0, 'rclip': 1},
        # same as fruit flies
        'bsf': {'dist_type': 'normal', 'lower': 0.27, 'upper': 0.33, 'lclip': 0, 'rclip': 1},
        # same as crabs
        'carp': {'dist_type': 'normal', 'lower': 0.27, 'upper': 0.33, 'lclip': 0, 'rclip': 1},
        'salmon': {'dist_type': 'normal', 'lower': 0.27, 'upper': 0.33, 'lclip': 0, 'rclip': 1},
        # same as earthworms
        'silkworms': {'dist_type': 'normal', 'lower': 0.045, 'upper': 0.055, 'lclip': 0, 'rclip': 1},
        # same as cows
        'pigs': {'dist_type': 'normal', 'lower': 0.765, 'upper': 0.935, 'lclip': 0, 'rclip': 1},
        }

priors_distributions = {'Daniela': daniela_priors, 'Marcus': marcus_priors, 'Peter': peter_priors}

#### Updating based on evidence

priors = s_model.simulate_priors(priors_distributions,SENT_SPECIES,NUM_SCENARIOS)

shrimp_prior_lst = s_model.shrimp_probability_sentience(priors,priors_distributions,NUM_SCENARIOS)

priors_df = s_model.print_priors(priors,SENT_SPECIES, to_plot=False)
print("Priors")
print(priors_df)
priors_df.to_csv(os.path.join('sentience_estimates', "Priors Sentience Summary Statistics.csv"), index_label="Species")

# Simple Scoring
ss_sent_stats, ss_sent_data = s_model.all_species_p_sentience_priors_based(s_model.ss_f, priors, "simple scoring", data, ss_proxies, SENT_SPECIES, NUM_SCENARIOS, sometimes_operates_scores, SCENARIO_RANGES, to_plot=False)
print("Simple Scoring Model:")
print(ss_sent_stats)
s_model.box_plot_adj_wr("Simple Scoring", ss_sent_data, species_caps, showfliers=True)

### #1 High Value Proxies 
hv1_sent_stats, hv1_sent_data = s_model.all_species_p_sentience_priors_based(s_model.hv1_f, priors, "#1_high value proxies", data, hv1_proxies, SENT_SPECIES, NUM_SCENARIOS, sometimes_operates_scores, SCENARIO_RANGES, to_plot=False)
print("#1 High-Value Proxies Model:")
print(hv1_sent_stats)
s_model.box_plot_adj_wr("#1 High-Value Proxies", hv1_sent_data, species_caps)

## Martina's High-Value Proxies
hvm_sent_stats, hvm_sent_data = s_model.all_species_p_sentience_priors_based(s_model.ss_f, priors, "Martina's High-Value Proxies", data, hvm_proxies, SENT_SPECIES, NUM_SCENARIOS, sometimes_operates_scores, SCENARIO_RANGES, to_plot=False)
print("Martina's High-Value Proxies Model:")
print(hvm_sent_stats)
s_model.box_plot_adj_wr("Martina's High-Value Proxies", hvm_sent_data, species_caps)

## Anna's High-Value Proxies
hva_sent_stats, hva_sent_data = s_model.all_species_p_sentience_priors_based(s_model.ss_f, priors, "Anna's High-Value Proxies", data, hva_proxies, SENT_SPECIES, NUM_SCENARIOS, sometimes_operates_scores, SCENARIO_RANGES, to_plot=False)
print("Anna's High-Value Proxies Model:")
print(hva_sent_stats)
s_model.box_plot_adj_wr("Anna's High-Value Proxies", hva_sent_data, species_caps)
    
sim_utils.clear_make_dir('birch_estimates')
    

birch_proxies = pd.read_csv(os.path.join('input_data', 'Birch Model Proxies.csv'))

birch_overalls, birch_sent_data = s_model.all_species_birch_est(SENT_SPECIES,data,priors,birch_proxies,sometimes_operates_scores,NUM_SCENARIOS,SCENARIO_RANGES)
print("Birch Model Sentience Estimates")
print(birch_overalls)
s_model.box_plot_adj_wr("Birch Model", birch_sent_data, species_caps)

proxies = {'simple scoring': {'List': ss_proxies_lst, 'Set': ss_proxies}, 
            '#1_high value proxies': {'List': hv1_proxies_lst, 'Set': hv1_proxies}, 
            '#1 pleasure and pain centric model': {"List": pp_proxies_lst, 'Set': pp_proxies},
            'high value proxies_anna': {"List": hva_proxies_lst, 'Set': hva_proxies}, 
            'high value proxies_martina': {'List': hvm_proxies_lst, 'Set': hvm_proxies}}

models = {'Simple Scoring': {'Proxies': ss_proxies, 'Function': s_model.ss_f},
        '#1 High-Value Proxies': {'Proxies': hv1_proxies, 'Function': s_model.hv1_f},
        "Martina's High-Value Proxies": {'Proxies': hvm_proxies, 'Function': s_model.ss_f}, 
        "Anna's High-Value Proxies": {'Proxies': hva_proxies, 'Function': s_model.ss_f}}   

class TestSimpleFunctions(unittest.TestCase):

    def test_proxies(self):
        pass_test = True
        for model in proxies:
            if set(proxies[model]['List']) != proxies[model]['Set']:
                pass_test = False
        self.assertTrue(pass_test)

    def test_get_sum(self):
        pass_test = True
        for x in [0,1]:  
            for model in proxies:
                test_data = {}
                model_proxies = proxies[model]['Set']
                for proxy in model_proxies:
                    test_data[proxy] = [x]*NUM_SCENARIOS
                for i in range(NUM_SCENARIOS):
                    expected = x*len(model_proxies)
                    actual = s_model.get_sum(test_data, i)
                    if expected != actual:
                        pass_test = False
        self.assertTrue(pass_test)

    def test_one_sim_p_sent_priors_based(self):
        pass_test = True
        fake_so_scores = {}
        fake_priors = {'pigs': np.array([0.5]*NUM_SCENARIOS)}
        for proxy in ss_proxies:
            fake_so_scores[proxy] = [1]*NUM_SCENARIOS
        for x in [0, 1]:
            test_scores = {}
            for proxy in ss_proxies:
                test_scores[proxy] = [x]*NUM_SCENARIOS
            for model in models:
                model_proxies = models[model]['Proxies']
                f = models[model]['Function']
                prediction = 0.5*(f(x)/f(1))**0.25
                for i in range(NUM_SCENARIOS):
                    p = s_model.one_sim_p_sentience_priors_based(f, fake_priors, species, model_proxies, test_scores, fake_so_scores, i)
                    if p != prediction:
                        pass_test = False
                        print("prediction: {}".format(prediction))
                        print("p: {}".format(p))
        self.assertTrue(pass_test)

birch_proxies_dict = {"1: possession of nociceptors": ['physiological responses to nociception or handling', \
                                                        'nociception', 'nociception (strict definition)', \
                                                        'noxious stimuli related vocalizations', 'movement away from noxious stimuli'],
                    "2: possession of integrative brain regions": ['centralized information processing', \
                                                        'vertebrate midbrain-like function'],
                    "3: possession of neural pathways connecting nociceptiorrs to brain regions": ['taste aversion behavior', \
                                                        'pain relief learning', 'long-term behavior alteration to avoid noxious stimulus (24+ hours)', \
                                                        'long-term behavior alteration to avoid noxious stimuli (30+ days)'], 
                    "4: behavioural response modulated by chemical compounds": ['affected by analgesics in a manner similar to humans', \
                                                        'opioid-like receptors', 'affected by recreational drugs in a similar manner to humans', \
                                                        'affected by antidepressants or anxiolytics in a similar manner to humans'], 
                    '5: evidence of motivational trade-offs': ['paying a cost to receive a reward', 'paying a cost to avoid a noxious stimulus', \
                                                        'self-control', 'predator avoidance tradeoffs', 'selective attention to noxious stimuli over other concurrent events'], 
                    "6: flexible self-protective behaviour": ['protective behavior', 'defensive behavior/fighting back'], 
                    "7: evidence of associative learning": ['classical conditioning', 'operant conditioning', 'operant conditioning with unfamiliar action', \
                                                        'contextual learning', 'observational or social learning', 'taste aversion behavior', 'pain relief learning', \
                                                        'long-term behavior alteration to avoid noxious stimulus (24+ hours)', 'long-term behavior alteration to avoid noxious stimuli (30+ days)'],
                    "8: evidence that the animal values a putative analgesic/anaesthetic when injured": ['self-administers analgesics', 'self-administers recreational drugs'], 
                    "9: evidence of affective states": ["anhedonia behavior", "learned helplessness behavior", 'displacement behavior', 'adverse effects of social isolation', \
                                                        'stereotypic behavior', 'fear/anxiety behavior', 'play behavior']}

class BirchModelTests(unittest.TestCase):
    def test_proxies(self):
        birch_proxies = pd.read_csv(os.path.join('input_data', 'Birch Model Proxies.csv'))
        pass_test = True
        for criterion in birch_proxies.columns:
            criterion_proxies = birch_proxies[criterion].dropna().values.tolist()
            if set(criterion_proxies) != set(birch_proxies_dict[criterion]):
                pass_test = False
                print("Failed at {}".format(criterion))
                print("Hard-coded proxies:", birch_proxies_dict[criterion])
                print("Downloaded proxies:", criterion_proxies)
        self.assertTrue(pass_test)

    def test_one_species_birch_sum(self):
        pass_test = True
        birch_sums_2 = {}
        
        for x in [0,1]:
            test_all_scores = {}
            for proxy in ss_proxies:
                if proxy in hv1_proxies:
                    test_all_scores[proxy] = [x*HC_WEIGHT]*NUM_SCENARIOS
                else:
                    test_all_scores[proxy] = [x]*NUM_SCENARIOS

            for criterion, lst in birch_proxies_dict.items():
                test_data = {}
                for proxy in lst:
                    if proxy in hv1_proxies:
                        test_data[proxy] = [x*HC_WEIGHT]*NUM_SCENARIOS
                    else:
                        test_data[proxy] = [x]*NUM_SCENARIOS
                birch_sums = s_model.one_species_birch_sum(birch_proxies, test_all_scores, NUM_SCENARIOS)
                prediction = sum([test_data[proxy][0] for proxy in test_data.keys()])
                for i in range(NUM_SCENARIOS):
                    p = birch_sums[criterion][i]
                    if p != prediction:
                        pass_test = False
                        print("Failed on: ")
                        print("x = {}".format(x))
                        print("Criterion: ", criterion)
                        print("prediction: ", prediction)
                        print("p:", p)

        self.assertTrue(pass_test)

    def test_one_species_birch_est(self):
        pass_test = True
        for x in [0,1]:
            human_all_scores = {}
            species_all_scores = {}
            for proxy in ss_proxies:
                if proxy in hv1_proxies:
                    human_all_scores[proxy] = [1*HC_WEIGHT]*NUM_SCENARIOS
                    species_all_scores[proxy] = [x*HC_WEIGHT]*NUM_SCENARIOS
                else:
                    human_all_scores[proxy] = [1]*NUM_SCENARIOS
                    species_all_scores[proxy] = [x]*NUM_SCENARIOS
            species_sum = 0
            human_sum = 0
            for lst in birch_proxies_dict.values():
                for proxy in lst:
                    species_sum += species_all_scores[proxy][0]
                    human_sum += human_all_scores[proxy][0]
            prediction = species_sum/human_sum
            # function
            human_birch_sums = s_model.one_species_birch_sum(birch_proxies, human_all_scores, NUM_SCENARIOS)
            species_birch_sums = s_model.one_species_birch_sum(birch_proxies, species_all_scores, NUM_SCENARIOS)

            overall_scores = []
            for i in range(NUM_SCENARIOS):
                score_i = 0
                hum_score_i = 0
                for c in species_birch_sums:
                    score_i += species_birch_sums[c][i]
                    hum_score_i += human_birch_sums[c][i]
                overall_scores.append(score_i/hum_score_i)

            for i in range(NUM_SCENARIOS):
                if overall_scores[i] != prediction:
                    pass_test = False
                    print("Failed at:")
                    print("x: ", x)
                    print("prediction: ", prediction)
                    print("p: ", overall_scores[i])
        self.assertTrue(pass_test)

res2 = unittest.main(argv=[''], verbosity=3, exit=False)

################## Welfare Range Computation ################## 

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (18.5 * 0.65, 10.5 * 0.65)

SPECIES = ['pigs', 'chickens', 'carp', 'salmon', 'octopuses', 'shrimp', 'crabs', 'crayfish', 'bees', 'bsf', 'silkworms']

SCENARIO_RANGES = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]

params = pickle.load(open(os.path.join('input_data', "Welfare Range Parameters.p"), 'rb'))
NUM_SCENARIOS = params['N_SCENARIOS']
HC_WEIGHT = params['HC_WEIGHT']
WEIGHT_NOS = params['WEIGHT_NOS']

sent_params = pickle.load(open(os.path.join('input_data', 'Sentience Parameters.p'), 'rb'))
SENT_HC_WEIGHT = sent_params['HC_WEIGHT']

# import simulated scores
pig_scores = pickle.load(open('{}_simulated_scores.p'.format(os.path.join('output_data', "wr_pigs")), 'rb'))
chicken_scores = pickle.load(open('{}_simulated_scores.p'.format(os.path.join('output_data', "wr_chickens")), 'rb'))
carp_scores = pickle.load(open('{}_simulated_scores.p'.format(os.path.join('output_data', "wr_carp")), 'rb'))
salmon_scores = pickle.load(open('{}_simulated_scores.p'.format(os.path.join('output_data', "wr_salmon")), 'rb'))
octopus_scores = pickle.load(open('{}_simulated_scores.p'.format(os.path.join('output_data', "wr_octopuses")), 'rb'))
shrimp_scores = pickle.load(open('{}_simulated_scores.p'.format(os.path.join('output_data', "wr_shrimp")), 'rb'))
crab_scores = pickle.load(open('{}_simulated_scores.p'.format(os.path.join('output_data', "wr_crabs")), 'rb'))
crayfish_scores = pickle.load(open('{}_simulated_scores.p'.format(os.path.join('output_data', "wr_crayfish")), 'rb'))
bee_scores = pickle.load(open('{}_simulated_scores.p'.format(os.path.join('output_data', "wr_bees")), 'rb'))
bsf_scores = pickle.load(open('{}_simulated_scores.p'.format(os.path.join('output_data', "wr_bsf")), 'rb'))
silkworm_scores = pickle.load(open('{}_simulated_scores.p'.format(os.path.join('output_data', "wr_silkworms")), 'rb'))

unknown_probabilities = pickle.load(open(os.path.join('input_data', "Welfare Range Unknown Probabilities.p"), 'rb'))

overlap_csv = os.path.join('input_data', 'Proxy Overlap.csv')
overlap_dict = {}
with open(overlap_csv) as f:
    reader = csv.reader(f, delimiter=',')
    for idx, rec in enumerate(reader):
        if idx == 0:
            continue
        else:
            sent_proxy = rec[0].strip()
            in_both = rec[1].strip()
            corr_proxy = rec[2].strip()
            if in_both == "y":
                if corr_proxy not in overlap_dict:
                    overlap_dict[corr_proxy] = []
                overlap_dict[corr_proxy].append(sent_proxy)

data = {'pigs': {'Scores': pig_scores, 'FFF': 75, 'Unknown Prob': unknown_probabilities['pigs']}, 
        'chickens': {'Scores': chicken_scores, 'FFF': 50, 'Unknown Prob': unknown_probabilities['chickens']}, 
        'carp': {'Scores': carp_scores, 'FFF': 72, 'Unknown Prob': unknown_probabilities['carp']}, 
        'salmon': {'Scores': salmon_scores, 'FFF': 72, 'Unknown Prob': unknown_probabilities['salmon']},
        'octopuses': {'Scores': octopus_scores, 'FFF': 45, 'Unknown Prob': unknown_probabilities['octopuses']}, 
        'shrimp': {'Scores': shrimp_scores, 'FFF': 80, 'Unknown Prob': unknown_probabilities['shrimp']}, 
        'crabs': {'Scores': crab_scores, 'FFF': 14, 'Unknown Prob': unknown_probabilities['crabs']}, 
        'crayfish': {'Scores': crayfish_scores, 'FFF': 55, 'Unknown Prob': unknown_probabilities['crayfish']},
        'bees': {'Scores': bee_scores, 'FFF': 110, 'Unknown Prob': unknown_probabilities['bees']}, 
        'bsf': {'Scores': bsf_scores, 'FFF': None, 'Unknown Prob': unknown_probabilities['bsf']}, 
        'silkworms': {'Scores': silkworm_scores, 'FFF': None, 'Unknown Prob': unknown_probabilities['silkworms']}}

print(test_simulations.test_wr_scores(data, overlap_dict, HC_WEIGHT, SENT_HC_WEIGHT, SPECIES))

sim_utils.clear_make_dir('welfare_range_estimates')

# import proxies lists for each model
model_proxies_df = pd.read_csv(os.path.join('input_data', 'WR Model Proxies.csv'))
# qualitative proxies
qual_proxies_list = model_proxies_df['qualitative'].dropna().values.tolist()
qual_proxies = set()
for proxy in qual_proxies_list:
    if proxy.lower() != "none":
        qual_proxies.add(proxy)

# cubic proxies
cubic_proxies_list = model_proxies_df['cubic'].dropna().values.tolist()
cubic_proxies = set()
for proxy in cubic_proxies_list:
    if proxy.lower() != "none":
        cubic_proxies.add(proxy)

# high-confidence proxies
hc_csv = os.path.join('input_data', 'WR High-Confidence Proxies.csv')
hc_proxies = set()
with open(hc_csv, newline='') as f:
    reader = csv.reader(f)
    hc_proxies_lists = list(reader)
for i, item in enumerate(hc_proxies_lists):
    if i == 0:
        continue
    else:
        hc_proxies.add(item[0])

# qualitative minus social (QMS) proxies
qms_proxies_list = model_proxies_df['qualitative minus social'].dropna().values.tolist()
qms_proxies = set()
for proxy in qms_proxies_list:
    if proxy.lower() != "none":
        qms_proxies.add(proxy)

# pleasure-and-pain-centric (PPC) proxies
ppc_proxies_list = model_proxies_df['pleasure-and-pain-centric'].dropna().values.tolist()
ppc_proxies = set()
for proxy in ppc_proxies_list:
    if proxy.lower() != "none":
        ppc_proxies.add(proxy)

# higher/lower cognitive pleasures (HLP_COG) proxies
hlp_cog_proxies_list = model_proxies_df['higher/lower pleasures - cognitive'].dropna().values.tolist()
hlp_cog_proxies = set()
for proxy in hlp_cog_proxies_list:
    if proxy.lower() != "none":
        hlp_cog_proxies.add(proxy)

# higher/lower hedonic pleasures (HLP_HED) proxies
hlp_hed_proxies_list = model_proxies_df['higher/lower pleasures - hedonic'].dropna().values.tolist()
hlp_hed_proxies = set()
for proxy in hlp_hed_proxies_list:
    if proxy.lower() != "none":
        hlp_hed_proxies.add(proxy)

# undiluted experience cognitive pleasures (UE_COG) proxies
ue_cog_proxies_list = model_proxies_df['undiluted experience - cognitive'].dropna().values.tolist()
ue_cog_proxies = set()
for proxy in ue_cog_proxies_list:
    if proxy.lower() != "none":
        ue_cog_proxies.add(proxy)

# higher/lower hedonic pleasures (HLP_HED) proxies
ue_hed_proxies_list = model_proxies_df['undiluted experience - hedonic'].dropna().values.tolist()
ue_hed_proxies = set()
for proxy in ue_hed_proxies_list:
    if proxy.lower() != "none":
        ue_hed_proxies.add(proxy)

print("For all the models below, the assumptions are that...")

if WEIGHT_NOS == "Yes":
    print("'Likely no's and 'Lean no's are given probabilities between [0,0.25) and [0.25,0.5) of being true.")
else:
    print("'Likely no's and 'Lean no's are given no probability of being true.")

print("Proxies we're higly confident matter for welfare capacities are given", HC_WEIGHT, "x the weight of other proxies.")

wr_model.unknown_probs_df(SPECIES,data)

## Qualitative Model

qual_wr_stats = wr_model.all_species_welfare_ranges_simple_scoring(wr_model.qual_f, "Qualitative", data, qual_proxies, hc_proxies, SPECIES, SCENARIO_RANGES, HC_WEIGHT, NUM_SCENARIOS, to_plot=False)

## High-Confidence (Simple Scoring)


ss_hc_wr_stats = wr_model.all_species_welfare_ranges_simple_scoring(wr_model.ss_hc_f, "High-Confidence (Simple Scoring)", data, hc_proxies, hc_proxies, SPECIES, SCENARIO_RANGES, HC_WEIGHT, NUM_SCENARIOS, to_plot=False)

## Cubic Model

cubic_wr_stats = wr_model.all_species_welfare_ranges_simple_scoring(wr_model.cubic_f, "Cubic", data, cubic_proxies, hc_proxies, SPECIES, SCENARIO_RANGES, HC_WEIGHT, NUM_SCENARIOS, to_plot=False)

## High-Confidence Proxies (Cubic Model)

hc_cubic_wr_stats = wr_model.all_species_welfare_ranges_simple_scoring(wr_model.cubic_hc_f, "High-Confidence (Cubic)", data, \
    hc_proxies, hc_proxies, SPECIES, SCENARIO_RANGES, HC_WEIGHT, NUM_SCENARIOS, to_plot=False)
    
## Qualitative Minus Social Model

qms_wr_stats = wr_model.all_species_welfare_ranges_simple_scoring(wr_model.qms_f, "Qualitative Minus Social", data, \
    qms_proxies, hc_proxies, SPECIES, SCENARIO_RANGES, HC_WEIGHT, NUM_SCENARIOS, to_plot=False)
    
## Pleasure-and-pain-centric Model

ppc_wr_stats = wr_model.all_species_welfare_ranges_simple_scoring(wr_model.ppc_f, "Pleasure-and-pain-centric", data, ppc_proxies, \
    hc_proxies, SPECIES, SCENARIO_RANGES, HC_WEIGHT, NUM_SCENARIOS, to_plot=False)
    
## Higher/Lower Pleasures Model

hlp_wr_stats = wr_model.all_species_welfare_ranges_2(wr_model.hlp_f, "Higher-Lower Pleasures", data, hlp_cog_proxies, \
    hlp_hed_proxies, hc_proxies, SPECIES, NUM_SCENARIOS, HC_WEIGHT, SCENARIO_RANGES, to_plot=False)

## Undiluted Experience Model

ue_wr_stats = wr_model.all_species_welfare_ranges_2(wr_model.ue_f, "Undiluted Experience", data, ue_cog_proxies, \
    ue_hed_proxies, hc_proxies, SPECIES, NUM_SCENARIOS, HC_WEIGHT, SCENARIO_RANGES, to_plot=False)

## Mixture Model    
    
model_results = {'Qualitative': qual_wr_stats, 'High-Confidence Simple Scoring': ss_hc_wr_stats, \
    'Cubic': cubic_wr_stats, 'High-Confidence Cubic': hc_cubic_wr_stats, \
    'Qualitative Minus Social': qms_wr_stats, 'Pleasure-and-pain-centric': ppc_wr_stats, \
    'Higher-Lower Pleasures': hlp_wr_stats, 'Undiluted Experience': ue_wr_stats}
    
mixture = wr_model.all_species_mixture(model_results, [1/8]*8, SPECIES, NUM_SCENARIOS, SCENARIO_RANGES)

## Mixture with Neuron Count

neuron_counts = {'pigs': 0.005350, 'chickens': 0.002439, 
                'carp': 0.000160, 'salmon': 0.001163, 
                'octopuses': 0.005407, 'shrimp': 0.000001, 
                'crabs': 0.000001, 'crayfish': 0.000001, 
                'bees': 0.000013, 'bsf': 0.000004, 
                'silkworms': 0.00001}

_count = wr_model.all_species_mixture_with_neuron_counts(model_results, [1/9]*9, SPECIES, neuron_counts, NUM_SCENARIOS, SCENARIO_RANGES )


SPECIES2 = ['pigs', 'chickens', 'carp', 'octopuses', 'bees', 'salmon', 'crayfish', 'shrimp',  'crabs', 'bsf', 'silkworms']

SPECIES_CAPS = ['Pigs', 'Chickens', 'Carp', 'Octopuses', 'Bees', 'Salmon', 'Crayfish', 'Shrimp',  'Crabs', 'Black Soldier Flies', 'Silkworms']

models = ['Qualitative', 'High-Confidence (Simple Scoring)', \
    'Cubic', 'High-Confidence (Cubic)', \
    'Qualitative Minus Social', 'Pleasure-and-pain-centric', \
    'Higher-Lower Pleasures', 'Undiluted Experience', "Mixture", "Mixture Neuron Count"]

    
### Qualitative
model = "Qualitative"
print(model)
print("P(Sentience) Adjusted Welfare Range:")
qual_df, qual_adj_wrs =  wr_model.all_species_adj_wr(model,NUM_SCENARIOS,SPECIES2,SCENARIO_RANGES)
print(qual_df)
wr_model.box_plot_adj_wr(model, qual_adj_wrs, SPECIES2)

### High-Confidence Simple Scoring
model = "High-Confidence (Simple Scoring)"
print(model)
print("P(Sentience) Adjusted Welfare Range:")
hc_ss_df, hc_ss_adj_wrs = wr_model.all_species_adj_wr(model,NUM_SCENARIOS,SPECIES2,SCENARIO_RANGES)
print(hc_ss_df)
wr_model.box_plot_adj_wr(model, hc_ss_adj_wrs, SPECIES2)

### Cubic
model = "Cubic"
print(model)
print("P(Sentience) Adjusted Welfare Range:")
cubic_df, cubic_adj_wrs = wr_model.all_species_adj_wr(model,NUM_SCENARIOS,SPECIES2,SCENARIO_RANGES)
print(cubic_df)
wr_model.box_plot_adj_wr(model, cubic_adj_wrs, SPECIES2)

### High-Confidence Cubic
model = "High-Confidence (Cubic)"
print(model)
print("P(Sentience) Adjusted Welfare Range:")
hc_cubic_df, hc_cubic_adj_wrs = wr_model.all_species_adj_wr(model,NUM_SCENARIOS,SPECIES2,SCENARIO_RANGES)
print(hc_cubic_df)
wr_model.box_plot_adj_wr(model, hc_cubic_adj_wrs, SPECIES2)

### Qualitative Minus Social
model = "Qualitative Minus Social"
print(model)
print("P(Sentience) Adjusted Welfare Range:")
qms_df, qms_adj_wrs = wr_model.all_species_adj_wr(model,NUM_SCENARIOS,SPECIES2,SCENARIO_RANGES)
print(qms_df)
wr_model.box_plot_adj_wr(model, qms_adj_wrs, SPECIES2)

### Pleasure-and-pain-centric
model = "Pleasure-and-pain-centric"
print(model)
print("P(Sentience) Adjusted Welfare Range:")
ppc_df, ppc_adj_wrs = wr_model.all_species_adj_wr(model,NUM_SCENARIOS,SPECIES2,SCENARIO_RANGES)
print(ppc_df)
wr_model.box_plot_adj_wr(model, ppc_adj_wrs, SPECIES2)

### Higher-lower Pleasures
model = "Higher-Lower Pleasures"
print(model)
print("P(Sentience) Adjusted Welfare Range:")
hlp_df, hlp_adj_wrs = wr_model.all_species_adj_wr(model,NUM_SCENARIOS,SPECIES2,SCENARIO_RANGES)
print(hlp_df)
wr_model.box_plot_adj_wr(model, hlp_adj_wrs, SPECIES2)

### Undiluted Experience
model = "Undiluted Experience"
print(model)
print("P(Sentience) Adjusted Welfare Range:")
ue_df, ue_adj_wrs = wr_model.all_species_adj_wr(model,NUM_SCENARIOS,SPECIES2,SCENARIO_RANGES)
print(ue_df)
wr_model.box_plot_adj_wr(model, ue_adj_wrs, SPECIES2)

### Mixture
model = "Mixture"
print(model)
print("P(Sentience) Adjusted Welfare Range:")
mix_df, mix_adj_wrs = wr_model.all_species_adj_wr(model,NUM_SCENARIOS,SPECIES2,SCENARIO_RANGES)
print(mix_df)
wr_model.box_plot_adj_wr(model, mix_adj_wrs, SPECIES2)

wr_model.box_plot_adj_wr(model, mix_adj_wrs, SPECIES2, showfliers=False)

## Mixture With Neuron Count Model
model = "Mixture Neuron Count"
print(model)
print("P(Sentience) Adjusted Welfare Range:")
mix_df, mix_adj_wrs = wr_model.all_species_adj_wr(model,NUM_SCENARIOS,SPECIES2,SCENARIO_RANGES)
print(mix_df)
wr_model.box_plot_adj_wr(model, mix_adj_wrs, SPECIES2)

wr_model.box_plot_adj_wr(model, mix_adj_wrs, SPECIES2, showfliers=False)

class TestSimpleFunctions(unittest.TestCase,):

    def test_proxies(self):
        pass_test = True
        for model in proxies:
            if set(proxies[model]['List']) != proxies[model]['Set']:
                pass_test = False
        self.assertTrue(pass_test)

    def test_filter_proxies(self):
        pass_test = True
        for model in simple_models:
            model_proxies = simple_models[model]["Proxies"]
            for animal in data.keys():
                animal_scores = data[animal]["Scores"]
                if set(wr_model.filter_proxies(animal_scores, model_proxies).keys()) != model_proxies:
                    pass_test = False
        self.assertTrue(pass_test)    

    def test_one_sim_wr(self):
        pass_test = True
        for x in [0, 1]:
            test_scores = {}
            for proxy in cubic_proxies:
                if proxy in hc_proxies:
                    test_scores[proxy] = [x*HC_WEIGHT]*NUM_SCENARIOS
                else:
                    test_scores[proxy] = [x]*NUM_SCENARIOS
            for model in simple_models:
                model_proxies = simple_models[model]['Proxies']
                filtered_scores = wr_model.filter_proxies(test_scores, model_proxies)

                human_sum = wr_model.get_human_sum(model, model_proxies, hc_proxies, HC_WEIGHT)
                f = simple_models[model]['Function']
                prediction = f(x)
                for i in range(NUM_SCENARIOS):
                    p = wr_model.one_sim_welfare_range(model, f, filtered_scores, i, 60, human_sum, HC_WEIGHT)
                    if p != prediction:
                        pass_test = False
                        print("Model: {}".format(model))
                        print("X: {}".format(x))
                        print("prediction: {}".format(prediction))
                        print("p: {}".format(p))
        self.assertTrue(pass_test)

def one_species_welfare_ranges_2(f, species_scores, cog_proxies, hed_proxies, hc_proxies, fff):
    cognitive_scores = wr_model.filter_proxies(species_scores, cog_proxies)
    hedonic_scores = wr_model.filter_proxies(species_scores, hed_proxies)
    human_sum_cog = wr_model.get_human_sum_2(cog_proxies, hc_proxies, HC_WEIGHT)
    human_sum_hed = wr_model.get_human_sum_2(hed_proxies, hc_proxies, HC_WEIGHT)
  
    welfare_range_list = []
    for i in range(NUM_SCENARIOS):
        welfare_range_i = wr_model.one_sim_welfare_range_2(f, cognitive_scores, hedonic_scores, human_sum_cog, human_sum_hed, i, fff)
        welfare_range_list.append(welfare_range_i)
    
    return welfare_range_list
    
class TestComplexFuctions(unittest.TestCase):
    def test_one_sim_relative_score(self):
        pass_test = True
        for x in [0,1]:
            test_scores = {}
            for proxy in cubic_proxies:
                if proxy in hc_proxies:
                    test_scores[proxy] = [x*HC_WEIGHT]*NUM_SCENARIOS
                else:
                    test_scores[proxy] = [x]*NUM_SCENARIOS
            for model in complex_models:
                for term in {"Hedonic Proxies", "Cognitive Proxies"}:
                    model_proxies = complex_models[model][term]
                    filtered_proxies = wr_model.filter_proxies(test_scores, model_proxies)
                    human_sum = wr_model.get_human_sum_2(model_proxies, hc_proxies, HC_WEIGHT)
                    prediction = x
                    for i in range(NUM_SCENARIOS):
                        p = wr_model.one_sim_relative_score(filtered_proxies, i, human_sum)
                        if p != prediction:
                            pass_test = False
                            print("Model: {}".format(model))
                            print("X: {}".format(x))
                            print("Term: {}".format(term))
                            print("Prediction: {}".format(prediction))
                            print("Actual: {}".format(p))
        self.assertTrue(pass_test)

    def test_one_sim_welare_range_2(self):
        pass_test = True
        for i, x in enumerate([(0.1, 1), (1, 0.1)]): 
            hed_test_scores = {}
            cog_test_scores = {}
            for proxy in cubic_proxies:
                if proxy in hc_proxies:
                    cog_test_scores[proxy] = [x[0]*HC_WEIGHT]*NUM_SCENARIOS
                    hed_test_scores[proxy] = [x[1]*HC_WEIGHT]*NUM_SCENARIOS
                else:
                    cog_test_scores[proxy] = [x[0]]*NUM_SCENARIOS
                    hed_test_scores[proxy] = [x[1]]*NUM_SCENARIOS

            for model in complex_models:
                human_sum_cog = wr_model.get_human_sum_2(complex_models[model]["Cognitive Proxies"], hc_proxies, HC_WEIGHT)
                human_sum_hed = wr_model.get_human_sum_2(complex_models[model]["Hedonic Proxies"], hc_proxies, HC_WEIGHT)

                cog_model_proxies = complex_models[model]["Cognitive Proxies"]
                cog_proxies = wr_model.filter_proxies(cog_test_scores, cog_model_proxies)
               
                hed_model_proxies = complex_models[model]["Hedonic Proxies"]
                hed_proxies = wr_model.filter_proxies(hed_test_scores, hed_model_proxies)
                
                cog_rel_score = wr_model.one_sim_relative_score(cog_proxies, 0, human_sum_cog)
                hed_rel_score = wr_model.one_sim_relative_score(hed_proxies, 0, human_sum_hed)

                f = complex_models[model]["Function"]
                prediction = f(cog_rel_score, hed_rel_score)

                for i in range(NUM_SCENARIOS):
                    p = wr_model.one_sim_welfare_range_2(f, cog_proxies, hed_proxies, human_sum_cog, human_sum_hed, i, 60)
                
                    if p != prediction:
                        pass_test = False
                        print("Model: {}".format(model))
                        print("X: {}".format(x))
                        print("Prediction: {}".format(prediction))
                        print("Actual: {}".format(p))
        self.assertTrue(pass_test)

    def test_one_species_wr_2(self):
        pass_test = True
        for x in {1}: 
            test_scores = {}
            for proxy in cubic_proxies:
                if proxy in hc_proxies:
                    test_scores[proxy] = [x*HC_WEIGHT]*NUM_SCENARIOS
                else:
                    test_scores[proxy] = [x]*NUM_SCENARIOS

        for model in complex_models:
            f = complex_models[model]["Function"]
            cog_proxies = complex_models[model]["Cognitive Proxies"]
            hed_proxies = complex_models[model]["Hedonic Proxies"]
            wr_list = wr_model.one_species_welfare_ranges_2(f, test_scores, cog_proxies, hed_proxies, hc_proxies, 60, NUM_SCENARIOS, HC_WEIGHT)
            expect = [1]*NUM_SCENARIOS
            if wr_list != expect:
                pass_test = False
                print("Model: ", model)
                print("Expected: ", expect[0:10])
                print("Actual: ", wr_list[0:10])
        self.assertTrue(pass_test)

class TestMultiplication(unittest.TestCase):
    models = ['Qualitative', 'High-Confidence (Simple Scoring)', \
                'Cubic', 'High-Confidence (Cubic)', \
                'Qualitative Minus Social', 'Pleasure-and-pain-centric', \
                'Higher-Lower Pleasures', 'Undiluted Experience', "Mixture"]
    def test_check_multiplication(self):
        pass_test = True
        
        for model in models: 
            
            for species in SPECIES2:
                if species != 'shrimp':
                    with open(os.path.join('sentience_estimates', '{}_psent_hv1_model.p'.format(species)), 'rb') as f_s:
                        species_psent = list(pickle.load(f_s))
                else:
                    with open(os.path.join('sentience_estimates', 'shrimp_assumed_psent.p'), 'rb') as f_s:
                        species_psent = list(pickle.load(f_s))
                with open(os.path.join('welfare_range_estimates', '{}_wr_{}_model.p'.format(species, model)), 'rb') as f_wr:
                    species_wr = list(pickle.load(f_wr)) 
                exp_adj_wrs = []
                for i in range(NUM_SCENARIOS):
                    adj_wr = max(species_psent[i]*species_wr[i],0)
                    exp_adj_wrs.append(adj_wr)
                adj_wr = wr_model.one_species_adj_wr(species, model, NUM_SCENARIOS)
                if exp_adj_wrs != adj_wr:
                    pass_test = False
        self.assertTrue(pass_test)

simple_models = {'Qualitative': {"Proxies": qual_proxies, "Function": wr_model.qual_f}, 
                'High-Confidence (Simple Scoring)': {"Proxies": hc_proxies, "Function": wr_model.qual_f}, 
                'Cubic': {"Proxies": cubic_proxies, "Function": wr_model.cubic_f}, 
                'High-Confidence (Cubic)': {"Proxies": hc_proxies, "Function": wr_model.cubic_f}, 
                'Qualitative Minus Social': {"Proxies": qms_proxies, "Function": wr_model.qms_f}, 
                'Pleasure-and-pain-centric': {"Proxies": ppc_proxies, "Function": wr_model.ppc_f}}

proxies = {'qualitative': {'List': qual_proxies_list, 'Set': qual_proxies}, 
            'cubic': {'List': cubic_proxies_list, 'Set': cubic_proxies}, 
            'qualitative minus social': {"List": qms_proxies_list, 'Set': qms_proxies},
            'pleasure-and-pain-centric': {"List": ppc_proxies_list, 'Set': ppc_proxies}, 
            'higher/lower pleasures - cognitive': {'List': hlp_cog_proxies_list, 'Set': hlp_cog_proxies},
            'higher-lower pleasures - hedonic': {'List': hlp_hed_proxies_list, 'Set': hlp_hed_proxies},
            'undiluted experience - cognitive': {'List': ue_cog_proxies_list, 'Set': ue_cog_proxies},
            'undiluted experience - hedonic': {'List': ue_hed_proxies_list, 'Set': ue_hed_proxies}}

complex_models = {'Higher-Lower Pleasures': {"Hedonic Proxies": hlp_hed_proxies, "Cognitive Proxies": hlp_cog_proxies, "Function": wr_model.hlp_f}, 
                'Undiluted Experience': {"Hedonic Proxies": ue_hed_proxies, "Cognitive Proxies": ue_cog_proxies, "Function": wr_model.ue_f}}

res2 = unittest.main(argv=[''], verbosity=3, exit=False)