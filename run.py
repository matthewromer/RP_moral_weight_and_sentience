# Rethink Priorities Moral Weights Computation - All-In-One Version:
#
# Updated by Matt Romer from the original RP code
# 

################## Setup ##################
import os
import csv
import platform
import pandas as pd
import warnings
import user_inputs
import pickle
import re
import copy
from sent_simulate_fn_file import *
from wr_simulate_fn_file import *
import matplotlib.pyplot as plt
import test_simulations
import s_model
import sim_utils

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

S_PARAMS = {'N_SCENARIOS': 200, 'UPDATE_EVERY': 1000, "WEIGHT_NOS": s_weight_nos, "HC_WEIGHT": s_hc_weight}

## Welfare Ranges 
print("For the WELFARE RANGES...")
wr_unknowns   = copy.deepcopy(wr_default_unknowns)
wr_weight_nos = "Yes"
wr_hc_weight  = 5

WR_PARAMS = {'N_SCENARIOS': 200, 'UPDATE_EVERY': 1000, "WEIGHT_NOS": wr_weight_nos, "HC_WEIGHT": wr_hc_weight}



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
