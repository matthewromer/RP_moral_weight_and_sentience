"""
Module for welfare range model

Created on Tue Feb 28 11:21:50 2023

@author: matthewromer
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import squigglepy as sq


def filter_proxies(species_scores, model_proxies):
    filtered_scores = {}
    for proxy, scores_list in species_scores.items():
        if proxy in model_proxies:
            filtered_scores[proxy] = scores_list
    return filtered_scores

def get_human_sum(model_name, model_proxies, hc_proxies, HC_WEIGHT):
    human_sum = 0

    for proxy in model_proxies:
        if model_name in {"High-Confidence (Simple Scoring)", "High-Confidence (Cubic)"}:
            human_sum += 1
        else:
            if proxy in hc_proxies:
                human_sum += HC_WEIGHT
            else:
                human_sum += 1
    return human_sum

def one_sim_welfare_range(model_name, f, filtered_scores, sim_idx, fff, human_sum, HC_WEIGHT):
    welfare_sum = 0
    for scores_list in filtered_scores.values():
        score_i = scores_list[sim_idx]
        if model_name in {"High-Confidence (Simple Scoring)", "High-Confidence (Cubic)"}:
            score_i = score_i/HC_WEIGHT
        welfare_sum += score_i

    adjusted_species_sum = f(welfare_sum)
    adjusted_human_sum = f(human_sum)
    if fff is not None:
        ffr = fff/60
        welfare_range = max(0.28*(adjusted_species_sum/adjusted_human_sum)*ffr + 0.72*(adjusted_species_sum/adjusted_human_sum), 0)
    else: 
        welfare_range = max(adjusted_species_sum/adjusted_human_sum, 0)
    return welfare_range

def one_species_welfare_ranges(model_name, f, species_scores, model_proxies, hc_proxies, fff, NUM_SCENARIOS, HC_WEIGHT):
    filtered_scores = filter_proxies(species_scores, model_proxies)
    human_sum = get_human_sum(model_name, model_proxies, hc_proxies, HC_WEIGHT)
    
    welfare_range_list = []
    for i in range(NUM_SCENARIOS):
        welfare_range_i = one_sim_welfare_range(model_name, f, filtered_scores, i, fff, human_sum, HC_WEIGHT)
        welfare_range_list.append(welfare_range_i)
    
    return welfare_range_list

def plot_range_distribution(species, welfare_range_list, SCENARIO_RANGES):
    welfare_range_array = np.array(welfare_range_list)
    plt.hist(welfare_range_array, bins=20, density=True)
    plt.axvline(x=np.percentile(welfare_range_array, SCENARIO_RANGES)[1], color='k', linestyle='dashed', linewidth=1)
    plt.axvline(x=np.percentile(welfare_range_array, SCENARIO_RANGES)[11], color='k', linestyle='dashed', linewidth=1)
    plt.title("Distribution of {} Welfare Ranges".format(species))
    plt.show()
    print('-')

def one_species_summary_stats(species, welfare_range_list, SCENARIO_RANGES, to_print=False):
    welfare_range_array = np.array(welfare_range_list)
    percentiles = np.percentile(welfare_range_array, SCENARIO_RANGES)
    fifth_percentile = percentiles[1]
    ninty_fifth_percentile = percentiles[11]
    median = percentiles[6]
    stats_tuple = (fifth_percentile, median, ninty_fifth_percentile)
    if to_print:
        print("5th-percentile welfare range: {}".format(fifth_percentile))
        print("50th-percentile welfare range: {}".format(median))
        print("95th-percentile welfare range: {}".format(ninty_fifth_percentile))
    return stats_tuple

def all_species_welfare_ranges_simple_scoring(f, model_name, data, model_proxies, hc_proxies, SPECIES, SCENARIO_RANGES, HC_WEIGHT, NUM_SCENARIOS, to_plot=False):
    fifth_percentiles = []
    medians = []
    ninty_fifth_percentiles = []

    for species in SPECIES: 
        species_scores = data[species]["Scores"]
        fff_species = data[species]["FFF"]
        species_welfare_range_lst = one_species_welfare_ranges(model_name, f, species_scores, model_proxies, hc_proxies, fff_species, NUM_SCENARIOS, HC_WEIGHT)
        pickle.dump(np.array(species_welfare_range_lst), open(os.path.join('welfare_range_estimates','{}_wr_{}_model.p'.format(species, model_name)), 'wb'))
        if to_plot:
            plot_range_distribution(species, species_welfare_range_lst, SCENARIO_RANGES)
        species_stats = one_species_summary_stats(species, species_welfare_range_lst, SCENARIO_RANGES)
        fifth_percentiles.append(round(species_stats[0],3))
        medians.append(round(species_stats[1],3))
        ninty_fifth_percentiles.append(round(species_stats[2],3))

    cols = ["5th-pct", "50th-pct", "95th-pct"]
    welfare_range_stats_df = pd.DataFrame(list(zip(fifth_percentiles, medians, ninty_fifth_percentiles)), \
        columns=cols, index=SPECIES)
    welfare_range_stats_df = welfare_range_stats_df.sort_values("50th-pct", ascending=False)
    path = os.path.join('welfare_range_estimates', "WR {} Summary Statistics.csv".format(model_name))
    wfr_stats_csv = welfare_range_stats_df.to_csv(path, index_label="Species")
    print(model_name.upper())
    print(welfare_range_stats_df)
    return welfare_range_stats_df


def unknown_probs_df(SPECIES,data):
    unknown_probs = []
    for species in SPECIES:
        unknown_prob = data[species]["Unknown Prob"]
        unknown_probs.append(unknown_prob)
    cols = ['Unknown Prob.']
    unknowns_df = pd.DataFrame(unknown_probs, columns=cols, index=SPECIES)
    print(unknowns_df) 


def qual_f(welfare_sum):
    adjusted_welfare_sum = welfare_sum
    return adjusted_welfare_sum

def ss_hc_f(welfare_sum):
    adjusted_welfare_sum = welfare_sum
    return adjusted_welfare_sum

def cubic_f(welfare_sum):
    adjusted_welfare_sum = welfare_sum**3
    return adjusted_welfare_sum

def cubic_hc_f(welfare_sum):
    adjusted_welfare_sum = welfare_sum**3
    return adjusted_welfare_sum

def qms_f(welfare_sum):
    adjusted_welfare_sum = welfare_sum
    return adjusted_welfare_sum

def ppc_f(welfare_sum):
    adjusted_welfare_sum = welfare_sum
    return adjusted_welfare_sum

def get_human_sum_2(model_proxies, hc_proxies, HC_WEIGHT):
    human_sum = 0

    for proxy in model_proxies:
        if proxy in hc_proxies:
            human_sum += HC_WEIGHT
        else:
            human_sum += 1

    return human_sum

def one_sim_relative_score(filtered_scores, sim_idx, human_sum):
    welfare_sum = 0
    for scores_list in filtered_scores.values():
        score_i = scores_list[sim_idx]
        welfare_sum += score_i

    relative_score = welfare_sum/human_sum
    return relative_score

def one_sim_welfare_range_2(f, cognitive_scores, hedonic_scores, human_sum_cog, human_sum_hed, sim_idx, fff):
    cog_ratio = one_sim_relative_score(cognitive_scores, sim_idx, human_sum_cog)
    hed_ratio = one_sim_relative_score(hedonic_scores, sim_idx, human_sum_hed)
    prelim_welfare_range = f(cog_ratio, hed_ratio)
    if fff is not None:
        adjusted_welfare_range = max(0.28*prelim_welfare_range*fff/60 + 0.72*prelim_welfare_range, 0)
    else:
        adjusted_welfare_range = max(prelim_welfare_range, 0)
    return adjusted_welfare_range

def one_species_welfare_ranges_2(f, species_scores, cog_proxies, hed_proxies, hc_proxies, fff, NUM_SCENARIOS, HC_WEIGHT):
    cognitive_scores = filter_proxies(species_scores, cog_proxies)
    hedonic_scores = filter_proxies(species_scores, hed_proxies)
    human_sum_cog = get_human_sum_2(cog_proxies, hc_proxies, HC_WEIGHT)
    human_sum_hed = get_human_sum_2(hed_proxies, hc_proxies, HC_WEIGHT)
  
    welfare_range_list = []
    for i in range(NUM_SCENARIOS):
        welfare_range_i = one_sim_welfare_range_2(f, cognitive_scores, hedonic_scores, human_sum_cog, human_sum_hed, i, fff)
        welfare_range_list.append(welfare_range_i)
    
    return welfare_range_list

def all_species_welfare_ranges_2(f, model_name, data, cognitive_proxies, hedonic_proxies, hc_proxies, SPECIES, NUM_SCENARIOS, HC_WEIGHT, SCENARIO_RANGES,  to_plot=False):
    fifth_percentiles = []
    medians = []
    ninty_fifth_percentiles = []

    for species in SPECIES: 
        species_scores = data[species]["Scores"]
        fff_species = data[species]["FFF"]
        species_welfare_range_lst = one_species_welfare_ranges_2(f, species_scores, cognitive_proxies, hedonic_proxies, \
            hc_proxies, fff_species, NUM_SCENARIOS, HC_WEIGHT)
        pickle.dump(np.array(species_welfare_range_lst), open(os.path.join('welfare_range_estimates', '{}_wr_{}_model.p'.format(species, model_name)), 'wb'))
        if to_plot:
            plot_range_distribution(species, species_welfare_range_lst)
        species_stats = one_species_summary_stats(species, species_welfare_range_lst, SCENARIO_RANGES)
        fifth_percentiles.append(round(species_stats[0],3))
        medians.append(round(species_stats[1],3))
        ninty_fifth_percentiles.append(round(species_stats[2],3))

    cols = ["5th-pct", "50th-pct", "95th-pct"]
    welfare_range_stats_df = pd.DataFrame(list(zip(fifth_percentiles, medians, ninty_fifth_percentiles)), \
        columns=cols, index=SPECIES)
    welfare_range_stats_df = welfare_range_stats_df.sort_values("50th-pct", ascending=False)
    path = os.path.join('welfare_range_estimates', "WR {} - Summary Statistics.csv".format(model_name))
    wfr_stats_csv = welfare_range_stats_df.to_csv(path, index_label="Species")
    print(model_name.upper())
    print(welfare_range_stats_df)
    return welfare_range_stats_df

def hlp_f(cog_ratio, hed_ratio):
    welfare_range = cog_ratio*hed_ratio
    return welfare_range

def ue_f(cog_ratio, hed_ratio):
    if cog_ratio > 0:
        welfare_range = hed_ratio/cog_ratio
    else:
        welfare_range = hed_ratio/0.01
    return welfare_range

def mixture_one_species(model_results, species, wts, NUM_SCENARIOS):
    # Qualitative
    qual_lower = model_results['Qualitative'].loc[species]['5th-pct']
    qual_upper = model_results['Qualitative'].loc[species]['95th-pct']

    # High-Confidence (simple scoring)
    ss_hc_lower = model_results['High-Confidence Simple Scoring'].loc[species]['5th-pct']
    ss_hc_upper = model_results['High-Confidence Simple Scoring'].loc[species]['95th-pct']
    
    # Cubic
    cubic_lower = model_results['Cubic'].loc[species]['5th-pct']
    cubic_upper = model_results['Cubic'].loc[species]['95th-pct']
    
    # High-confidence (Cubic)
    hc_cubic_lower = model_results['High-Confidence Cubic'].loc[species]['5th-pct']
    hc_cubic_upper = model_results['High-Confidence Cubic'].loc[species]['95th-pct']
    
    # Qualitative Minus Social
    qms_lower = model_results['Qualitative Minus Social'].loc[species]['5th-pct']
    qms_upper = model_results['Qualitative Minus Social'].loc[species]['95th-pct']
    
    # Pleasure-and-pain-centric
    ppc_lower = model_results['Pleasure-and-pain-centric'].loc[species]['5th-pct']
    ppc_upper = model_results['Pleasure-and-pain-centric'].loc[species]['95th-pct']
    
    # Higher-Lower Pleasures
    hlp_lower = model_results['Higher-Lower Pleasures'].loc[species]['5th-pct']
    hlp_upper = model_results['Higher-Lower Pleasures'].loc[species]['95th-pct']
    
    # Undiluted Experience
    ue_lower = model_results['Undiluted Experience'].loc[species]['5th-pct']
    ue_upper = model_results['Undiluted Experience'].loc[species]['95th-pct']

    mix = sq.mixture([sq.norm(qual_lower, qual_upper, lclip=0), sq.norm(ss_hc_lower, ss_hc_upper, lclip=0), \
        sq.norm(cubic_lower, cubic_upper, lclip=0), sq.norm(hc_cubic_lower, hc_cubic_upper, lclip=0), \
        sq.norm(qms_lower, qms_upper, lclip=0), sq.norm(ppc_lower, ppc_upper, lclip=0), \
        sq.norm(hlp_lower, hlp_upper, lclip=0), sq.lognorm(ue_lower, ue_upper, lclip=0)], weights = wts) 

    dist = sq.sample(mix, n=NUM_SCENARIOS)
    pickle.dump(dist, open(os.path.join('welfare_range_estimates', '{}_wr_Mixture_model.p'.format(species)), 'wb'))

    return dist

def one_species_stats(dist, SCENARIO_RANGES):
    percentiles = np.percentile(dist, SCENARIO_RANGES)
    fifth_percentile = percentiles[1]
    ninty_fifth_percentile = percentiles[11]
    median = percentiles[6]
    stats_tuple = (fifth_percentile, median, ninty_fifth_percentile)
    return stats_tuple

def all_species_mixture(model_results, weights, SPECIES, NUM_SCENARIOS, SCENARIO_RANGES):
    fifth_percentiles = []
    medians = []
    ninty_fifth_percentiles = []
    weights_dict = {'Qualitative': weights[0], 'High-Confidence (Simple Scoring)': weights[1], \
    'Cubic': weights[2], 'High-Confidence (Cubic)': weights[3], \
    'Qualitative Minus Social': weights[4], 'Pleasure-and-pain-centric': weights[5], \
    'Higher-Lower Pleasures': weights[6], 'Undiluted Experience': weights[7]}

    for species in SPECIES: 
        mix_species = mixture_one_species(model_results, species, weights, NUM_SCENARIOS)
        species_stats = one_species_stats(mix_species, SCENARIO_RANGES)
        fifth_percentiles.append(round(species_stats[0],3))
        medians.append(round(species_stats[1],3))
        ninty_fifth_percentiles.append(round(species_stats[2],3))

    cols = ["5th-pct", "50th-pct", "95th-pct"]
    mixture_df = pd.DataFrame(list(zip(fifth_percentiles, medians, ninty_fifth_percentiles)), \
        columns=cols, index=SPECIES)
    mixture_df = mixture_df.sort_values("50th-pct", ascending=False)
    path = os.path.join('welfare_range_estimates', "WR Mixture Model - Summary Statistics.csv")
    mixture_df.to_csv(path, index_label="Species")
    print("Weights:")
    print(weights_dict)
    print("Mixture of all Models:")
    print(mixture_df)
    return mixture_df

def mixture_one_species_with_neuron_count(model_results, species, wts, neuron_counts, NUM_SCENARIOS):
    # Neuron Count
    neuron_count = neuron_counts[species]
    # Qualitative
    qual_lower = model_results['Qualitative'].loc[species]['5th-pct']
    qual_upper = model_results['Qualitative'].loc[species]['95th-pct']

    # High-Confidence (simple scoring)
    ss_hc_lower = model_results['High-Confidence Simple Scoring'].loc[species]['5th-pct']
    ss_hc_upper = model_results['High-Confidence Simple Scoring'].loc[species]['95th-pct']
    
    # Cubic
    cubic_lower = model_results['Cubic'].loc[species]['5th-pct']
    cubic_upper = model_results['Cubic'].loc[species]['95th-pct']
    
    # High-confidence (Cubic)
    hc_cubic_lower = model_results['High-Confidence Cubic'].loc[species]['5th-pct']
    hc_cubic_upper = model_results['High-Confidence Cubic'].loc[species]['95th-pct']
    
    # Qualitative Minus Social
    qms_lower = model_results['Qualitative Minus Social'].loc[species]['5th-pct']
    qms_upper = model_results['Qualitative Minus Social'].loc[species]['95th-pct']
    
    # Pleasure-and-pain-centric
    ppc_lower = model_results['Pleasure-and-pain-centric'].loc[species]['5th-pct']
    ppc_upper = model_results['Pleasure-and-pain-centric'].loc[species]['95th-pct']
    
    # Higher-Lower Pleasures
    hlp_lower = model_results['Higher-Lower Pleasures'].loc[species]['5th-pct']
    hlp_upper = model_results['Higher-Lower Pleasures'].loc[species]['95th-pct']
    
    # Undiluted Experience
    ue_lower = model_results['Undiluted Experience'].loc[species]['5th-pct']
    ue_upper = model_results['Undiluted Experience'].loc[species]['95th-pct']

    mix = sq.mixture([sq.uniform(neuron_count, neuron_count), sq.norm(qual_lower, qual_upper, lclip=0), sq.norm(ss_hc_lower, ss_hc_upper, lclip=0), \
        sq.norm(cubic_lower, cubic_upper, lclip=0), sq.norm(hc_cubic_lower, hc_cubic_upper, lclip=0), \
        sq.norm(qms_lower, qms_upper, lclip=0), sq.norm(ppc_lower, ppc_upper, lclip=0), \
        sq.norm(hlp_lower, hlp_upper, lclip=0), sq.lognorm(ue_lower, ue_upper, lclip=0)], weights = wts) 

    dist = sq.sample(mix, n=NUM_SCENARIOS)
    pickle.dump(dist, open(os.path.join('welfare_range_estimates', '{}_wr_Mixture Neuron Count_model.p'.format(species)), 'wb'))

    return dist

def all_species_mixture_with_neuron_counts(model_results, weights, SPECIES, neuron_counts, NUM_SCENARIOS, SCENARIO_RANGES ):
    fifth_percentiles = []
    medians = []
    ninty_fifth_percentiles = []
    weights_dict = {'Neuron Count': weights[0], 'Qualitative': weights[1], 'High-Confidence (Simple Scoring)': weights[2], \
    'Cubic': weights[3], 'High-Confidence (Cubic)': weights[4], \
    'Qualitative Minus Social': weights[5], 'Pleasure-and-pain-centric': weights[6], \
    'Higher-Lower Pleasures': weights[7], 'Undiluted Experience': weights[8]}

    for species in SPECIES: 
        mix_species = mixture_one_species_with_neuron_count(model_results, species, weights, neuron_counts, NUM_SCENARIOS)
        species_stats = one_species_stats(mix_species, SCENARIO_RANGES)
        fifth_percentiles.append(round(species_stats[0],3))
        medians.append(round(species_stats[1],3))
        ninty_fifth_percentiles.append(round(species_stats[2],3))

    cols = ["5th-pct", "50th-pct", "95th-pct"]
    mixture_df = pd.DataFrame(list(zip(fifth_percentiles, medians, ninty_fifth_percentiles)), \
        columns=cols, index=SPECIES)
    mixture_df = mixture_df.sort_values("50th-pct", ascending=False)
    path = os.path.join('welfare_range_estimates', "WR Mixture Model Neuron Count - Summary Statistics.csv")
    mixture_df.to_csv(path, index_label="Species")
    print("Weights:")
    print(weights_dict)
    print("Mixture of all Models:")
    print(mixture_df)
    return mixture_df

def one_species_adj_wr(species, model_name, NUM_SCENARIOS):
    if species != 'shrimp':
        with open(os.path.join('sentience_estimates', '{}_psent_hv1_model.p'.format(species)), 'rb') as f_s:
            species_psent = list(pickle.load(f_s))
    else:
        with open(os.path.join('sentience_estimates', 'shrimp_assumed_psent.p'), 'rb') as f_s:
            species_psent = list(pickle.load(f_s))

    with open(os.path.join('welfare_range_estimates', '{}_wr_{}_model.p'.format(species, model_name)), 'rb') as f_wr:
        species_wr = list(pickle.load(f_wr)) 
    species_adj_wr = []

    for i in range(NUM_SCENARIOS):
        psent_i = species_psent[i]
        wr_i = species_wr[i]
        adj_wr_i = max(psent_i*wr_i, 0)
        species_adj_wr.append(adj_wr_i)
    
    return species_adj_wr

import seaborn as sns

def all_species_adj_wr(model_name,NUM_SCENARIOS,SPECIES2,SCENARIO_RANGES,output_dir_adj):
    fifth_percentiles = []
    medians = []
    ninty_fifth_percentiles = []
    species_adj_wrs = []

    for species in SPECIES2: 
        wrs_species = one_species_adj_wr(species, model_name,NUM_SCENARIOS)
        species_adj_wrs.append(wrs_species)
        species_stats = one_species_stats(np.array(wrs_species),SCENARIO_RANGES)
        fifth_percentiles.append(round(species_stats[0],3))
        medians.append(round(species_stats[1],3))
        ninty_fifth_percentiles.append(round(species_stats[2],3))

    cols = ["5th-pct", "50th-pct", "95th-pct"]
    adj_wr_df = pd.DataFrame(list(zip(fifth_percentiles, medians, ninty_fifth_percentiles)), \
        columns=cols, index=SPECIES2)
    adj_wr_df = adj_wr_df.sort_values("50th-pct", ascending=False)
    path = os.path.join('welfare_range_estimates', "Adjusted {} Welfare Ranges - Summary Statistics.csv".format(model_name))
    adj_wr_df.to_csv(path, index_label="Species")
    
    print(model_name)
    print("P(Sentience) Adjusted Welfare Range:")
    print(adj_wr_df)
    

    for i in range(0,len(SPECIES2)):
        pickle.dump(species_adj_wrs[i], open(os.path.join(output_dir_adj,'{} {}.p'.format(SPECIES2[i], model_name)), 'wb'))    
    
    return adj_wr_df, species_adj_wrs


def box_plot_adj_wr(model_name, species_adj_wrs, SPECIES2, showfliers=True):
    sns.set_style(style='white')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=species_adj_wrs, orient='h', ax=ax, showfliers=showfliers)
    ax.set_yticks([i for i in range(len(SPECIES2))])
    ax.set_yticklabels([SPECIES2[i].capitalize() for i in range(len(SPECIES2))])
    ax.set_title("Preliminary P(Sentience)-Adjusted Welfare Ranges - {} Model".format(model_name))
    fig.savefig(os.path.join('welfare_range_estimates', "Adjusted {} Welfare Ranges - Box Plot.png".format(model_name)), dpi=300, bbox_inches='tight')

def get_quartiles(mix_adj_wrs,SPECIES2):
    percentiles = {}
    for i, species in enumerate(SPECIES2):
        percentiles[species] = {'25th': 0, '75th': 0}
        percentiles[species]['25th'] = np.percentile(mix_adj_wrs[i], 25)
        percentiles[species]['75th'] = np.percentile(mix_adj_wrs[i], 75)
    return percentiles


