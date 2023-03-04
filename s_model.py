"""
Module for sentience model

Created on Tue Feb 28 11:21:50 2023

@author: matthewromer
"""

import os
import pickle
import squigglepy as sq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def filter_proxies(species_scores, model_proxies):
    filtered_scores = {}
    for proxy, scores_list in species_scores.items():
        if proxy in model_proxies:
            filtered_scores[proxy] = scores_list
    return filtered_scores

def get_sum(filtered_scores, sim_idx):
    sum = 0
    for scores_list in filtered_scores.values():
        score_i = scores_list[sim_idx]
        sum += score_i  
    return sum

def plot_range_distribution(species, ps_sentience_list, SCENARIO_RANGES):
    ps_sentience_array = np.array(ps_sentience_list)
    plt.hist(ps_sentience_array, bins=20, density=True)
    plt.axvline(x=np.percentile(ps_sentience_array, SCENARIO_RANGES)[1], color='k', linestyle='dashed', linewidth=1)
    plt.axvline(x=np.percentile(ps_sentience_array, SCENARIO_RANGES)[11], color='k', linestyle='dashed', linewidth=1)
    plt.title("Distribution of {} Probabilities of Sentience".format(species))
    plt.show()
    print('-')

def one_species_summary_stats(species, ps_sentience_list, SCENARIO_RANGES, to_print=False):
    ps_sentience_array = np.array(ps_sentience_list)
    percentiles = np.percentile(ps_sentience_array, SCENARIO_RANGES)
    fifth_percentile = percentiles[1]
    ninty_fifth_percentile = percentiles[11]
    median = percentiles[6]
    stats_tuple = (fifth_percentile, median, ninty_fifth_percentile)
    if to_print:
        print(species)
        print("5th-percentile probability of sentience: {}".format(fifth_percentile))
        print("50th-percentile probability of sentience: {}".format(median))
        print("95th-percentile probability of sentience: {}".format(ninty_fifth_percentile))
    return stats_tuple

def unknown_probs_df(SPECIES,data):
    unknown_probs = []
    for species in SPECIES:
        unknown_prob = data[species]["Unknown Prob"]
        unknown_probs.append(unknown_prob)
    cols = ['Unknown Prob.']
    unknowns_df = pd.DataFrame(unknown_probs, columns=cols, index=SPECIES)
    print(unknowns_df) 
    return unknowns_df
    
def simulate_priors(priors_distributions,species_lst,NUM_SCENARIOS):
    priors = {}
    for species in species_lst:
        models = []
        for person in ['Daniela', 'Marcus', 'Peter']:
            dist_type = priors_distributions[person][species]['dist_type']
            lower = priors_distributions[person][species]['lower']
            upper = priors_distributions[person][species]['upper']
            lclip = priors_distributions[person][species]['lclip']
            rclip = priors_distributions[person][species]['rclip']
            if dist_type == 'lognormal':
                model = sq.lognorm(x=lower, y=upper, credibility = 90, lclip=lclip, rclip=rclip)
            elif dist_type == 'normal':
                model = sq.norm(x=lower, y=upper, credibility = 90, lclip=lclip, rclip=rclip)
            models.append(model)
        prior_lst = sq.sample(sq.mixture(models, [1/3, 1/3, 1/3]), NUM_SCENARIOS)
        priors[species] = prior_lst
    return priors


def shrimp_probability_sentience(priors,priors_distributions,NUM_SCENARIOS):
    models = []
    for person in ['Daniela', 'Marcus', 'Peter']:
        dist_type = priors_distributions[person]['crabs']['dist_type']
        lower = priors_distributions[person]['crabs']['lower']
        upper = priors_distributions[person]['crabs']['upper']
        lclip = priors_distributions[person]['crabs']['lclip']
        rclip = priors_distributions[person]['crabs']['rclip']
        if dist_type == 'lognormal':
            model = sq.lognorm(x=lower, y=upper, credibility = 90, lclip=lclip, rclip=rclip)
        elif dist_type == 'normal':
            model = sq.norm(x=lower, y=upper, credibility = 90, lclip=lclip, rclip=rclip)
        models.append(model)
    shrimp_prior_lst = sq.sample(sq.mixture(models, [1/3, 1/3, 1/3]), NUM_SCENARIOS)
    pickle.dump(shrimp_prior_lst, open(os.path.join('sentience_estimates', 'shrimp_assumed_psent.p'), 'wb'))
    return shrimp_prior_lst      

def one_sim_p_sentience_priors_based(f, priors, species, model_proxies, species_scores, sometimes_operates_scores, sim_idx):
    s_filtered_scores = filter_proxies(species_scores, model_proxies)
    so_filtered_scores = filter_proxies(sometimes_operates_scores, model_proxies)
    species_sum = get_sum(s_filtered_scores, sim_idx)
    sometimes_operates_sum = get_sum(so_filtered_scores, sim_idx)
    prior = priors[species][sim_idx]

    p_sentience = min(prior*(species_sum/sometimes_operates_sum)**0.25,0.99)
    p_sentience = max(p_sentience, 0)
    return p_sentience

def one_species_ps_sentience_priors_based(f, priors, species, species_scores, model_proxies, model_name, NUM_SCENARIOS, sometimes_operates_scores): 
    ps_sentience_list = []
    for i in range(NUM_SCENARIOS):
        p_sentience_i = one_sim_p_sentience_priors_based(f, priors, species, model_proxies, species_scores, sometimes_operates_scores, i)
        ps_sentience_list.append(p_sentience_i)

    if model_name == "#1_high value proxies":
        pickle.dump(ps_sentience_list, open(os.path.join('sentience_estimates', '{}_psent_hv1_model.p'.format(species)), 'wb'))
    
    return ps_sentience_list

def print_priors(priors, species_lst, to_plot):
    fifth_percentiles = []
    medians = []
    ninty_fifth_percentiles = []
    species_sent_data = []

    for species in species_lst: 
        fifth_percentiles.append(round(np.percentile(priors[species], 5),3))
        medians.append(round(np.percentile(priors[species], 50),3))
        ninty_fifth_percentiles.append(round(np.percentile(priors[species], 95),3))

    cols = ["5th-pct", "50th-pct", "95th-pct"]
    p_sentience_stats_df = pd.DataFrame(list(zip(fifth_percentiles, medians, ninty_fifth_percentiles)), \
        columns=cols, index=species_lst)
    p_sentience_stats_df = p_sentience_stats_df.sort_values("50th-pct", ascending=False)
    path = os.path.join('sentience_estimates', "Priors Sentience Summary Statistics.csv")
    p_sentience_stats_df.to_csv(path, index_label="Species")

    return p_sentience_stats_df

def all_species_p_sentience_priors_based(f, priors, model_name, data, model_proxies, species_lst, NUM_SCENARIOS, sometimes_operates_scores, SCENARIO_RANGES, to_plot=False):
    fifth_percentiles = []
    medians = []
    ninty_fifth_percentiles = []
    species_sent_data = []

    for species in species_lst: 
        species_scores = data[species]["Scores"]
        species_p_sentience_lst = one_species_ps_sentience_priors_based(f, priors, species, species_scores, model_proxies, model_name, NUM_SCENARIOS, sometimes_operates_scores)
        if to_plot:
            plot_range_distribution(species, species_p_sentience_lst)
        species_sent_data.append(species_p_sentience_lst)
        species_stats = one_species_summary_stats(species, species_p_sentience_lst, SCENARIO_RANGES)
        fifth_percentiles.append(round(species_stats[0],3))
        medians.append(round(species_stats[1],3))
        ninty_fifth_percentiles.append(round(species_stats[2],3))

    cols = ["5th-pct", "50th-pct", "95th-pct"]
    p_sentience_stats_df = pd.DataFrame(list(zip(fifth_percentiles, medians, ninty_fifth_percentiles)), \
        columns=cols, index=species_lst)
    p_sentience_stats_df = p_sentience_stats_df.sort_values("50th-pct", ascending=False)
    path = os.path.join('sentience_estimates', "Sent {} Summary Statistics.csv".format(model_name))

    p_sentience_stats_df.to_csv(path, index_label="Species")

    return p_sentience_stats_df, species_sent_data

def box_plot_adj_wr(model_name, species_adj_wrs, species_caps, showfliers=True):
    sns.set_style(style='white')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=species_adj_wrs, orient='h', ax=ax, showfliers=showfliers)
    ax.set_yticklabels(species_caps)
    ax.set_title("Probability of Sentience - {} Model".format(model_name))
    fig.savefig(os.path.join('sentience_estimates', "{} Probability of Sentience  - Box Plot.png".format(model_name)), dpi=300, bbox_inches='tight')
    
# simple scoring function
def ss_f(sentience_sum):
    adjusted_sentience_sum = sentience_sum
    return adjusted_sentience_sum    

#1_High-Value Proxies Function
def hv1_f(sentience_sum):
    adjusted_sentience_sum = sentience_sum
    return adjusted_sentience_sum

def get_overlap(criterion_proxies, species_scores):
    criterion_scores = {}
    for proxy in criterion_proxies:
        criterion_scores[proxy] = species_scores[proxy]

    return criterion_scores

def one_species_birch_sum(birch_proxies, species_scores, NUM_SCENARIOS):
    birch_sums = {}
    for criterion in birch_proxies.columns:
        birch_sums[criterion] = []
        criterion_proxies = birch_proxies[criterion].dropna().values.tolist()
        criterion_scores = get_overlap(criterion_proxies, species_scores)

        for i in range(NUM_SCENARIOS):
            criterion_sum_i = 0
            for proxy in criterion_scores:
                crit_score_i = criterion_scores[proxy][i]
                criterion_sum_i += crit_score_i
        
            birch_sums[criterion].append(criterion_sum_i)

    return birch_sums

def one_species_birch_est(priors, birch_proxies, species_scores, species, sometimes_operates_scores,NUM_SCENARIOS,SCENARIO_RANGES):
    so_birch_sums = one_species_birch_sum(birch_proxies, sometimes_operates_scores,NUM_SCENARIOS)
    species_birch_sums = one_species_birch_sum(birch_proxies, species_scores,NUM_SCENARIOS)
    
    species_birch_scores = {}

    fifth_percentiles = []
    medians = []
    ninty_fifth_percentiles = []
    criteria = ['overall'] 

    overall_scores = []
    for i in range(NUM_SCENARIOS):
        prior = priors[species][i]
        score_i = 0
        so_score_i = 0
        for c in species_birch_sums:
            score_i += species_birch_sums[c][i]
            so_score_i += so_birch_sums[c][i]
        posterior = min(prior*(score_i/(so_score_i))**0.25,0.99)
        posterior = max(posterior,0)
        overall_scores.append(posterior)
    overall_birch_array = np.array(overall_scores)
    overall_birch_percentiles = np.percentile(overall_birch_array, SCENARIO_RANGES)

    fifth_percentile = overall_birch_percentiles[1]
    ninty_fifth_percentile = overall_birch_percentiles[11]
    median = overall_birch_percentiles[6]

    fifth_percentiles.append(round(fifth_percentile,3))
    medians.append(round(median,3))
    ninty_fifth_percentiles.append(round(ninty_fifth_percentile,3))

    for criterion in species_birch_sums:
        criteria.append(criterion)
        species_birch_scores[criterion] = []
        crit_list = species_birch_sums[criterion]
        for i in range(len(crit_list)):
            species_birch_scores[criterion].append(species_birch_sums[criterion][i])
        
        criterion_birch_array = np.array(species_birch_scores[criterion])
        criterion_birch_percentiles = np.percentile(criterion_birch_array, SCENARIO_RANGES)

        fifth_percentile = criterion_birch_percentiles[1]
        ninty_fifth_percentile = criterion_birch_percentiles[11]
        median = criterion_birch_percentiles[6]

        fifth_percentiles.append(round(fifth_percentile,3))
        medians.append(round(median,3))
        ninty_fifth_percentiles.append(round(ninty_fifth_percentile,3))

    cols = ["5th-pct", "50th-pct", "95th-pct"]
    crit = species_birch_scores.keys()
    birch_sentience_stats_df = pd.DataFrame(list(zip(fifth_percentiles, medians, ninty_fifth_percentiles)), \
        columns=cols, index=criteria)
        
    return birch_sentience_stats_df

def all_species_birch_est(species_lst,data,priors,birch_proxies,sometimes_operates_scores,NUM_SCENARIOS,SCENARIO_RANGES):
    fifth_pctiles = []
    med_pctiles = []
    ninetyfifth_pctiles = []
    species_sent_data = []

    for species in species_lst:
        species_scores = data[species]['Scores']
        spec_birch_est = one_species_birch_est(priors, birch_proxies, species_scores, species, sometimes_operates_scores,NUM_SCENARIOS,SCENARIO_RANGES)
        path = os.path.join('birch_estimates', "{}_birch_estimates.csv".format(species))
        spec_birch_est.to_csv(path, index_label="Species")
        spec_overall = spec_birch_est.loc[['overall']]
        species_sent_data.append(spec_overall)
        fifth_pctiles.append(spec_overall['5th-pct'].values[0])
        med_pctiles.append(spec_overall['50th-pct'].values[0])
        ninetyfifth_pctiles.append(spec_overall['95th-pct'].values[0])
        
    birch_overalls = pd.DataFrame(list(zip(fifth_pctiles, med_pctiles, ninetyfifth_pctiles)), \
        columns=['5th-pct', '50th-pct', '95th-pct'],index=species_lst)
    birch_overalls = birch_overalls.sort_values("50th-pct", ascending=False)
    path = os.path.join('sentience_estimates', "Sent {} Summary Statistics.csv".format("Birch Model"))
    birch_overalls.to_csv(path, index_label="Species")
    return birch_overalls, species_sent_data