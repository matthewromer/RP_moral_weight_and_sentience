#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to analyze impact of differing methods of handing correlated uncertainty
for the RP moral weights
"""

# Imports
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from heatmap_wr_ranges import heatmap_wr_ranges
from scatter_wr_ranges import scatter_wr_ranges
from compute_y import compute_y
import seaborn as sns
from compute_summary_stats_arr import compute_summary_stats_arr
from adj_wr_correlation import adj_wr_correlation

# Inputs
output_dir_adj = 'Sent_Adj_WR_Outputs'
output_dir_unadj = 'Sent_Adj_WR_Outputs2'
models = ['Qualitative', 'High-Confidence (Simple Scoring)',
          'Cubic', 'High-Confidence (Cubic)',
          'Qualitative Minus Social', 'Pleasure-and-pain-centric',
          'Higher-Lower Pleasures', 'Undiluted Experience', 'Neuron Count',
          'Mixture Neuron Count']

species_list = ['pigs', 'chickens']

num_samples = 10000
num_models = len(models)
data = {}
data_unadj = {}


# Plot setup
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
mpl.rcParams['figure.dpi'] = 300
area = 2
num_bins = 100
xlims = [0, 2]
ylims = [0, 2]
text_loc = [1.05, 1.8]
my_pal = {"lightsteelblue", "lightcoral", "thistle", "navajowhite"}

# For each species, load the welfare range simulation
# data for each welfare model
for species in species_list:
    for model in models:
        data_unadj[species, model] = pickle.load(open(os.path.join(
            output_dir_unadj, '{} {}.p'.format(species, model)), 'rb'))

# For each species, load the p(sentience)-adjusted welfare range simulation
# data for each welfare model
for species in species_list:
    for model in models:
        data[species, model] = \
            pickle.load(open(os.path.join(output_dir_adj,
                        '{} {}.p'.format(species, model)), 'rb'))

r = np.corrcoef(data['pigs', 'Mixture Neuron Count'],
                data['chickens', 'Mixture Neuron Count'])
pig_mean = round(np.mean(data['pigs', 'Mixture Neuron Count']), 4)
chicken_mean = round(np.mean(data['chickens', 'Mixture Neuron Count']), 4)
correlation_coeff = round(r[0, 1], 4)

compute_summary_stats_arr(data['pigs', 'Mixture Neuron Count'],
                          print_en=True, name="Pigs Mixture Neuron Count")
compute_summary_stats_arr(data['chickens', 'Mixture Neuron Count'],
                          print_en=True, name="Chickens Mixture Neuron Count")

# Plot uncorrelated reults for pigs vs. chickens
scatter_wr_ranges(data['pigs', 'Mixture Neuron Count'],
                  data['chickens', 'Mixture Neuron Count'],
                  'Pigs', 'Chickens', xlims, ylims, pig_mean, chicken_mean,
                  correlation_coeff, 'Independent Sampling from Mixture Model',
                  text_loc, area=area, print_en=False)

heatmap_wr_ranges(data['pigs', 'Mixture Neuron Count'],
                  data['chickens', 'Mixture Neuron Count'],
                  'Pigs', 'Chickens', xlims, ylims, pig_mean, chicken_mean,
                  correlation_coeff, 'Independent Sampling from Mixture Model',
                  text_loc, num_bins=num_bins, print_en=True)

y1_raw, y2_raw = compute_y(np.array(data['pigs', 'Mixture Neuron Count']),
                           np.array(data['chickens', 'Mixture Neuron Count']))

n = int(num_samples*0.025)
m = int(num_samples*0.975)-n

indx = sorted(np.argsort(y1_raw)[n:])
y1_trimmed = y1_raw[indx]
indx2 = sorted(np.argsort(y1_trimmed)[:m])
y1_trimmed = y1_trimmed[indx2]

indx = sorted(np.argsort(y2_raw)[n:])
y2_trimmed = y2_raw[indx]
indx2 = sorted(np.argsort(y1_trimmed)[:m])
y2_trimmed = y2_trimmed[indx2]


# Order results and plot paired results
data_sort = data
for species in species_list:
    for model in models:
        data_sort[species, model].sort()

r = np.corrcoef(data_sort['pigs', 'Mixture Neuron Count'],
                data_sort['chickens', 'Mixture Neuron Count'])
pig_mean = round(np.mean(data_sort['pigs', 'Mixture Neuron Count']), 4)
chicken_mean = round(np.mean(data_sort['chickens', 'Mixture Neuron Count']), 4)
correlation_coeff = round(r[0, 1], 4)

compute_summary_stats_arr(data_sort['pigs', 'Mixture Neuron Count'],
                          print_en=True,
                          name="Pigs Mixture Neuron Count - Ordered")
compute_summary_stats_arr(data_sort['chickens', 'Mixture Neuron Count'],
                          print_en=True,
                          name="Chickens Mixture Neuron Count - Ordered")

scatter_wr_ranges(data_sort['pigs', 'Mixture Neuron Count'],
                  data_sort['chickens', 'Mixture Neuron Count'],
                  'Pigs', 'Chickens', xlims, ylims, pig_mean, chicken_mean,
                  correlation_coeff, 'Sampling from Ordered Data',
                  text_loc, area=area, print_en=False)

heatmap_wr_ranges(data_sort['pigs', 'Mixture Neuron Count'],
                  data_sort['chickens', 'Mixture Neuron Count'],
                  'Pigs', 'Chickens', xlims, ylims, pig_mean, chicken_mean,
                  correlation_coeff, 'Sampling from Ordered Data',
                  text_loc, num_bins=num_bins, print_en=True)

y1_ordered, y2_ordered = compute_y(
                    np.array(data_sort['pigs', 'Mixture Neuron Count']),
                    np.array(data_sort['chickens', 'Mixture Neuron Count']))


# Sample from distribution for each welfare model and plot paired results
samples_per_model = int(num_samples/num_models)
data_per_model = {}
for species in species_list:
    data_per_model[species] = []
    for model in models:
        data_per_model[species].extend(
            np.random.choice(data_unadj[species, model], samples_per_model))
    data_per_model[species] = \
        adj_wr_correlation(species, data_per_model[species], num_samples)

r = np.corrcoef(data_per_model['pigs'], data_per_model['chickens'])
pig_mean = round(np.mean(data_per_model['pigs']), 4)
chicken_mean = round(np.mean(data_per_model['chickens']), 4)
correlation_coeff = round(r[0, 1], 4)

compute_summary_stats_arr(data_per_model['pigs'], print_en=True,
                          name="Pigs - Paired from Component")
compute_summary_stats_arr(data_per_model['chickens'], print_en=True,
                          name="Chickens - Paired from Component")

scatter_wr_ranges(data_per_model['pigs'], data_per_model['chickens'],
                  'Pigs', 'Chickens', xlims, ylims, pig_mean, chicken_mean,
                  correlation_coeff, 'Paired Sampling from Constituent Models',
                  text_loc, area=area, print_en=False)

heatmap_wr_ranges(data_per_model['pigs'], data_per_model['chickens'],
                  'Pigs', 'Chickens', xlims, ylims, pig_mean, chicken_mean,
                  correlation_coeff, 'Paired Sampling from Constituent Models',
                  text_loc, num_bins=num_bins, print_en=True)


chicken_wr = np.array(data_per_model['chickens'])
pig_wr = np.array(data_per_model['pigs'])

y1_paired, y2_paired = compute_y(np.array(data_per_model['pigs']),
                                 np.array(data_per_model['chickens']))


# Results for shifting from chicken to pork (Box Plot)
box_data = [y1_raw, y1_trimmed, y1_ordered, y1_paired]
fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
sns.boxplot(data=box_data, orient='h', ax=ax, showfliers=False, palette=my_pal)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(["Uncorrelated", "Trimming\nTop/Bottom 2.5%", "Ordering",
                    "Generating Samples\nPair-Wise from\nConstituent Models"])
ax.set_xlabel("Weighted Days of Suffering Averted")
ax.set_title("Suffering Averted by Switching from Chicken to Pork")
fig.savefig('./Plots/Switching_Suffering.png')

# Results for shifting from chicken to pork (Box Plot)
box_data = [y2_raw, y2_trimmed, y2_ordered, y2_paired]
fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
sns.boxplot(data=box_data, orient='h', ax=ax, showfliers=False, palette=my_pal)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(["Uncorrelated", "Trimming\nTop/Bottom 2.5%", "Ordering",
                    "Generating Samples\nPair-Wise from\nConstituent Models"])
ax.set_xlabel("Weighted Days of Suffering Caused")
ax.set_title("Suffering Caused by Eating Chicken and Pork")
fig.savefig('./Plots/Eating_Both_Suffering.png')


# For each species, load the p(sentience)-adjusted welfare range simulation
# data for each welfare model
species_list = ['chickens', 'carp']
for species in species_list:
    for model in models:
        data[species, model] = \
            pickle.load(open(os.path.join(output_dir_adj,
                        '{} {}.p'.format(species, model)), 'rb'))


# Sample from distribution for each welfare model and plot paired results
data_per_model = {}
for species in species_list:
    data_per_model[species] = []
    for model in models:
        data_per_model[species].extend(
            np.random.choice(data[species, model], samples_per_model))

r = np.corrcoef(data_per_model['chickens'], data_per_model['carp'])
pig_mean = round(np.mean(data_per_model['chickens']), 4)
chicken_mean = round(np.mean(data_per_model['carp']), 4)
correlation_coeff = round(r[0, 1], 4)

text_loc = [3*0.525, 3*0.9]
heatmap_wr_ranges(data_per_model['chickens'], data_per_model['carp'],
                  'Chickens', 'Carp', xlims, ylims, pig_mean, chicken_mean,
                  correlation_coeff,
                  'Paired Sampling from Constituent Models - Carp',
                  text_loc, num_bins=num_bins, print_en=True, lims=[0, 3])

# For each species, load the p(sentience)-adjusted welfare range simulation
# data for each welfare model
species_list = ['chickens', 'shrimp']
for species in species_list:
    for model in models:
        data[species, model] = pickle.load(open(os.path.join(output_dir_adj,
                                           '{} {}.p'.format(species, model)),
                                           'rb'))


# Sample from distribution for each welfare model and plot paired results
data_per_model = {}
for species in species_list:
    data_per_model[species] = []
    for model in models:
        data_per_model[species].extend(np.random.choice(
            data[species, model], samples_per_model))

compute_summary_stats_arr(data_per_model['shrimp'], print_en=True,
                          name="Shrimp - Paired from Component")


r = np.corrcoef(data_per_model['chickens'], data_per_model['shrimp'])
pig_mean = round(np.mean(data_per_model['chickens']), 4)
chicken_mean = round(np.mean(data_per_model['shrimp']), 4)
correlation_coeff = round(r[0, 1], 4)

text_loc = [5*0.525, 5*0.9]
heatmap_wr_ranges(data_per_model['chickens'], data_per_model['shrimp'],
                  'Chickens', 'Shrimp', xlims, ylims, pig_mean, chicken_mean,
                  correlation_coeff, 'Paired Sampling from\
                  Constituent Models - Shrimp', text_loc,
                  num_bins=num_bins, print_en=True, lims=[0, 5])
