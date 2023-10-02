#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to analyze impact of differing methods of handing correlated uncertainty
for the RP moral weights 
"""

#Imports 
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import squigglepy as sq
import random
import matplotlib as mpl
from heatmap_wr_ranges import heatmap_wr_ranges
from scatter_wr_ranges import scatter_wr_ranges

#Inputs
output_dir_adj = 'Sent_Adj_WR_Outputs'
models = ['Qualitative', 'High-Confidence (Simple Scoring)', \
    'Cubic', 'High-Confidence (Cubic)', \
    'Qualitative Minus Social', 'Pleasure-and-pain-centric', \
    'Higher-Lower Pleasures', 'Undiluted Experience', 'Neuron Count', \
    'Mixture', 'Mixture Neuron Count']
    
species_list = ['pigs','chickens']

num_samples = 10000
num_models = len(models)
data = {}

# Plot setup
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
mpl.rcParams['figure.dpi'] = 300
area = 2
numBins = 20
xlims = [0,2]
ylims = [0,2]
textLoc = [1.05,1.8]

# For each species, load the p(sentience)-adjusted welfare range simulation 
# data for each welfare model
for species in species_list:
    for model in models:
        data[species,model] = pickle.load(open(os.path.join(output_dir_adj,'{} {}.p'.format(species, model)), 'rb'))

r = np.corrcoef(data['pigs','Mixture Neuron Count'], data['chickens','Mixture Neuron Count'])
pig_mean = round(np.mean(data['pigs','Mixture Neuron Count']),4)
chicken_mean = round(np.mean(data['chickens','Mixture Neuron Count']),4)
correlation_coeff = round(r[0,1],4)

# Plot uncorrelated reults for pigs vs. chickens 
scatter_wr_ranges(data['pigs','Mixture Neuron Count'], \
                  data['chickens','Mixture Neuron Count'],\
                  'Pigs','Chickens',xlims,ylims,pig_mean,chicken_mean,\
                  correlation_coeff,'Independent Sampling from Mixture Model',\
                  textLoc,area=area) 

heatmap_wr_ranges(data['pigs','Mixture Neuron Count'], \
                  data['chickens','Mixture Neuron Count'],\
                  'Pigs','Chickens',xlims,ylims,pig_mean,chicken_mean,\
                  correlation_coeff,'Independent Sampling from Mixture Model',\
                  textLoc,numBins=numBins,printEn=True)
        

# Order results and plot paired results 
dataSort = data
for species in species_list:
    for model in models:
        dataSort[species,model].sort()

r = np.corrcoef(dataSort['pigs','Mixture Neuron Count'], dataSort['chickens','Mixture Neuron Count'])
pig_mean = round(np.mean(dataSort['pigs','Mixture Neuron Count']),4)
chicken_mean = round(np.mean(dataSort['chickens','Mixture Neuron Count']),4)
correlation_coeff = round(r[0,1],4)

scatter_wr_ranges(dataSort['pigs','Mixture Neuron Count'], \
                  dataSort['chickens','Mixture Neuron Count'],\
                  'Pigs','Chickens',xlims,ylims,pig_mean,chicken_mean,\
                  correlation_coeff,'Sampling from Ordered Data',\
                  textLoc,area=area) 

heatmap_wr_ranges(dataSort['pigs','Mixture Neuron Count'], \
                  dataSort['chickens','Mixture Neuron Count'],\
                  'Pigs','Chickens',xlims,ylims,pig_mean,chicken_mean,\
                  correlation_coeff,'Sampling from Ordered Data',\
                  textLoc,numBins=numBins,printEn=True)
    
   

# Sample from distribution for each welfare model and plot paired results 
samples_per_model = int(num_samples/num_models)
data_per_model = {}
for species in species_list:
    data_per_model[species] = []
    for model in models:
        data_per_model[species].extend(random.sample(data[species,model],samples_per_model))

r = np.corrcoef(data_per_model['pigs'], data_per_model['chickens'])
pig_mean = round(np.mean(data_per_model['pigs']),4)
chicken_mean = round(np.mean(data_per_model['chickens']),4)
correlation_coeff = round(r[0,1],4)

scatter_wr_ranges(data_per_model['pigs'],data_per_model['chickens'],\
                  'Pigs','Chickens',xlims,ylims,pig_mean,chicken_mean,\
                  correlation_coeff,'Paired Sampling from Constituent Models',\
                  textLoc,area=area)

heatmap_wr_ranges(data_per_model['pigs'],data_per_model['chickens'],\
                  'Pigs','Chickens',xlims,ylims,pig_mean,chicken_mean,\
                  correlation_coeff,'Paired Sampling from Constituent Models',\
                  textLoc,numBins=numBins,printEn=True)