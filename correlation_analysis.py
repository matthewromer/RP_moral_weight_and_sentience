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
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('P(sentience)-Adjusted Welfare Range of Pigs')
ax.set_ylabel('P(sentience)-Adjusted Welfare Range of Chickens')
plt.text(1.2,1.8, 'Mean (Pigs) = {} \nMean (Chickens) = {} \nCorrelation = {}'.format(pig_mean,chicken_mean,correlation_coeff))
plt.scatter(data['pigs','Mixture Neuron Count'], data['chickens','Mixture Neuron Count'], s=area)
ax.set_xlim([0, 2])
ax.set_ylim([0, 2])
plt.title('Welfare Ranges (Independent Sampling from Mixture Model)')
plt.grid()
plt.show()        

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('gray')
plt.hist2d(data['pigs','Mixture Neuron Count'], data['chickens','Mixture Neuron Count'], bins=numBins,norm=mpl.colors.LogNorm())
ax.set_xlabel('P(sentience)-Adjusted Welfare Range of Pigs')
ax.set_ylabel('P(sentience)-Adjusted Welfare Range of Chickens')
plt.text(1.3,1.8, 'Mean (Pigs) = {} \nMean (Chickens) = {} \nCorrelation = {}'.format(pig_mean,chicken_mean,correlation_coeff))
ax.set_xlim([0, 2])
ax.set_ylim([0, 2])
plt.title('Welfare Ranges (Independent Sampling from Mixture Model)')
plt.grid()
plt.show()
        

# Order results and plot paired results 
dataSort = data
for species in species_list:
    for model in models:
        dataSort[species,model].sort()

r = np.corrcoef(dataSort['pigs','Mixture Neuron Count'], dataSort['chickens','Mixture Neuron Count'])
pig_mean = round(np.mean(dataSort['pigs','Mixture Neuron Count']),4)
chicken_mean = round(np.mean(dataSort['chickens','Mixture Neuron Count']),4)
correlation_coeff = round(r[0,1],4)

        
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('P(sentience)-Adjusted Welfare Range of Pigs')
ax.set_ylabel('P(sentience)-Adjusted Welfare Range of Chickens')
plt.text(1.2,1.8, 'Mean (Pigs) = {} \nMean (Chickens) = {} \nCorrelation = {}'.format(pig_mean,chicken_mean,correlation_coeff))
plt.scatter(dataSort['pigs','Mixture Neuron Count'], dataSort['chickens','Mixture Neuron Count'], s=area)
plt.title('Welfare Ranges (Sampling from Ordered Data)')
ax.set_xlim([0, 2])
ax.set_ylim([0, 2])
plt.grid()
plt.show()        

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('gray')
plt.hist2d(dataSort['pigs','Mixture Neuron Count'], dataSort['chickens','Mixture Neuron Count'], bins=numBins,norm=mpl.colors.LogNorm())
ax.set_xlabel('P(sentience)-Adjusted Welfare Range of Pigs')
ax.set_ylabel('P(sentience)-Adjusted Welfare Range of Chickens')
plt.text(1.3,1.8, 'Mean (Pigs) = {} \nMean (Chickens) = {} \nCorrelation = {}'.format(pig_mean,chicken_mean,correlation_coeff))
ax.set_xlim([0, 2])
ax.set_ylim([0, 2])
plt.title('Welfare Ranges (Sampling from Ordered Data)')
plt.grid()
plt.show()

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

        
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('P(sentience)-Adjusted Welfare Range of Pigs')
ax.set_ylabel('P(sentience)-Adjusted Welfare Range of Chickens')
plt.text(1.2,1.8, 'Mean (Pigs) = {} \nMean (Chickens) = {} \nCorrelation = {}'.format(pig_mean,chicken_mean,correlation_coeff))
plt.scatter(data_per_model['pigs'], data_per_model['chickens'], s=area)
ax.set_xlim([0, 2])
ax.set_ylim([0, 2])
plt.title('Welfare Ranges (Paired Sampling from Constituent Models)')
plt.grid()
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('gray')
h = plt.hist2d(data_per_model['pigs'], data_per_model['chickens'], density=True,bins=numBins,norm=mpl.colors.LogNorm())
ax.set_xlabel('P(sentience)-Adjusted Welfare Range of Pigs')
ax.set_ylabel('P(sentience)-Adjusted Welfare Range of Chickens')
plt.text(1.3,1.8, 'Mean (Pigs) = {} \nMean (Chickens) = {} \nCorrelation = {}'.format(pig_mean,chicken_mean,correlation_coeff))
cbar = fig.colorbar(h[3], ax=ax)
cbar.set_label('Density', rotation=270)
ax.set_xlim([0, 2])
ax.set_ylim([0, 2])
plt.title('Welfare Ranges (Paired Sampling from Constituent Models)')
plt.grid()
plt.show()
