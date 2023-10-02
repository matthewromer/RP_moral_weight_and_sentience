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
import matplotlib.pyplot as plt
import random
import matplotlib as mpl
from heatmap_wr_ranges import heatmap_wr_ranges
from scatter_wr_ranges import scatter_wr_ranges
from compute_y import compute_y
import seaborn as sns

#Inputs
output_dir_adj = 'Sent_Adj_WR_Outputs2'
models = ['Qualitative', 'High-Confidence (Simple Scoring)', \
    'Cubic', 'High-Confidence (Cubic)', \
    'Qualitative Minus Social', 'Pleasure-and-pain-centric', \
    'Higher-Lower Pleasures', 'Undiluted Experience', 'Neuron Count', \
    'Mixture Neuron Count']
    
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
my_pal = {"lightsteelblue","lightcoral","palegreen","mediumpurple"}

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
    
y1Raw, y2Raw = compute_y(np.array(data['pigs','Mixture Neuron Count']),\
                                 np.array(data['chickens','Mixture Neuron Count']))
    

n = int(num_samples*0.025)
m = int(num_samples*0.975)-n

pigsMixtureNeuronArr = np.array(data['pigs','Mixture Neuron Count'])
indx = sorted(np.argsort(pigsMixtureNeuronArr)[n:])
pigsMixtureNeuronTrimmed = pigsMixtureNeuronArr[indx]
indx2 = sorted(np.argsort(pigsMixtureNeuronTrimmed)[:m])
pigsMixtureNeuronTrimmed = pigsMixtureNeuronTrimmed[indx2]

chickensMixtureNeuronArr = np.array(data['chickens','Mixture Neuron Count'])
indx = sorted(np.argsort(chickensMixtureNeuronArr)[n:])
chickensMixtureNeuronTrimmed = chickensMixtureNeuronArr[indx]
indx2 = sorted(np.argsort(chickensMixtureNeuronTrimmed)[:m])
chickensMixtureNeuronTrimmed = chickensMixtureNeuronTrimmed[indx2]

y1Trimmed, y2Trimmed = compute_y(pigsMixtureNeuronTrimmed,chickensMixtureNeuronTrimmed)


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
    
y1Ordered, y2Ordered = compute_y(np.array(dataSort['pigs','Mixture Neuron Count']),\
                                 np.array(dataSort['chickens','Mixture Neuron Count']))


# Sample from distribution for each welfare model and plot paired results 
samples_per_model = int(num_samples/num_models)
data_per_model = {}
for species in species_list:
    data_per_model[species] = []
    for model in models:
        data_per_model[species].extend(np.random.choice(data[species,model],samples_per_model))

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
    

chicken_wr = np.array(data_per_model['chickens'])
pig_wr = np.array(data_per_model['pigs'])

y1Paired, y2Paired = compute_y(np.array(data_per_model['pigs']),np.array(data_per_model['chickens']))

plt.hist(y1Paired,bins=30)
plt.show()

plt.hist(y2Paired,bins=30)
plt.show()

#Results for shifting from chicken to pork (Box Plot)
boxData = [y1Raw, y1Trimmed, y1Ordered, y1Paired]
fig, ax = plt.subplots(figsize = (7,4),dpi=300)
sns.boxplot(data=boxData, orient='h', ax=ax, showfliers=False, palette=my_pal)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(["Uncorrelated","Trimming Top/Bottom 2.5%","Ordering","Generating Samples Pair-Wise\nfrom Constituent Models"])        
ax.set_xlabel("Weighted Days of Suffering Averted")
ax.set_title("Suffering Averted by Switching from Chicken to Pork")        

#Results for shifting from chicken to pork (Box Plot)
boxData = [y2Raw, y2Trimmed, y2Ordered, y2Paired]
fig, ax = plt.subplots(figsize = (7,4),dpi=300)
sns.boxplot(data=boxData, orient='h', ax=ax, showfliers=False, palette=my_pal)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(["Uncorrelated","Trimming Top/Bottom 2.5%","Ordering","Generating Samples Pair-Wise\nfrom Constituent Models"])               
ax.set_xlabel("Weighted Days of Suffering Caused")
ax.set_title("Suffering Caused by Eating Chicken and Pork")        

