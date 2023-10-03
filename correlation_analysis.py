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
import matplotlib as mpl
from heatmap_wr_ranges import heatmap_wr_ranges
from scatter_wr_ranges import scatter_wr_ranges
from compute_y import compute_y
import seaborn as sns
from computeSummaryStatsArr import computeSummaryStatsArr
from adj_wr_correlation import adj_wr_correlation

#Inputs
output_dir_adj = 'Sent_Adj_WR_Outputs'
output_dir_unadj = 'Sent_Adj_WR_Outputs2'
models = ['Qualitative', 'High-Confidence (Simple Scoring)', \
    'Cubic', 'High-Confidence (Cubic)', \
    'Qualitative Minus Social', 'Pleasure-and-pain-centric', \
    'Higher-Lower Pleasures', 'Undiluted Experience', 'Neuron Count', \
    'Mixture Neuron Count']
    
species_list = ['pigs','chickens']

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
numBins = 100
xlims = [0,2]
ylims = [0,2]
textLoc = [1.05,1.8]
my_pal = {"lightsteelblue","lightcoral","thistle","navajowhite"}

# For each species, load the welfare range simulation 
# data for each welfare model
for species in species_list:
    for model in models:
        data_unadj[species,model] = pickle.load(open(os.path.join(output_dir_unadj,'{} {}.p'.format(species, model)), 'rb'))

# For each species, load the p(sentience)-adjusted welfare range simulation 
# data for each welfare model
for species in species_list:
    for model in models:
        data[species,model] = pickle.load(open(os.path.join(output_dir_adj,'{} {}.p'.format(species, model)), 'rb'))

r = np.corrcoef(data['pigs','Mixture Neuron Count'], data['chickens','Mixture Neuron Count'])
pig_mean = round(np.mean(data['pigs','Mixture Neuron Count']),4)
chicken_mean = round(np.mean(data['chickens','Mixture Neuron Count']),4)
correlation_coeff = round(r[0,1],4)

computeSummaryStatsArr(data['pigs','Mixture Neuron Count'],printEn=True,name="Pigs Mixture Neuron Count")
computeSummaryStatsArr(data['chickens','Mixture Neuron Count'],printEn=True,name="Chickens Mixture Neuron Count")

# Plot uncorrelated reults for pigs vs. chickens 
scatter_wr_ranges(data['pigs','Mixture Neuron Count'], \
                  data['chickens','Mixture Neuron Count'],\
                  'Pigs','Chickens',xlims,ylims,pig_mean,chicken_mean,\
                  correlation_coeff,'Independent Sampling from Mixture Model',\
                  textLoc,area=area,printEn=False) 

heatmap_wr_ranges(data['pigs','Mixture Neuron Count'], \
                  data['chickens','Mixture Neuron Count'],\
                  'Pigs','Chickens',xlims,ylims,pig_mean,chicken_mean,\
                  correlation_coeff,'Independent Sampling from Mixture Model',\
                  textLoc,numBins=numBins,printEn=True)
    
y1Raw, y2Raw = compute_y(np.array(data['pigs','Mixture Neuron Count']),\
                                 np.array(data['chickens','Mixture Neuron Count']))
    

n = int(num_samples*0.025)
m = int(num_samples*0.975)-n

indx = sorted(np.argsort(y1Raw)[n:])
y1Trimmed = y1Raw[indx]
indx2 = sorted(np.argsort(y1Trimmed)[:m])
y1Trimmed = y1Trimmed[indx2]

indx = sorted(np.argsort(y2Raw)[n:])
y2Trimmed = y2Raw[indx]
indx2 = sorted(np.argsort(y1Trimmed)[:m])
y2Trimmed = y2Trimmed[indx2]


# Order results and plot paired results 
dataSort = data
for species in species_list:
    for model in models:
        dataSort[species,model].sort()

r = np.corrcoef(dataSort['pigs','Mixture Neuron Count'], dataSort['chickens','Mixture Neuron Count'])
pig_mean = round(np.mean(dataSort['pigs','Mixture Neuron Count']),4)
chicken_mean = round(np.mean(dataSort['chickens','Mixture Neuron Count']),4)
correlation_coeff = round(r[0,1],4)

computeSummaryStatsArr(dataSort['pigs','Mixture Neuron Count'],printEn=True,name="Pigs Mixture Neuron Count - Ordered")
computeSummaryStatsArr(dataSort['chickens','Mixture Neuron Count'],printEn=True,name="Chickens Mixture Neuron Count - Ordered")

scatter_wr_ranges(dataSort['pigs','Mixture Neuron Count'], \
                  dataSort['chickens','Mixture Neuron Count'],\
                  'Pigs','Chickens',xlims,ylims,pig_mean,chicken_mean,\
                  correlation_coeff,'Sampling from Ordered Data',\
                  textLoc,area=area,printEn=False) 

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
        data_per_model[species].extend(np.random.choice(data_unadj[species,model],samples_per_model))
    data_per_model[species] = adj_wr_correlation(species,data_per_model[species],num_samples)

r = np.corrcoef(data_per_model['pigs'], data_per_model['chickens'])
pig_mean = round(np.mean(data_per_model['pigs']),4)
chicken_mean = round(np.mean(data_per_model['chickens']),4)
correlation_coeff = round(r[0,1],4)

computeSummaryStatsArr(data_per_model['pigs'],printEn=True,name="Pigs - Paired from Component")
computeSummaryStatsArr(data_per_model['chickens'],printEn=True,name="Chickens - Paired from Component")

scatter_wr_ranges(data_per_model['pigs'],data_per_model['chickens'],\
                  'Pigs','Chickens',xlims,ylims,pig_mean,chicken_mean,\
                  correlation_coeff,'Paired Sampling from Constituent Models',\
                  textLoc,area=area,printEn=False)

heatmap_wr_ranges(data_per_model['pigs'],data_per_model['chickens'],\
                  'Pigs','Chickens',xlims,ylims,pig_mean,chicken_mean,\
                  correlation_coeff,'Paired Sampling from Constituent Models',\
                  textLoc,numBins=numBins,printEn=True)
    

chicken_wr = np.array(data_per_model['chickens'])
pig_wr = np.array(data_per_model['pigs'])

y1Paired, y2Paired = compute_y(np.array(data_per_model['pigs']),np.array(data_per_model['chickens']))


#Results for shifting from chicken to pork (Box Plot)
boxData = [y1Raw, y1Trimmed, y1Ordered, y1Paired]
fig, ax = plt.subplots(figsize = (12,7),dpi=300)
sns.boxplot(data=boxData, orient='h', ax=ax, showfliers=False, palette=my_pal)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(["Uncorrelated","Trimming\nTop/Bottom 2.5%","Ordering","Generating Samples\nPair-Wise from\nConstituent Models"])               
ax.set_xlabel("Weighted Days of Suffering Averted")
ax.set_title("Suffering Averted by Switching from Chicken to Pork")        
fig.savefig('./Plots/Switching_Suffering.png')   

#Results for shifting from chicken to pork (Box Plot)
boxData = [y2Raw, y2Trimmed, y2Ordered, y2Paired]
fig, ax = plt.subplots(figsize = (12,7),dpi=300)
sns.boxplot(data=boxData, orient='h', ax=ax, showfliers=False, palette=my_pal)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(["Uncorrelated","Trimming\nTop/Bottom 2.5%","Ordering","Generating Samples\nPair-Wise from\nConstituent Models"])               
ax.set_xlabel("Weighted Days of Suffering Caused")
ax.set_title("Suffering Caused by Eating Chicken and Pork")        
fig.savefig('./Plots/Eating_Both_Suffering.png')   


# For each species, load the p(sentience)-adjusted welfare range simulation 
# data for each welfare model
species_list = ['chickens','carp']
for species in species_list:
    for model in models:
        data[species,model] = pickle.load(open(os.path.join(output_dir_adj,'{} {}.p'.format(species, model)), 'rb'))


# Sample from distribution for each welfare model and plot paired results 
data_per_model = {}
for species in species_list:
    data_per_model[species] = []
    for model in models:
        data_per_model[species].extend(np.random.choice(data[species,model],samples_per_model))

r = np.corrcoef(data_per_model['chickens'], data_per_model['carp'])
pig_mean = round(np.mean(data_per_model['chickens']),4)
chicken_mean = round(np.mean(data_per_model['carp']),4)
correlation_coeff = round(r[0,1],4)

textLoc = [3*0.525,3*0.9]
heatmap_wr_ranges(data_per_model['chickens'],data_per_model['carp'],\
                  'Chickens','Carp',xlims,ylims,pig_mean,chicken_mean,\
                  correlation_coeff,'Paired Sampling from Constituent Models - Carp',\
                  textLoc,numBins=numBins,printEn=True,lims = [0,3])
    
# For each species, load the p(sentience)-adjusted welfare range simulation 
# data for each welfare model
species_list = ['chickens','shrimp']
for species in species_list:
    for model in models:
        data[species,model] = pickle.load(open(os.path.join(output_dir_adj,'{} {}.p'.format(species, model)), 'rb'))


# Sample from distribution for each welfare model and plot paired results 
data_per_model = {}
for species in species_list:
    data_per_model[species] = []
    for model in models:
        data_per_model[species].extend(np.random.choice(data[species,model],samples_per_model))

r = np.corrcoef(data_per_model['chickens'], data_per_model['shrimp'])
pig_mean = round(np.mean(data_per_model['chickens']),4)
chicken_mean = round(np.mean(data_per_model['shrimp']),4)
correlation_coeff = round(r[0,1],4)

textLoc = [5*0.525,5*0.9]
heatmap_wr_ranges(data_per_model['chickens'],data_per_model['shrimp'],\
                  'Chickens','Shrimp',xlims,ylims,pig_mean,chicken_mean,\
                  correlation_coeff,'Paired Sampling from Constituent Models - Shrimp',\
                  textLoc,numBins=numBins,printEn=True,lims = [0,5])    
