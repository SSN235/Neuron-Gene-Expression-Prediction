## Imports
import os
import sys
import requests
import json
from typing import Dict, List, Tuple
from pathlib import Path
from bs4 import BeautifulSoup

# Data handling
import numpy as np
import pandas as pd

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# ML utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

# Graph processing (for SWC parsing)
import networkx as nx

##########################################
# Change to script directory
##########################################

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"Working directory: {os.getcwd()}")

##########################################
#Parse features from neuromorpho swc files
#########################################

def parse_swc(filename):
    #use pandas to read the swc and return a processable dataframe
    swc_dataframe = pd.read_csv(filename, sep=r'[\s,]+', comment='#', header=None, names=['n','type','x','y','z','radius','parent'], engine='python')

    return swc_dataframe

    
##########################################
#Extract features from neuromorpho swc files
#########################################

def extract_swc_features(swc_df):
    #creates dictionary for the features
    features = {}

    #checks the length of the input dataframe
    if len(swc_df) == 0:
        print("Error: Dataframe is empty")
    
    #Go through the rows and find the soma radius
    soma_rows = swc_df[swc_df['type'] == 1] #type 1 is a soma (in case of multiple soma, sum their radii)
    if len(soma_rows) > 0:        
        #creates a new key for the 
        features['soma_radius'] = soma_rows['radius'].sum()
    else:
        features['soma_radius'] = np.nan

    #extract total dendritic length from the swc file
    dendrite_rows = swc_df[swc_df['type'].isin([3, 4])]  # types 3 & 4 are dendrites
    if len(dendrite_rows) > 0:
        # Compute distances between consecutive nodes
        coords = dendrite_rows[['x', 'y', 'z']].values
        distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        features['total_dendritic_length'] = distances.sum()
    else:
        features['total_dendritic_length'] = 0

    # extract total number of bifurcation points (nodes with 2+ children)
    child_count = swc_df[swc_df['parent'] > 0].groupby('parent').size()
    bifurcations = (child_count >= 2).sum()
    features['bifurcations'] = bifurcations

    #extracts number of terminals
    terminals = sum(1 for node in swc_df['n'] if node not in swc_df['parent'].values)
    features['terminals'] = terminals

    #extracts the branch density
    features['branch_density'] = features['bifurcations'] / (features['total_dendritic_length'] + 1e-6)

        
    return features


##########################################
#use main function to extract features for all neurons
#########################################

def main():
    #creates empty lists for the list of features and neuron_names
    features_list = []
    neuron_names = []

    print("\nExtracting features from SWC files...")

    # iterate through all swc files in the directory
    for swc_file in os.listdir("data/neuromorpho"):
        if swc_file.endswith('.swc'):  # Only process .swc files
            try:

                swc_path = f"data/neuromorpho/{swc_file}"
                swc_df = parse_swc(swc_path)
                features = extract_swc_features(swc_df)
                features_list.append(features)
                
                # Extract neuron name (filename without .swc extension)
                neuron_name = swc_file.replace('.swc', '')
                neuron_names.append(neuron_name)
                
                print(f"✓ {neuron_name}")
                
            except Exception as e:
                print(f"✗ Skipped {swc_file}: {e}")

    #convert features list to dataframe
    features_dataframe = pd.DataFrame(features_list)
    features_dataframe['neuron_name'] = neuron_names
    
    print(f"\nExtracted features for {len(features_dataframe)} neurons")
    
    # loads neuron metadata to get specific brain regions
    print("\nLoading neuron metadata...")
    neuron_metadata = pd.read_csv("data/neuromorpho/neuron_metadata.csv")
    
    print(f"Loaded metadata for {len(neuron_metadata)} neurons")
    
    # combines features with metadata to get actual sub-regions
    print("\nMerging features with metadata...")
    features_dataframe = features_dataframe.merge(
        neuron_metadata[['neuron_name', 'brain_region', 'cell_type']],
        on='neuron_name',
        how='left'
    )
    
    print(f"Merged successfully")

    # Save to csv
    features_dataframe.to_csv("data/features.csv", index=False)
    print(f"\n✓ Saved {len(features_dataframe)} neurons to data/features.csv")
    
    print("\n" + "="*60)
    print("Features extracted:")
    print("="*60)
    print(features_dataframe[['neuron_name', 'soma_radius', 'total_dendritic_length', 'brain_region', 'cell_type']])
    
    print("\n" + "="*60)
    print("Summary statistics:")
    print("="*60)
    print(f"Total neurons: {len(features_dataframe)}")
    print(f"Features per neuron: {len([col for col in features_dataframe.columns if col not in ['neuron_name', 'brain_region', 'cell_type']])}")
    print(f"Soma radius range: {features_dataframe['soma_radius'].min():.2f} to {features_dataframe['soma_radius'].max():.2f}")
    print(f"Dendritic length range: {features_dataframe['total_dendritic_length'].min():.2f} to {features_dataframe['total_dendritic_length'].max():.2f}")
    
    print("\nBrain regions represented:")
    for region in features_dataframe['brain_region'].unique():
        count = len(features_dataframe[features_dataframe['brain_region'] == region])
        print(f"  {region}: {count} neurons")
    
    ##########################################
    # Calculate Regional Morphology Statistics
    ##########################################
    
    print("\n" + "="*60)
    print("CALCULATING REGIONAL MORPHOLOGY STATISTICS")
    print("="*60)
    
    # Select morphological feature columns (exclude neuron_name, brain_region, cell_type)
    morphology_features = [col for col in features_dataframe.columns 
                          if col not in ['neuron_name', 'brain_region', 'cell_type']]
    
    print(f"\nMorphological features to analyze:")
    for feat in morphology_features:
        print(f"  - {feat}")
    
    # Calculate mean and std for each feature by brain region
    region_stats_mean = features_dataframe.groupby('brain_region')[morphology_features].mean()
    region_stats_std = features_dataframe.groupby('brain_region')[morphology_features].std()
    region_counts = features_dataframe.groupby('brain_region').size()
    
    print(f"\nRegional statistics computed for {len(region_stats_mean)} brain regions")
    
    # Save to CSV
    region_stats_mean.to_csv("data/region_morphology_stats_mean.csv")
    region_stats_std.to_csv("data/region_morphology_stats_std.csv")
    region_counts.to_csv("data/region_neuron_counts.csv")
    
    print(f"\n✓ Saved region statistics to:")
    print(f"  - data/region_morphology_stats_mean.csv")
    print(f"  - data/region_morphology_stats_std.csv")
    print(f"  - data/region_neuron_counts.csv")
    
    print("\n" + "="*60)
    print("REGIONAL MORPHOLOGY STATISTICS (MEAN)")
    print("="*60)
    print(region_stats_mean)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) # Calls the main function and exits with its return code