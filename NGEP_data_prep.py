import pandas as pd
import numpy as np
import sys
import os

# Change to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"Working directory: {os.getcwd()}")

##########################################
#loads datasets
##########################################

#uses pandas to read csv files from data and feature extractions
print("Loading features...")
features_dataframe = pd.read_csv("data/features.csv")

print("Loading Allen expression data...")
expression_dataframe = pd.read_csv("data/allen/expression_by_structure.csv")

print(f"✓ Features: {features_dataframe.shape}")
print(f"✓ Expression data: {expression_dataframe.shape}")

##########################################
# create mapping from keywords to Allen regions
##########################################

# define which Allen regions map to which brain_region keywords
REGION_KEYWORDS = {
    # --- Fallback / General Regions ---
    'cortex': [
        'isocortex', 'neocortex', 'cortical'
    ],
    'neocortex': [
        'isocortex', 'neocortex'
    ],

    # --- Major Cortical Regions ---
    'occipital': [
        'occipital', 'visual cortex', 'V1', 'VISp'
    ],
    'somatosensory': [
        'somatosensory', 'S1', 'barrel', 'SSp'
    ],
    'frontal': [
        'frontal', 'motor cortex', 'M1', 'MOp', 'prefrontal'
    ],
    'parietal': [
        'parietal', 'posterior parietal', 'PTL'
    ],
    'temporal': [
        'temporal', 'auditory', 'TEa', 'AUD'
    ],
    'cingulate': [
        'cingulate', 'anterior cingulate', 'ACA'
    ],
    'motor': [
        'motor', 'primary motor', 'secondary motor', 'MOp', 'MOs'
    ],
    'retrosplenial': [
        'retrosplenial', 'RSP'
    ],
    'orbital': [
        'orbital', 'ORB'
    ],
    'insula': [
        'insula', 'AI'
    ],

    # --- Cortical Layers ---
    'layer 1': [
        'layer 1', 'layer I', 'L1'
    ],
    'layer 2': [
        'layer 2', 'layer II', 'L2'
    ],
    'layer 3': [
        'layer 3', 'layer III', 'L3'
    ],
    'layer 2-3': [
        'layer 2-3', 'layer 2/3', 'layer II/III', 'L2/3'
    ],
    'layer 4': [
        'layer 4', 'layer IV', 'L4'
    ],
    'layer 5': [
        'layer 5', 'layer V', 'L5'
    ],
    'layer 6': [
        'layer 6', 'layer VI', 'L6'
    ],
    'layer 6b': [
        'layer 6b', 'subplate', 'SP'
    ],
}

#print out the region mapping directory
print("\n" + "="*60)
print("REGION KEYWORDS FOR MATCHING")
print("="*60)
for keyword, allen_patterns in REGION_KEYWORDS.items():
    print(f"\n{keyword.upper()} matches:")
    for pattern in allen_patterns:
        print(f"  - {pattern}")

##########################################
#extract Allen regions
##########################################

def get_expression_for_keywords(region_string, expression_dataframe, region_keywords):
    
    # handles NaN, None, and non-string types
    if region_string is None or (isinstance(region_string, float) and np.isnan(region_string)):
        return np.nan, [], []
    
    #ensures region_string is a string
    if not isinstance(region_string, str):
        region_string = str(region_string)
    
    #splits region string and normalize, in order to be able to search for all the keywords
    neuron_regions = [r.strip().lower() for r in region_string.split(',')]
    
    #creates a dictionary to hold the keywords that match
    matched_keywords = []
    
    #iterates through the keywords given and appends ones that are in the region map
    for keyword in region_keywords.keys():
        if keyword.lower() in neuron_regions:
            matched_keywords.append(keyword)
    
    if not matched_keywords:
        #uses generic neocortex patterns if no keywords match
        matched_keywords = ['occipital', 'somatosensory', 'cortex']
    
    #finds all expression records matching any of the patterns for matched keywords
    matches = pd.DataFrame()
    matched_regions = []
    
    for keyword in matched_keywords:
        allen_patterns = region_keywords[keyword]
        
        for pattern in allen_patterns:
            pattern_matches = expression_dataframe[
                expression_dataframe['structure_name'].str.contains(pattern, case=False, na=False)
            ]
            if len(pattern_matches) > 0:
                matches = pd.concat([matches, pattern_matches], ignore_index=True)
                matched_regions.extend(pattern_matches['structure_name'].unique())
    
    if len(matches) == 0:
        return np.nan, [], matched_keywords
    
    # Remove duplicates
    matches = matches.drop_duplicates()
    
    expr_value = matches['expression_energy'].mean()
    return expr_value, list(set(matched_regions)), matched_keywords

##########################################
#create target vector with region-specific values
##########################################

print("\n" + "="*60)
print("CREATING REGION-SPECIFIC TARGETS")
print("="*60)

expression_values = []
matched_keywords_list = []

for index, row in features_dataframe.iterrows():
    neuron_name = row['neuron_name']
    brain_region = row.get('brain_region', np.nan)  # Safer access
    
    # gets region-specific expression value
    expr_value, regions_used, keywords_used = get_expression_for_keywords(
        brain_region, 
        expression_dataframe, 
        REGION_KEYWORDS
    )
    
    expression_values.append(expr_value)
    matched_keywords_list.append(keywords_used)
    
    print(f"\nNeuron: {neuron_name}")
    print(f"  Region: {brain_region}")
    print(f"  Matched keywords: {keywords_used}")
    expr_str = f"{expr_value:.6f}" if not np.isnan(expr_value) else "NaN"
    print(f"Expression value: {expr_str}")
    print(f"  Matched {len(regions_used)} Allen regions")

y = np.array(expression_values)

#prints status of data preparation as well as metrics about the expression data distribution
print(f"\n{'='*60}")
print(f"Target vector created:")
print(f"  - Total neurons: {len(y)}")
print(f"  - Missing values: {np.isnan(y).sum()}")
if np.isnan(y).sum() == 0:
    print(f"  - Value range: {np.nanmin(y):.6f} to {np.nanmax(y):.6f}")
    print(f"  - Mean: {np.nanmean(y):.6f}")
    print(f"  - Std: {np.nanstd(y):.6f}")
else:
    print(f"  - Valid values: {(~np.isnan(y)).sum()}")
    if np.sum(~np.isnan(y)) > 0:
        print(f"  - Value range: {np.nanmin(y):.6f} to {np.nanmax(y):.6f}")
        print(f"  - Mean: {np.nanmean(y):.6f}")
        print(f"  - Std: {np.nanstd(y):.6f}")

##########################################
# Save combined dataset
##########################################

# Add expression column
features_dataframe['expression_energy'] = y

# Save
features_dataframe.to_csv("data/features_with_expression.csv", index=False)

print(f"\n{'='*60}")
print(f"✓ Saved to: data/features_with_expression.csv")
print(f"  Shape: {features_dataframe.shape}")

print("\n" + "="*60)
print("Region-specific mapping complete!")
print("Each neuron now has a target expression value matched to its sub-regions.")
print("="*60)