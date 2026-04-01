"""
NGEP: Neural Gene Expression Prediction
Neuron Data Extraction Module

Downloads neuron morphologies from NeuroMorpho.org REST API.
Features: Pagination, duplicate detection, metadata extraction.
"""

import os
import sys
import requests
import json
from typing import Dict, List, Set, Tuple
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# Change to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"Working directory: {os.getcwd()}")

##########################################
# Configuration
##########################################

# Search parameters
SPECIES = "mouse"
BRAIN_REGION = "neocortex"
MAX_NEURONS = 7500  # Target number of neurons

# API endpoints
NEUROMORPHO_API = 'https://neuromorpho.org/api/neuron/select'
PAGE_SIZE = 500  # Results per API call

# Paths
DATA_DIR = "data/neuromorpho"
METADATA_FILE = f"{DATA_DIR}/neuron_metadata.csv"
NAMES_FILE = f"{DATA_DIR}/neuron_names.csv"

##########################################
# Helper Functions
##########################################

def load_previously_downloaded_neurons() -> Set[str]:
    """Load set of previously downloaded neuron names from metadata."""
    if os.path.exists(METADATA_FILE):
        try:
            df = pd.read_csv(METADATA_FILE)
            return set(df['neuron_name'].tolist())
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}")
            return set()
    return set()


def load_previously_identified_neurons() -> Set[str]:
    """Load set of previously identified neuron names."""
    if os.path.exists(NAMES_FILE):
        try:
            df = pd.read_csv(NAMES_FILE)
            return set(df.iloc[:, 0].tolist())
        except Exception as e:
            print(f"Warning: Could not load names file: {e}")
            return set()
    return set()


def fetch_neurons_from_api(species: str, brain_region: str, max_neurons: int, 
                          skip_neurons: Set[str]) -> Tuple[List[str], List[Dict]]:
    """
    Paginate through NeuroMorpho API to fetch neuron metadata.
    
    Args:
        species: Species name (e.g., 'mouse')
        brain_region: Brain region filter (e.g., 'neocortex')
        max_neurons: Target number of neurons
        skip_neurons: Set of neuron names to skip (already downloaded)
    
    Returns:
        Tuple of (neuron_names list, metadata list)
    """
    neuron_names = []
    metadata = []
    page = 0
    
    print(f"\nSearching for neurons in '{brain_region}' (target: {max_neurons})")
    print(f"Skipping {len(skip_neurons)} previously processed neurons\n")
    
    while len(neuron_names) < max_neurons:
        params = {
            "q": f"species:{species}",
            "page": page,
            "pagesize": PAGE_SIZE
        }
        
        try:
            response = requests.get(NEUROMORPHO_API, params=params, timeout=120)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"✗ Failed to fetch page {page}: {e}")
            break
        
        data = response.json()
        
        # Check for embedded neuron data
        if "_embedded" not in data or "neuronResources" not in data["_embedded"]:
            print("No neuron data in response. Reached end of database.")
            break
        
        page_neurons = data["_embedded"]["neuronResources"]
        
        if len(page_neurons) == 0:
            print("No neurons on this page. Reached end of database.")
            break
        
        # Process neurons on this page
        for neuron in page_neurons:
            if len(neuron_names) >= max_neurons:
                print(f"Reached target of {max_neurons} neurons.")
                break
            
            neuron_name = neuron.get("neuron_name")
            brain_regions = neuron.get("brain_region", [])
            
            # Skip if already processed
            if neuron_name in skip_neurons:
                continue
            
            # Filter by brain region
            if brain_region not in brain_regions:
                continue
            
            # Add to results
            neuron_names.append(neuron_name)
            metadata.append({
                "neuron_name": neuron_name,
                "brain_region": ", ".join(brain_regions) if brain_regions else "unknown",
                "brain_region_list": brain_regions,
                "cell_type": neuron.get("cell_type", "unknown"),
                "species": neuron.get("species", "unknown"),
                "lab": neuron.get("lab", "unknown"),
                "brain_region_primary": brain_region
            })
            
            if len(neuron_names) % 100 == 0 or len(neuron_names) <= 5:
                print(f"  [{len(neuron_names)}/{max_neurons}] {neuron_name}")
        
        if len(neuron_names) >= max_neurons:
            break
        
        page += 1
    
    print(f"\nTotal neurons found: {len(neuron_names)}")
    return neuron_names, metadata


def download_swc_files(neuron_names: List[str], skip_neurons: Set[str]) -> Tuple[int, int, int]:
    """
    Download SWC morphology files for neurons.
    
    Args:
        neuron_names: List of neuron names to download
        skip_neurons: Set of neuron names already downloaded
    
    Returns:
        Tuple of (downloaded_count, skipped_count, failed_count)
    """
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0
    
    print(f"\nDownloading SWC files for {len(neuron_names)} neurons\n")
    
    for i, neuron_name in enumerate(neuron_names, 1):
        # Skip if already downloaded
        if neuron_name in skip_neurons:
            print(f"⊘ [{i}/{len(neuron_names)}] Already downloaded: {neuron_name}")
            skipped_count += 1
            continue
        
        # Fetch neuron page
        page_url = f"https://neuromorpho.org/neuron_info.jsp?neuron_name={neuron_name}"
        try:
            page_response = requests.get(page_url, timeout=30)
            page_response.raise_for_status()
        except requests.RequestException as e:
            print(f"✗ [{i}/{len(neuron_names)}] Failed to fetch page: {neuron_name}")
            failed_count += 1
            continue
        
        # Parse for SWC link
        soup = BeautifulSoup(page_response.text, 'html.parser')
        swc_link = soup.find('a', {'href': lambda x: x and '.swc' in x.lower()})
        
        if not swc_link:
            print(f"✗ [{i}/{len(neuron_names)}] No SWC link found: {neuron_name}")
            failed_count += 1
            continue
        
        # Construct full URL
        swc_url = swc_link['href']
        if not swc_url.startswith('http'):
            swc_url = f"https://neuromorpho.org/{swc_url}"
        
        # Download SWC file
        try:
            swc_response = requests.get(swc_url, timeout=30)
            swc_response.raise_for_status()
        except requests.RequestException as e:
            print(f"✗ [{i}/{len(neuron_names)}] Failed to download: {neuron_name}")
            failed_count += 1
            continue
        
        # Save SWC file
        filepath = f"{DATA_DIR}/{neuron_name}.swc"
        try:
            with open(filepath, 'w') as f:
                f.write(swc_response.text)
            print(f"✓ [{i}/{len(neuron_names)}] Downloaded: {neuron_name}")
            downloaded_count += 1
        except IOError as e:
            print(f"✗ [{i}/{len(neuron_names)}] Failed to save: {neuron_name}")
            failed_count += 1
    
    return downloaded_count, skipped_count, failed_count


##########################################
# Main Execution
##########################################

def main():
    """Main execution flow."""
    
    print("\n" + "="*70)
    print("NGEP: NEURON DATA EXTRACTION")
    print("="*70)
    print(f"Species: {SPECIES}")
    print(f"Brain Region: {BRAIN_REGION}")
    print(f"Target neurons: {MAX_NEURONS}")
    print("="*70)
    
    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Load previously processed neurons
    previously_downloaded = load_previously_downloaded_neurons()
    previously_identified = load_previously_identified_neurons()
    skip_set = previously_downloaded | previously_identified
    
    print(f"\nPreviously downloaded: {len(previously_downloaded)}")
    print(f"Previously identified: {len(previously_identified)}")
    print(f"Total to skip: {len(skip_set)}")
    
    # Check if we need more neurons
    if len(previously_identified) >= MAX_NEURONS:
        print(f"\nAlready have {len(previously_identified)} neurons (target: {MAX_NEURONS})")
        print("Skipping API search.")
        neuron_names = list(previously_identified)
        metadata = pd.read_csv(METADATA_FILE).to_dict('records')
    else:
        # Fetch neurons from API
        neuron_names, metadata = fetch_neurons_from_api(
            SPECIES, BRAIN_REGION, MAX_NEURONS, skip_set
        )
        
        # Add previously identified neurons
        if previously_identified:
            neuron_names = list(previously_identified) + neuron_names
            existing_metadata = pd.read_csv(METADATA_FILE).to_dict('records')
            metadata = existing_metadata + metadata
    
    # Download SWC files
    downloaded, skipped, failed = download_swc_files(neuron_names, previously_downloaded)
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(METADATA_FILE, index=False)
    
    # Save neuron names
    names_df = pd.DataFrame(neuron_names, columns=['neuron_name'])
    names_df.to_csv(NAMES_FILE, index=False)
    
    # Print summary
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    print(f"Successfully downloaded: {downloaded}")
    print(f"Previously downloaded: {skipped}")
    print(f"Failed: {failed}")
    print(f"Total identified: {len(neuron_names)}")
    print("="*70)
    
    # Print sample metadata
    print("\nSample Metadata (first 5):")
    print("-"*70)
    print(metadata_df[['neuron_name', 'brain_region', 'cell_type']].head())
    
    print("\n✓ Neuron extraction complete!")
    print(f"✓ Metadata saved to: {METADATA_FILE}")
    print(f"✓ Names saved to: {NAMES_FILE}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())