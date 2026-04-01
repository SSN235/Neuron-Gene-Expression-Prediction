"""
NGEP: Neural Gene Expression Prediction
Gene Data Extraction Module

Downloads gene expression data from Allen Brain Atlas API.
Features: Multi-gene support, species mapping, pagination, duplicate detection.
"""

import os
import sys
import requests
import json
from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np

# Change to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"Working directory: {os.getcwd()}")

##########################################
# Configuration
##########################################

# Gene selection
GENES_TO_PREDICT = ["Pvalb"]  # Inhibitory neuron genes

# Species selection
SPECIES_INPUT = "mouse"

# API endpoints
ALLEN_API_IDS = "http://api.brain-map.org/api/v2/data/SectionDataSet/query.json"
ALLEN_API_DATA = "http://api.brain-map.org/api/v2/data/SectionDataSet/query.json"

# Pagination settings
BATCH_SIZE = 2000

# Paths
DATA_DIR = "data/allen"
EXPRESSION_FILE = f"{DATA_DIR}/expression_by_structure.csv"
DATASETS_FILE = f"{DATA_DIR}/section_datasets.json"

##########################################
# Species to Product ID Mapping
##########################################

SPECIES_TO_PRODUCT_ID = {
    # Adult Brain ISH Data
    "mouse": "1",
    "mouse_adult": "1",
    "human": "2",
    "human_adult": "2",
    
    # Developing Brain ISH Data
    "mouse_developing": "3",
    "mouse_development": "3",
    "human_developing": "4",
    "human_development": "4",
    
    # Non-Human Primate Data
    "macaque": "8",
    "rhesus": "8",
    "rhesus_macaque": "8",
    "macaque_developing": "9",
    
    # Other Primate Data
    "marmoset": "10",
    "common_marmoset": "10",
    
    # Specialized Data
    "mouse_connectivity": "143",
    "mouse_spinal_cord": "14",
}


def validate_species(species: str) -> str:
    """Validate species input and return Product ID."""
    species_lower = species.lower().strip()
    
    if species_lower not in SPECIES_TO_PRODUCT_ID:
        print(f"\n✗ ERROR: Species '{species}' not recognized.")
        print("\nSupported species (use lowercase):")
        print("  Adult Brain Data:")
        print("    - 'mouse' (Mouse Brain ISH)")
        print("    - 'human' (Human Brain ISH)")
        print("\n  Developing Brain Data:")
        print("    - 'mouse_developing' (Developing Mouse Brain ISH)")
        print("    - 'human_developing' (Developing Human Brain ISH)")
        print("\n  Non-Human Primate Data:")
        print("    - 'macaque' or 'rhesus' (Rhesus Macaque Brain ISH)")
        print("    - 'macaque_developing' (Developing Macaque Brain ISH)")
        print("    - 'marmoset' or 'common_marmoset' (Common Marmoset ISH)")
        print("\n  Specialized Data:")
        print("    - 'mouse_connectivity' (Mouse Connectivity Projection)")
        print("    - 'mouse_spinal_cord' (Mouse Spinal Cord ISH)")
        sys.exit(1)
    
    return SPECIES_TO_PRODUCT_ID[species_lower]


##########################################
# Helper Functions
##########################################

def load_previously_downloaded_datasets() -> Set[int]:
    """Load set of previously downloaded dataset IDs."""
    if os.path.exists(EXPRESSION_FILE):
        try:
            df = pd.read_csv(EXPRESSION_FILE)
            if 'section_dataset_id' in df.columns:
                return set(df['section_dataset_id'].unique().tolist())
        except Exception as e:
            print(f"Warning: Could not load expression data: {e}")
    return set()


def load_previously_fetched_datasets() -> List[Dict]:
    """Load previously fetched dataset metadata."""
    if os.path.exists(DATASETS_FILE):
        try:
            with open(DATASETS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load datasets file: {e}")
    return []


def fetch_dataset_ids(species_id: str, gene: str) -> List[Dict]:
    """
    Paginate through Allen API to fetch dataset IDs for a gene.
    
    Args:
        species_id: Product ID for species
        gene: Gene acronym (e.g., 'Pvalb')
    
    Returns:
        List of dataset dictionaries with 'id' and 'gene' keys
    """
    datasets = []
    start_row = 0
    
    print(f"\nFetching dataset IDs for {gene}...")
    
    while True:
        params = {
            "criteria": f"products[id$eq'{species_id}'], genes[acronym$eq'{gene}']",
            "include": "genes",
            "start_row": start_row,
            "num_rows": BATCH_SIZE
        }
        
        try:
            response = requests.get(ALLEN_API_IDS, params=params, timeout=60)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"✗ Failed to fetch datasets: {e}")
            break
        
        data = response.json()
        
        if "msg" not in data or len(data["msg"]) == 0:
            print("Reached end of dataset results.")
            break
        
        experiments = data["msg"]
        print(f"Fetched {len(experiments)} datasets (rows {start_row}-{start_row + len(experiments)})")
        
        # Extract dataset IDs
        for exp in experiments:
            datasets.append({
                "id": exp["id"],
                "gene": exp["genes"][0]["acronym"]
            })
        
        # Check if we've reached the end
        if len(experiments) < BATCH_SIZE:
            break
        
        start_row += BATCH_SIZE
    
    print(f"Total datasets for {gene}: {len(datasets)}")
    return datasets


def fetch_expression_data(dataset_id: int, skip_if_exists: bool = False) -> List[Dict]:
    """
    Fetch expression data for a specific dataset.
    
    Args:
        dataset_id: Allen dataset ID
        skip_if_exists: Skip if already downloaded
    
    Returns:
        List of expression records
    """
    expression_data = []
    struct_start = 0
    
    while True:
        params = {
            "id": dataset_id,
            "include": "structure_unionizes(structure)",
            "start_row": struct_start,
            "num_rows": BATCH_SIZE
        }
        
        try:
            response = requests.get(ALLEN_API_DATA, params=params, timeout=60)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"✗ Failed to fetch expression data for dataset {dataset_id}: {e}")
            break
        
        data = response.json()
        
        if "msg" not in data or len(data["msg"]) == 0:
            break
        
        structures = data["msg"][0].get("structure_unionizes", [])
        
        if len(structures) == 0:
            break
        
        print(f"  Fetched {len(structures)} structures for dataset {dataset_id}")
        
        # Extract expression values
        for struct_data in structures:
            structure = struct_data.get("structure")
            if structure:
                expression_data.append({
                    "section_dataset_id": dataset_id,
                    "structure_name": structure.get("name"),
                    "structure_acronym": structure.get("acronym"),
                    "expression_energy": struct_data.get("expression_energy"),
                    "expression_density": struct_data.get("expression_density"),
                    "expression_intensity": struct_data.get("expression_intensity")
                })
        
        # Check if more data available
        if len(structures) < BATCH_SIZE:
            break
        
        struct_start += BATCH_SIZE
    
    return expression_data


##########################################
# Main Execution
##########################################

def main():
    """Main execution flow."""
    
    print("\n" + "="*70)
    print("NGEP: GENE EXPRESSION DATA EXTRACTION")
    print("="*70)
    print(f"Species: {SPECIES_INPUT}")
    print(f"Genes: {', '.join(GENES_TO_PREDICT)}")
    print("="*70)
    
    # Validate species and get Product ID
    species_id = validate_species(SPECIES_INPUT)
    print(f"\nSpecies '{SPECIES_INPUT}' mapped to Product ID: {species_id}")
    
    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Load previously processed data
    previously_downloaded = load_previously_downloaded_datasets()
    previously_fetched = load_previously_fetched_datasets()
    
    print(f"\nPreviously downloaded datasets: {len(previously_downloaded)}")
    print(f"Previously fetched dataset IDs: {len(previously_fetched)}")
    
    # Combine skip sets
    skip_ids = previously_downloaded | set([d["id"] for d in previously_fetched])
    
    # Fetch dataset IDs (if not already done)
    if len(previously_fetched) == 0:
        print(f"\nFetching dataset IDs from Allen API...")
        all_datasets = []
        for gene in GENES_TO_PREDICT:
            datasets = fetch_dataset_ids(species_id, gene)
            # Filter out already processed
            new_datasets = [d for d in datasets if d["id"] not in skip_ids]
            all_datasets.extend(new_datasets)
        
        # Save for reference
        with open(DATASETS_FILE, 'w') as f:
            json.dump(all_datasets, f, indent=2)
        
        datasets_to_process = all_datasets
    else:
        datasets_to_process = [d for d in previously_fetched if d["id"] not in skip_ids]
    
    print(f"\nDatasets to process: {len(datasets_to_process)}")
    
    # Load existing expression data (if any)
    expression_data = []
    if os.path.exists(EXPRESSION_FILE):
        try:
            df = pd.read_csv(EXPRESSION_FILE)
            expression_data = df.to_dict('records')
            print(f"Loaded {len(expression_data)} existing expression records")
        except Exception as e:
            print(f"Warning: Could not load existing expression data: {e}")
    
    # Fetch expression data for each dataset
    print(f"\nFetching expression data from {len(datasets_to_process)} datasets...")
    for i, dataset in enumerate(datasets_to_process, 1):
        dataset_id = dataset["id"]
        
        if dataset_id in previously_downloaded:
            print(f"⊘ [{i}/{len(datasets_to_process)}] Already processed: {dataset_id}")
            continue
        
        print(f"\n[{i}/{len(datasets_to_process)}] Processing dataset {dataset_id}...")
        data = fetch_expression_data(dataset_id)
        expression_data.extend(data)
    
    # Save expression data
    if expression_data:
        expression_df = pd.DataFrame(expression_data)
        expression_df.to_csv(EXPRESSION_FILE, index=False)
        
        print("\n" + "="*70)
        print("EXPRESSION DATA SUMMARY")
        print("="*70)
        print(f"Total records: {len(expression_df)}")
        print(f"Unique datasets: {expression_df['section_dataset_id'].nunique()}")
        print(f"Unique structures: {expression_df['structure_name'].nunique()}")
        print("="*70)
        
        print(f"\n✓ Expression data saved to: {EXPRESSION_FILE}")
    else:
        print("\n✗ No expression data to save")
        return 1
    
    print("✓ Gene expression extraction complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())