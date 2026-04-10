"""
NGEP Local Validator Backend (Parametrized)

Adds:
- NUM_NEURONS control
- GENE_TYPE control
- Optional live Allen Brain API support

Set USE_ALLEN_API = False to match training behavior (recommended)
"""

from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
from flask_cors import CORS
import requests as http_requests
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

# ──────────────────────────────────────────────────────────────────────
# CONFIG (EDIT THESE)
# ──────────────────────────────────────────────────────────────────────

NUM_NEURONS = 100
GENE_TYPE = "Pvalb"       # change this
USE_ALLEN_API = False     # True = dynamic gene, False = training-consistent

SPECIES = "mouse"
BRAIN_REGION = "neocortex"

# ──────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

NEUROMORPHO_API = 'https://neuromorpho.org/api/neuron/select'

# ──────────────────────────────────────────────────────────────────────
# MODEL (unchanged)
# ──────────────────────────────────────────────────────────────────────

class NGEPModel(nn.Module):
    def __init__(self, input_size=14, hidden1=128, hidden2=64):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2, 1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

# ──────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────────────

def parse_swc_text(swc_text):
    rows = []
    for line in swc_text.strip().split('\n'):
        if not line or line.startswith('#'):
            continue
        parts = re.split(r'[\s,]+', line)
        if len(parts) >= 7:
            try:
                rows.append([float(p) for p in parts[:7]])
            except:
                continue
    return np.array(rows) if rows else None


def extract_features(swc):
    if swc is None:
        return None

    soma = swc[swc[:, 1] == 1]
    soma_radius = np.sum(soma[:, 5]) if len(soma) else 0

    dend = swc[np.isin(swc[:, 1], [3, 4])]
    if len(dend) > 1:
        coords = dend[:, 2:5]
        length = np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1))
    else:
        length = 0

    parents = swc[:, 6]
    valid = parents[parents > 0]
    bif = np.sum(np.unique(valid, return_counts=True)[1] >= 2) if len(valid) else 0

    node_ids = set(swc[:, 0])
    parent_ids = set(parents)
    terminals = len(node_ids - parent_ids)

    bd = bif / (length + 1e-6)

    return np.array([
        soma_radius,
        length,
        bif,
        terminals,
        bd,
        length / (soma_radius + 1e-6),
        bif / (terminals + 1e-6),
        terminals / (length + 1e-6),
        soma_radius * length,
        bif * terminals,
        np.log(bif + 1),
        np.log(terminals + 1),
        soma_radius ** 2,
        length ** 2
    ], dtype=np.float32)

# ──────────────────────────────────────────────────────────────────────
# NEURON FETCHING
# ──────────────────────────────────────────────────────────────────────

def fetch_neurons(count=100):
    neurons = []
    page = random.randint(0, 100)

    while len(neurons) < count:
        params = {"q": f"species:{SPECIES}", "page": page, "pagesize": 500}
        r = http_requests.get(NEUROMORPHO_API, params=params, timeout=20)

        if r.status_code != 200:
            break

        data = r.json()
        items = data.get("_embedded", {}).get("neuronResources", [])

        for n in items:
            if len(neurons) >= count:
                break

            regions = n.get("brain_region", [])
            if not any("neocortex" in r.lower() for r in regions):
                continue

            neurons.append({
                "name": n.get("neuron_name"),
                "region": ", ".join(regions),
                "archive": n.get("archive", "")
            })

        page += 1

    return neurons

# ──────────────────────────────────────────────────────────────────────
# SWC DOWNLOAD
# ──────────────────────────────────────────────────────────────────────

def download_swc(name, archive=""):
    url = f"https://neuromorpho.org/dableFiles/{archive.lower()}/CNG%20version/{name}.CNG.swc"
    try:
        r = http_requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.text
    except:
        pass
    return None

# ──────────────────────────────────────────────────────────────────────
# GENE EXPRESSION
# ──────────────────────────────────────────────────────────────────────

# Minimal fallback map (you can expand this)
STATIC_EXPRESSION = {
    "Pvalb": 6.7
}

ALLEN_API = "http://api.brain-map.org/api/v2/data/query.json"

def get_expression(region, gene=GENE_TYPE):
    if not USE_ALLEN_API:
        return STATIC_EXPRESSION.get(gene, 6.7)

    try:
        query = f"""
        model::StructureUnionize,
        rma::criteria,
        [genes.name$eq'{gene}']
        """

        r = http_requests.get(ALLEN_API, params={"criteria": query}, timeout=10)
        data = r.json()

        if "msg" in data and len(data["msg"]) > 0:
            return data["msg"][0].get("expression_energy")

    except:
        return None

    return None

# ──────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────

def run_pipeline(num_neurons=NUM_NEURONS, gene=GENE_TYPE):
    neurons = fetch_neurons(num_neurons)

    results = []

    for n in neurons:
        swc = download_swc(n["name"], n["archive"])
        parsed = parse_swc_text(swc) if swc else None
        feats = extract_features(parsed)

        if feats is None:
            continue

        expr = get_expression(n["region"], gene)

        results.append({
            "neuron": n["name"],
            "region": n["region"],
            "expression": expr
        })

    return results

# ──────────────────────────────────────────────────────────────────────
# API
# ──────────────────────────────────────────────────────────────────────

@app.route("/run", methods=["POST"])
def run():
    data = request.json or {}

    num_neurons = data.get("num_neurons", NUM_NEURONS)
    gene = data.get("gene", GENE_TYPE)

    results = run_pipeline(num_neurons, gene)

    return jsonify(results)

# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Running with {NUM_NEURONS} neurons | Gene: {GENE_TYPE}")
    app.run(port=5000, debug=True)