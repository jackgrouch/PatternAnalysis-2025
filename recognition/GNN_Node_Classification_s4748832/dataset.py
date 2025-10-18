import os
import json
import pandas as pd
import torch
from torch_geometric.data import Data

def loader():
    # Path to the dataset folder
    DATA_PATH = "C:\\Users\\jackg\\OneDrive\\Documents\\COMP3710\\COMP3710_A3\\PatternAnalysis-2025\\recognition\\GNN_Node_Classification_s4748832\\data\\facebook_large\\facebook_large"
    # --- Load edges ---
    edges_file = os.path.join(DATA_PATH, "musae_facebook_edges.csv")
    edges = pd.read_csv(edges_file)
    edge_index = torch.tensor(edges.values.T, dtype=torch.long)

    # --- Load node features ---
    features_file = os.path.join(DATA_PATH, "musae_facebook_features.json")
    with open(features_file, "r") as f:
        features_dict = json.load(f)

    # Ensure consistent order of nodes
    node_ids = sorted(features_dict.keys(), key=int)
    X = torch.zeros((len(node_ids),4714))
    for n in node_ids:
        X[int(n),features_dict[str(n)]] = 1
    return X
print(loader().shape)



