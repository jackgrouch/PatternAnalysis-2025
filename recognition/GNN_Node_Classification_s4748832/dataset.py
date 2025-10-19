import os
import json
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np


def loader():
    # Path to the dataset folder
    DATA_PATH = "C:\\Users\\jackg\\OneDrive\\Documents\\COMP3710\\COMP3710_A3\\PatternAnalysis-2025\\recognition\\GNN_Node_Classification_s4748832\\data\\facebook_large\\facebook_large"
    # --- Load edges ---
    edges_file = os.path.join(DATA_PATH, "musae_facebook_edges.csv")
    edges = pd.read_csv(edges_file)
    edge_index = torch.tensor(edges.values.T, dtype=torch.long)
    
    labels = {"tvshow":0,"government":1,"company":2,"politician":3}
    target_file = os.path.join(DATA_PATH, "musae_facebook_target.csv")
    target = pd.read_csv(target_file)
    Y = torch.zeros(22470)
    for i,y in enumerate(target["page_type"]):
        Y[i] = labels[y]


    # --- Load node features ---
    features_file = os.path.join(DATA_PATH, "musae_facebook_features.json")
    with open(features_file, "r") as f:
        features_dict = json.load(f)

    # Ensure consistent order of nodes
    node_ids = sorted(features_dict.keys(), key=int)
    X = torch.zeros((len(node_ids),4714))
    for n in node_ids:
        X[int(n),features_dict[str(n)]] = 1
    return X,edge_index,Y

import numpy as np
from sklearn.model_selection import KFold



class KFoldGraphCV:
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, 
                K: int, device, random_state: int = 42):
        """
        K-Fold loader for node-level CV with zeroed val/test features.

        Args:
            X: full node features (N x F)
            Y: node labels (N,)
            train_indices: indices to perform K-Fold CV on
            test_indices: indices to hold out for final testing
            K: number of folds
            random_state: for reproducibility
        """
        self.device = device
        self.X = X
        self.Y = Y
        indices = np.random.permutation(np.arange(len(X)))
        split = int(0.9 * len(indices))
        self.test_indices = indices[split:]
        self.train_indices = indices[:split]
        self.K = K
        self.random_state = random_state
        self.X_KTrain = self._create_X_KTrain()
        self.folds = self._create_folds()

    def _create_X_KTrain(self):
        """Create full KFold graph with test nodes zeroed"""
        X_kfold = self.X.clone()
        X_kfold[self.test_indices] = 0  # zero out test nodes
        return X_kfold

    def _create_folds(self):
        folds = []
        kf = KFold(n_splits=self.K, shuffle=True, random_state=self.random_state)
        for train_fold_idx, val_fold_idx in kf.split(self.train_indices):
            # original indices
            train_nodes = self.train_indices[train_fold_idx]
            val_nodes   = self.train_indices[val_fold_idx]
            
            # copy X_KFOLD and zero out val nodes for this fold
            X_fold = self.X_KTrain.clone()
            X_fold = X_fold.to(device=self.device)
            X_fold[val_nodes] = 0
            
            
            folds.append({
                'X_fold': X_fold,      # full graph with test nodes zeroed + val nodes zeroed
                'train_nodes': train_nodes,
                'val_nodes': val_nodes
            })
        return folds

    def get_test_graph(self):
        """Return full train/test graph"""
        X_full_train = self.X_KTrain.clone()
        X_full_train = X_full_train.to(device=self.device)
        return {
            
            'X_Train': X_full_train,  # test nodes zeroed, all train nodes present
            'train_nodes': self.train_indices,
            'test_nodes': self.test_indices
        }

    def __len__(self):
        return len(self.folds)

    def __getitem__(self, idx):
        return self.folds[idx]