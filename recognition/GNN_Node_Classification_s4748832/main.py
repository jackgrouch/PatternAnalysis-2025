import torch
from dataset import KFoldGraphCV,loader
from train import GNN,validate
from predict import predict , calc_test_score
import matplotlib.pyplot as plt

K_FOLDS = 5
Node_Features_X,Edges,Node_Classes_Y,labels = loader()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DataLoader = KFoldGraphCV(Node_Features_X,Node_Classes_Y, K_FOLDS ,device)


def train_ensembles():
    
    lr = 0.01    
    epochs = 100
    ensemble = []
    folds = DataLoader.folds
    X_full_train = DataLoader.X_Full_Train
    for i,fold in enumerate(folds):
        gnn = GNN()
        losses,vals = validate(X_full_train,fold,gnn,Edges,lr,epochs,Node_Classes_Y,device)
        
        print(f"final accuracy on validation fold {i+1}: {vals[-1]}")
        print(f"final loss on K-fold training set {i+1}: {losses[-1]}")
        ensemble.append(gnn)
    return ensemble
ensemble = train_ensembles()
print(f"Final Test Accuracy: {calc_test_score(Node_Features_X,Edges,ensemble,Node_Classes_Y,DataLoader.test_indices)}")
