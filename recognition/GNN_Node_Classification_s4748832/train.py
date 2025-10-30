from dataset import loader
from dataset import KFoldGraphCV
from Modules import GNN , Loss
import torch
from torch_geometric.nn import GCNConv

import matplotlib.pyplot as plt

   

def validate(X_full_train:torch.Tensor,fold:dict,gnn:GNN,edges:torch.Tensor,lr,epochs:int,
            Node_Classes_Y:torch.Tensor,device):
    """ a function that trains and tests the ensembles """
    EPS = 0.01 # set a convergence value
    X = fold["X_fold"] # get the training for validation fold k
    train_nodes = fold["train_nodes"] # train nodes 
    val_nodes = fold["val_nodes"]# validation nodes
    gnn.to(device)
    # use adam optimizer
    optimizer = torch.optim.Adam(gnn.parameters(),lr =lr)
    # define use the special cross entropy loss funtion
    loss_function = Loss()
    losses = []
    vals = []
    # do full batchw runs this is fine
    # for the GNN because its a small dataset
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = gnn(X,edges)
        pred_full = gnn(X_full_train,edges)
        loss = loss_function.loss(pred,Node_Classes_Y,train_nodes)
        val = loss_function.test_accuracy(pred_full,Node_Classes_Y,val_nodes)
        losses.append(loss.item())
        vals.append(val.item())
        loss.backward()
        optimizer.step()
        
        print(f"epoch {epoch} loss: {losses[-1]}")
        if len(vals) >= 16:
            print(f"test_accuracy: {vals[-8]}")
            if sum(vals[-16:-8]) + EPS >= sum(vals[-8:]) :
                break
    return losses,vals


print("validating")
def train_ensembles(Node_Features_X,Edges,Node_Classes_Y,train_indices,test_indices):
    ''' this trains each GNN on different folds'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K_FOLDS = 5
    
    DataLoader = KFoldGraphCV(Node_Features_X,Node_Classes_Y, K_FOLDS ,device,train_indices,test_indices)
    lr = 0.01    
    epochs = 100
    ensemble = []
    folds = DataLoader.folds
    X_full_train = DataLoader.X_Full_Train
    for i,fold in enumerate(folds):
        gnn = GNN()
        losses,vals = validate(X_full_train,fold,gnn,Edges,lr,epochs,Node_Classes_Y,device)
        print(f"final accuracy on fold {i}: {vals[-1]}")
        print(f"final loss on fold {i}: {losses[-1]}")
        ensemble.append(gnn)
    return ensemble

if __name__ == "__main__":
    K_FOLDS = 5
    Node_Features_X,Node_Classes_Y,train_indices,test_indices,Edges,labels = loader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble = train_ensembles(Node_Features_X,Edges,Node_Classes_Y,train_indices,test_indices)
    torch.save(ensemble, "ensemble_models.pt")
    

    
    
    
    
        



