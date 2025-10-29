from dataset import loader
from dataset import KFoldGraphCV
import torch
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt




class GNN(torch.nn.Module):
    
    def __init__(self):
        in_feats = 128
        hidden_feats = 64
        num_classes = 4
        super().__init__()
        self.conv_in = GCNConv(in_feats, hidden_feats,bias=False)
        self.conv_hidden = GCNConv(hidden_feats, hidden_feats,bias = False)    
        self.conv_out = GCNConv(hidden_feats, num_classes,bias=False)
        self.activation = torch.nn.LeakyReLU()
        self.norm1 = torch.nn.BatchNorm1d(in_feats)
        self.norm2 = torch.nn.BatchNorm1d(hidden_feats)
        self.norm3 = torch.nn.BatchNorm1d(hidden_feats)
        #4714 -> 128
        self.compressor = torch.nn.Linear(in_features=4714,out_features=128)
        
        
    def forward(self,x,edge_index): 
        zero_mask = (x.abs().sum(dim=1) == 0).float().unsqueeze(1)
        # 1️ Message passing (layer 1)
        #22470 x 4714 ->  22470 x 128
        x = self.norm1(self.compressor(x))
        x = x * (1 - zero_mask)
        #22470 x 128 -> 22470 x 64
        h1 = self.norm2(self.activation(self.conv_in(x, edge_index)))  
        h1 = h1 * (1 - zero_mask)
        #22470 x 64 -> 22470 x 64
        h2 = self.norm3(self.activation(self.conv_hidden(h1,edge_index)))
        h2 = h2 * (1 - zero_mask)
        #22470 x 64 -> 22470 x 4
        out = self.conv_out(h2, edge_index)
        return out
    
class Loss(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.num_classes = 4
        self.f = torch.nn.CrossEntropyLoss()
        
        
    def loss(self,out,target,nodes):
        ''' calc cross entropy loss'''
        # out is of 22470 x 4
        return self.f(out[nodes],target[nodes])
    def test_accuracy(self,out,target,nodes):
        ''' calc classification rate'''
        out = out[nodes]
        target = target[nodes]
        return (torch.argmax(out,dim = 1) == target).float().mean()
        
    

def validate(X_full_train:torch.Tensor,fold:dict,gnn:GNN,edges:torch.Tensor,lr,epochs:int,
            Node_Classes_Y:torch.Tensor,device):
    EPS = 0.01
    X = fold["X_fold"]
    train_nodes = fold["train_nodes"]
    val_nodes = fold["val_nodes"]
    gnn.to(device)
    optimizer = torch.optim.Adam(gnn.parameters(),lr =lr)
    loss_function = Loss()
    losses = []
    vals = []
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
def train_ensembles(Node_Features_X,Edges,Node_Classes_Y):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K_FOLDS = 5
    
    DataLoader = KFoldGraphCV(Node_Features_X,Node_Classes_Y, K_FOLDS ,device)
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
    

    

    
    
    
    
        



