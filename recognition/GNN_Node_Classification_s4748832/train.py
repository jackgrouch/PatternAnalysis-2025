from dataset import loader
from dataset import KFoldGraphCV
import torch
from torch_geometric.nn import GCNConv


Node_Features_X,Edges,Node_Classes_Y = loader() #22470 x 4714 , 2x |E| , 22470
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


K_FOLDS = 5
DataLoader = KFoldGraphCV(Node_Features_X,Node_Classes_Y, K_FOLDS ,device)

class GNN(torch.nn.Module):
    
    def __init__(self,in_feats,hidden_feats,num_classes):
        super().__init__()
        self.conv_in = torch.nn.Sequential(GCNConv(in_feats, hidden_feats),
                                        torch.nn.LeakyReLU(),
                                        torch.nn.BatchNorm1d(hidden_feats))
        self.conv_hidden = torch.nn.Sequential(
                                GCNConv(in_feats, hidden_feats),
                                torch.nn.LeakyReLU(),
                                torch.nn.BatchNorm1d(hidden_feats))
        self.conv_out = GCNConv(hidden_feats, num_classes)
        
        
    def forward(self,x,edge_index): 
        # 1️ Message passing (layer 1)
        h1 = self.conv_in(x, edge_index)  # => A_norm @ X @ W1 => n_sample*n_hidden
        h2 = self.conv_hidden(h1,edge_index)
        h3 = self.conv_hidden(h2,edge_index)
        out = self.conv_out(h3, edge_index)


