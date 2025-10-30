
import torch
from torch_geometric.nn import GCNConv


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