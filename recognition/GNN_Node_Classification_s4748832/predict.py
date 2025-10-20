from train import GNN
import torch
def predict(Node_Features_X:torch.Tensor,
        Edges,GNNs:list[GNN]):
    weight = 1.0 / len(GNNs)
    scores = torch.zeros((len(Node_Features_X),4))
    soft = torch.nn.Softmax(dim=1)
    for gnn in GNNs:
        gnn.eval()  # very important for inference stability
        with torch.no_grad():
            pred = gnn(Node_Features_X,Edges)
            scores = scores + weight*soft(pred)
    predictions = torch.argmax(scores,dim =1)
    return predictions
            
def calc_test_score(Node_Features_X:torch.Tensor,
        Edges,GNNs:list[GNN],Node_Classes_Y:torch.Tensor,test_nodes):
    predictions = predict(Node_Features_X,Edges,GNNs)
    test_acc = (predictions[test_nodes] == Node_Classes_Y[test_nodes]).float().mean()
    return test_acc