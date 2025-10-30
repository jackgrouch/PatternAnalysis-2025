from Modules import GNN
from dataset import loader
from dataset import KFoldGraphCV
import torch

def predict(Node_Features_X:torch.Tensor,
        Edges,GNNs:list[GNN]):
    ''' get the predictions'''
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
    ''' calculate the accuracy which is num_class_correct/total'''
    predictions = predict(Node_Features_X,Edges,GNNs)
    test_acc = (predictions[test_nodes] == Node_Classes_Y[test_nodes]).float().mean()
    return test_acc
if __name__  == "__main__":
    K_FOLDS = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Node_Features_X,Node_Classes_Y,train_indices,test_indices,Edges,labels = loader()
    #ensemble = torch.load("ensemble_models.pt")
    ensemble = torch.load("ensemble_models.pt", weights_only=False, map_location="cpu")
    
    prediction = predict(Node_Features_X,Edges,ensemble)
    prediction = prediction[test_indices][:100]
    actual =  Node_Classes_Y[test_indices][:100]
    for i in range(0,10):
        print(f" for facebook page node id: {test_indices[i]}")
        print(f"page type: {labels[int(actual[i])]} predicted page type: {labels[int(prediction[i])]}")
    print(f"Final Test Accuracy: {calc_test_score(Node_Features_X,Edges,ensemble,Node_Classes_Y,test_indices)}")