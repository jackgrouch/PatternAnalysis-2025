from Modules import GNN
from dataset import loader
from sklearn.decomposition import PCA
from dataset import KFoldGraphCV
import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib.patches import Patch

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
    return predictions,scores
            
def calc_test_score(Node_Features_X:torch.Tensor,
        Edges,GNNs:list[GNN],Node_Classes_Y:torch.Tensor,test_nodes):
    ''' calculate the accuracy which is num_class_correct/total'''
    predictions,scores = predict(Node_Features_X,Edges,GNNs)
    test_acc = (predictions[test_nodes] == Node_Classes_Y[test_nodes]).float().mean()
    return test_acc
if __name__  == "__main__":
    K_FOLDS = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Node_Features_X,Node_Classes_Y,train_indices,test_indices,Edges,labels = loader()
    #ensemble = torch.load("ensemble_models.pt")
    ensemble = torch.load("ensemble_models.pt", weights_only=False, map_location="cpu")
    
    
    
    prediction,scores = predict(Node_Features_X,Edges,ensemble)
    
    scores = scores[test_indices]
    prediction = prediction[test_indices][:100]
    actual =  Node_Classes_Y[test_indices][:100]
    
    for i in range(0,10):
        print(f" for facebook page node id: {test_indices[i]}")
        print(f"page type: {labels[int(actual[i])]} predicted page type: {labels[int(prediction[i])]}")
    print(f"Final Test Accuracy: {calc_test_score(Node_Features_X,Edges,ensemble,Node_Classes_Y,test_indices)}")
    
    pca = PCA(2)
    X = pca.fit_transform(scores)
    # (N,2)
    y = Node_Classes_Y[test_indices].cpu().numpy()  # integer labels, possibly not 0..K-1
    unique = np.unique(y)
    n_classes = len(unique)

    # build a label->index mapping so colours are contiguous 0..K-1
    label_to_idx = {label: i for i, label in enumerate(unique)}
    y_mapped = np.array([label_to_idx[v] for v in y])

    # create a discrete colormap with K entries (tab10 can handle up to 10)
    cmap = plt.get_cmap('tab10', n_classes)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X[:,0], X[:,1], c=y_mapped, cmap=cmap, s=20, alpha=0.8)

    # create legend entries using the SAME colormap
    # class_names must correspond to the actual labels in `unique` order
    # e.g. class_names_map = {label_value: "label_name", ...}
    class_names_map = {0: 'tvshow', 1: 'government', 2: 'company', 3: 'politician'}  # replace with your mapping
    legend_elements = [
    Patch(facecolor=cmap(i), label=class_names_map[unique[i]])
    for i in range(n_classes)
]

    plt.legend(handles=legend_elements, title="Classes")
    plt.title("PCA reduced 4d-score vector of GNN ")
    plt.xlabel("Component 1")
    plt.xlabel("Component 2")
    plt.show()