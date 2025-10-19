from dataset import loader
from dataset import KFoldGraphCV
import torch


Node_Features_X,Edges,Node_Classes_Y = loader() #22470 x 4714 , , 22470
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(Node_Features_X.shape)
print(Edges.shape)
print(Node_Classes_Y.shape)
K_FOLDS = 5
DataLoader = KFoldGraphCV(Node_Features_X,Node_Classes_Y, K_FOLDS ,device)



