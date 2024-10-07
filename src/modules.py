import torch
import torch.nn as nn
import torch.nn.functional as F

from facenet_pytorch import InceptionResnetV1, MTCNN
    
class Extractor(nn.Module):
    def __init__(self, pretrained_model, ):
        super(Extractor, self).__init__()
        self.extractor = pretrained_model

    def forward(self, x):
        x = self.extractor(x)
        return x
    
class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.logits = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.logits(x)
        return x
    
class Tracker(nn.Module):
    def __init__(self, mtcnn_model):
        super(Tracker, self).__init__()
        self.mtcnn = mtcnn_model

    def forward(self, x):
        x = self.mtcnn(x)
        return x
    



    