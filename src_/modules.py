import torch
import torch.nn as nn

class ArcFaceFeatureExtractor(nn.Module):
    def __init__(self, pretrained_model):
        super(ArcFaceFeatureExtractor, self).__init__()
        self.arcface = pretrained_model

    def forward(self, x):
        faces = self.arcface.get(x)
        if faces:
            return faces[0].normed_embedding
        return None
    
class Classify(nn.Module):
    def __init__(self, num_classes, extractor):
        super(Classify, self).__init__()
        self.extractor = extractor
        self.logits = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.extractor(x)
        if x is None:
            return None
        x = self.logits(x)
        return x