import torch
import torch.nn as nn
import torch.nn.functional as F

from facenet_pytorch import InceptionResnetV1, MTCNN

class FaceNet(nn.Module):
    def __init__(self, num_classes, device):
        super(FaceNet, self).__init__()
        self.device = device
        self.facenet = InceptionResnetV1(pretrained='vggface2', classify=False).eval().to(self.device)
        self.mtcnn = MTCNN(keep_all=True, device=self.device)

        for param in self.facenet.parameters():
            param.requires_grad = False

        self.logits = nn.Linear(512, num_classes)

    def forward(self, x):
        with torch.no_grad():
            x = self.mtcnn(x)
            if x is None:
                return None
            x = self.facenet(x)

        x = self.logits(x)
        return x

    