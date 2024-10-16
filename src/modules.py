import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
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
        self.logits = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.logits(x)
        return x

class AugmentationLayer(nn.Module):
    def __init__(self, p: float):
        """
        Args:
        p: Percentage of channels to be augmented (0 <= p <= 1)
        """
        super(AugmentationLayer, self).__init__()
        self.p = p

        # Define the transformations (without noise)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))
        ])

    def forward(self, M):
        """
        M: Input feature map of shape (b, c, h, w)
        Returns:
        M': Augmented feature map
        """
        b, c, h, w = M.shape
        
        num_channels_to_augment = int(self.p * c)
        channel_indices = torch.randperm(c)[:num_channels_to_augment]

        M_aug = M.clone()

        for idx in channel_indices:
            for batch_idx in range(b):

                channel_map = M[batch_idx, idx].unsqueeze(0)
                channel_map = transforms.ToPILImage()(channel_map) 
                channel_map = self.transform(channel_map)
                channel_map = transforms.ToTensor()(channel_map)

                channel_map = channel_map + torch.randn_like(channel_map) * 0.05
                
                M_aug[batch_idx, idx] = channel_map.squeeze(0)  

        return M_aug
    



    