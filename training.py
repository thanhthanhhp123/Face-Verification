from src.model import FaceVerification
from torchvision import transforms as v2
from torchvision import datasets
import torch
from torch import nn
from torch.utils.data import DataLoader
import logging
import time

logging.basicConfig(level=logging.INFO)
device = torch.device("mps")
logging.info(f'Using device: {device}')

transforms = v2.Compose([
                v2.RandomHorizontalFlip(),
                v2.RandomRotation(15),
                v2.RandomResizedCrop(160, scale=(0.8, 1.0)),
                v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                v2.RandomGrayscale(p=0.1),
                v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                v2.Resize(160),
                v2.ToTensor(),
                v2.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
            ])
train_ds = datasets.ImageFolder(root=r'/Users/tranthanh/Documents/Deep Learning/Face-Verification/Face Dataset/Train', transform=transforms)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
num_classes = len(train_loader.dataset.classes)
model = FaceVerification(num_classes=num_classes, device=device, train_backbone=True)
start_time = time.time()
logging.info('Start training')
model.train(train_loader, num_epochs=50)
logging.info(f'Training done, Total time: {time.time()-start_time}')