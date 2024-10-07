import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as v2
import numpy as np
import logging
import cv2
import os
import warnings
from facenet_pytorch import InceptionResnetV1, MTCNN
from .modules import Extractor, Classifier, Tracker
from .utils import *

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
pwd = os.path.dirname(os.path.abspath(__file__))
pwd_ = os.path.dirname(pwd)

class FaceVerification():
    def __init__(self, num_classes, device= 'cuda' if torch.cuda.is_available() else 'cpu',
                 train_backbone=False):
        self.device = device
        self.extractor = Extractor(InceptionResnetV1(pretrained='vggface2', classify=False)).to(self.device)
        self.classifier = Classifier(num_classes).to(self.device)
        self.tracker = Tracker(MTCNN(image_size=160, margin=0, min_face_size=20)).to(self.device)
        self.transforms = v2.Compose([
                v2.RandomHorizontalFlip(),
                v2.RandomRotation(15),
                v2.RandomResizedCrop(160, scale=(0.8, 1.0)),
                v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                v2.RandomGrayscale(p=0.1),
                v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                v2.Resize(160),
                v2.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
                v2.ToTensor(),
            ])

        self.train_backbone = train_backbone
        if not train_backbone:
            for param in self.extractor.parameters():
                param.requires_grad = False

        self.extactor_opt = torch.optim.Adam(self.extractor.parameters(), lr=0.0001)
        self.classifier_opt = torch.optim.Adam(self.classifier.parameters(), lr=0.001)

        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_loader, num_epochs):
        losses = []
        for epoch in range(num_epochs):
            self.extractor.train()
            self.classifier.train()
            total_loss = 0
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.extactor_opt.zero_grad()
                self.classifier_opt.zero_grad()

                face = self.tracker(images)
                embeddings = self.extractor(face)
                outputs = self.classifier(embeddings)

                loss = self.criterion(outputs, labels)
                loss.backward()

                self.extactor_opt.step()
                if not self.train_backbone:
                    self.extactor_opt.step()
                self.classifier_opt.step()

                total_loss += loss.item()
            logging.info(f'Epoch {epoch}/{num_epochs}, Loss: {total_loss/len(train_loader)}')
            torch.save(losses, os.path.join(pwd_, 'losses.npy'))
            self.save(path=os.path.join(pwd_, 'model.pth'))

    def evaluate(self, test_loader):
        self.extractor.eval()
        self.classifier.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                embeddings = self.extractor(images)
                outputs = self.classifier(embeddings)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        logging.info(f'Accuracy: {100 * correct / total}')
    
    def _predict(self, image):
        faces, boxes = get_faces(image, self.tracker)
        if faces is not None:
            for face, box in zip(faces, boxes):
                embedding = get_embedding(self.extractor, face, self.transforms, self.device)
                prediction = get_prediction(embedding)
                name = get_name(self.dataset, prediction)
                image = draw_boxes(image, [box])
                draw = ImageDraw.Draw(image)
                draw.text((box[0], box[1]), name)
        return image
    
    
    def save(self, path):
        torch.save({
            'extractor': self.extractor.state_dict(),
            'classifier': self.classifier.state_dict(),
            'extractor_opt': self.extactor_opt.state_dict(),
            'classifier_opt': self.classifier_opt.state_dict(),
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.extractor.load_state_dict(checkpoint['extractor'])
        self.classifier.load_state_dict(checkpoint['classifier'])
        self.extactor_opt.load_state_dict(checkpoint['extractor_opt'])
        self.classifier_opt.load_state_dict(checkpoint['classifier_opt'])
        self.extractor.eval()
        self.classifier.eval()



