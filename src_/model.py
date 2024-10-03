import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import logging

logging.basicConfig(level=logging.INFO)
cwd = os.getcwd()

class FaceVerification:
    def __init__(self, extractor, classifier, device):
        self.extractor = extractor
        self.classifier = classifier
        self.device = device
        self.extractor.to(self.device)
        self.classifier.to(self.device)
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = 100
        self.save_path = os.path.join(cwd, 'models/')

        self.train_extractor = False
        self.extractor_optimizer = optim.Adam(self.extractor.parameters(), lr=0.0001)


    def train(self, dataloader):
        if self.train_extractor:
            self.extractor.train()
        else:
            self.extractor.eval()
        self.classifier.train()
        for epoch in range(self.epochs):
            for i, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                features = self.extractor(x)
                output = self.classifier(features)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                if self.train_extractor:
                    self.extractor_optimizer.zero_grad()
                    self.extractor_optimizer.step()
                if i % 10 == 0:
                    logging.info(f'Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}')
    def save(self):
        torch.save(self.classifier.state_dict(), self.save_path + 'classifier.pth')
        torch.save(self.extractor.state_dict(), self.save_path + 'extractor.pth')
    
    def load(self):
        self.classifier.load_state_dict(torch.load(self.save_path + 'classifier.pth'))
        self.extractor.load_state_dict(torch.load(self.save_path + 'extractor.pth'))
