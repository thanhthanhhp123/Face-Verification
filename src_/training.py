import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from modules import ArcFaceFeatureExtractor, Classify
from model import FaceVerification

import i

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
