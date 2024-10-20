import cv2
from facenet_pytorch import MTCNN, fixed_image_standardization, InceptionResnetV1
from PIL import Image
import datetime
import pickle
import csv
import os
import numpy as np
import tqdm
import time

import torch
from torchvision import transforms as v2

from src.modules import *

device = 'cpu' 
facenet_model = InceptionResnetV1(pretrained="casia-webface", classify=False)
feature_agg = NetworkFeatureAggregator(facenet_model, ['block8'], device = device, train_backbone=True)
facenet = FaceNetClassifierWithAugFMap(num_classes=20, feature_agg=feature_agg, aug_layer=AugmentationLayer()).to(device)

transform = v2.Compose(
    [v2.Resize((160, 160)),
     v2.ToTensor(),
     fixed_image_standardization]
)

checkpoint = torch.load(r'models/best_model_812_1910.pth', map_location=device)
facenet.load_state_dict(checkpoint['facenet_state_dict'])
extractor = facenet.extractor
extractor.to(device)
extractor.eval()

def get_embeddings(image):
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embeddings = extractor(image)['block8']
    return embeddings

def save_database(db_path = 'FaceDataset/Train_cropped'):
#Save embeddings of all images in the dataset
    embeddings = {}
    for person in tqdm.tqdm(os.listdir(db_path), desc='Saving database...'):
        person_path = os.path.join(db_path, person)
        for image in os.listdir(person_path):
            image_path = os.path.join(person_path, image)
            img = cv2.imread(image_path)
            print(image_path)   
            embeddings[person] = get_embeddings(img)
    with open('database/embeddings_db.pkl', 'wb') as file:
        pickle.dump(embeddings, file)
    return 

def find_closest(embeddings_db, query_embedding):
    min_distance = float('inf')
    closest_person = None
    for person, embedding in embeddings_db.items():
        distance = torch.nn.functional.pairwise_distance(embedding.to(device), query_embedding.to(device)).mean()
        if distance < min_distance:
            min_distance = distance
            closest_person = person
    return closest_person

def log_attendance(name):
    with open('attendance.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        now = datetime.datetime.now()
        writer.writerow([name, now.strftime("%Y-%m-%d %H:%M:%S")])


if __name__ == '__main__':
    if not os.path.exists('database/embeddings_db.pkl'):
        save_database()

    with open('database/embeddings_db.pkl', 'rb') as file:
        embeddings_db = pickle.load(file)
    
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    #Set FPS
    cap.set(cv2.CAP_PROP_FPS, 60)
    mtcnn = MTCNN(select_largest=True, keep_all=False)
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        cv2.putText(frame, 'Attendance System', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs = mtcnn.detect(frame)
        if boxes is not None:
            best_box = boxes[probs.argmax()]
            x0, y0, x1, y1 = best_box
            face = frame[int(y0):int(y1), int(x0):int(x1)]
            embeddings = get_embeddings(face)
            name = find_closest(embeddings_db, embeddings)
            cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
            cv2.putText(frame, name, (int(x0), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
        end_time = time.time()
        elapsed_time = end_time - start_time  # Thời gian xử lý 1 frame
        fps = 1 / elapsed_time
        cv2.putText(frame, f'FPS: {fps:.2f}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

        

