import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch


def draw_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
    return image

def _get_faces(image, mtcnn_model):
    boxes, _ = mtcnn_model.detect(image)
    if boxes is not None:
        faces = [image.crop(box) for box in boxes]
        return faces, boxes
    return None, None
def get_faces(image):
    pass

def get_embedding(embed_model, face, transforms, device):
    face = transforms(face).unsqueeze(0).to(device)
    embedding = embed_model(face)
    return embedding

def get_prediction(embedding):
    with torch.no_grad():
        prediction = torch.argmax(embedding, dim=1).item()
    return prediction

def get_name(dataset, prediction):
    return dataset.classes[prediction]

def recognize(image, transforms):
    faces, boxes = get_faces(image)
    if faces is not None:
        for face, box in zip(faces, boxes):
            embedding = get_embedding(face, transforms)
            prediction = get_prediction(embedding)
            name = get_name(prediction)
            image = draw_boxes(image, [box])
            draw = ImageDraw.Draw(image)
            draw.text((box[0], box[1]), name)
    return image

def tracking(mtcnn_model,image):
    boxes, _ = mtcnn_model.detect(image)
    if boxes is not None:
        num_faces = len(boxes)
        faces_infor = {}
        for face, box in zip(range(num_faces), boxes):
            faces_infor[face] = {'box': box}
        return faces_infor
    else:
        print('No face detected')
    return None