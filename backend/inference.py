import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import cv2
import os

#load model
model = load_model("backend/model/skin_segmentation_unet.keras")

def preprocess_image(image_path, target_size=(128, 128)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def postprocess_mask(mask, save_path="predicted_mask.png", threshold=0.1):
    print("prediction stast= min:", mask.min(), "max", mask.max(), "mean", mask.mean())

    binarized = (mask[0].squeeze() > threshold).astype(np.uint8)* 255
    Image.fromarray(binarized).save(save_path)
    print(f"Saved binarized mask with threshold={threshold} to {save_path}")
    return binarized
    
def predict_mask(image_path):
    input_image = preprocess_image(image_path)
    pred_mask = model.predict(input_image)

    return postprocess_mask(pred_mask)

