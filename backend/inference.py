import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import load_model
from PIL import Image
import cv2

#load model
model = load_model("backend/model/skin_segmentation_unet.keras")

def preprocess_image(image_path, target_size=(128, 128)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def postprocess_mask(mask):
    mask = (mask > 0.5).astype(np.uint8)*255
    return mask[0]
    
def predict_mask(image_path):
    input_image = preprocess_image(image_path)
    pred_mask = model.predict(input_image)
    return postprocess_mask(pred_mask)

# test the prediction works
# if __name__ == "__main__":
#     result = predict_mask("ISIC_0000001.jpg")
#     cv2.imwrite("output_mask.png", result)