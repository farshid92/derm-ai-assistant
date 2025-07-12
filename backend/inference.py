import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import cv2
import os

#  dice_coef function
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

#load model
model = load_model("backend/model/skin_segmentation_unet_fine_tuned_augmented2.keras",custom_objects={'dice_coef': dice_coef})

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

    print("Input image shape:", input_image.shape)
    print("Predicted mask shape:", pred_mask.shape)

    return postprocess_mask(pred_mask)

if __name__ == "__main__":
    test_image_path = "ISIC_0000001.jpg"
    predict_mask(test_image_path)