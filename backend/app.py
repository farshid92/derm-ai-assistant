from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
import numpy as np
import io
import uuid
import tensorflow as tf
import keras
from keras.models import load_model



# Define the custom metric function that was used during training
@keras.saving.register_keras_serializable()
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# Register the custom function for model loading
keras.saving.register_keras_serializable()(dice_coef)


app = FastAPI()
model = load_model("backend/model/skin_segmentation_unet_fine_tuned_augmented2.keras")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    orig_size = image.size  # (width, height)
    image_resized = image.resize((128, 128))
    input_image = np.array(image_resized) / 255.0
    input_image = np.expand_dims(input_image, axis=0)

    pred_mask = model.predict(input_image)
    pred_mask = np.squeeze(pred_mask)
    
    binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    mask_img = Image.fromarray(binary_mask)
    # Resize mask back to original image size
    mask_img = mask_img.resize(orig_size, resample=Image.NEAREST)

    output_path = f"backend/outputs/{uuid.uuid4().hex}.png"
    mask_img.save(output_path)

    return FileResponse(output_path, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)