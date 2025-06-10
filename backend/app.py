from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
import numpy as np
import io
import uuid
import tensorflow as tf
from tensorflow import _tf_uses_legacy_keras
from keras import models
from keras import layers
from keras import Model
from tensorflow.keras.models import load_model

app = FastAPI()
model = load_model("backend/model/skin_segmentation_unet.keras")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((128, 128))
    input_image = np.array(image) / 255.0
    input_image = np.expand_dims(input_image, axis=0)

    pred_mask = model.predict(input_image)
    pred_mask = np.squeeze(pred_mask)
    result_img = Image.fromarray((pred_mask * 255).astype(np.uint8))

    output_path = f"backend/outputs/{uuid.uuid4().hex}.png"
    result_img.save(output_path)

    return FileResponse(output_path, media_type="image/png")