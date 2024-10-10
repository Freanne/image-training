import os
import logging
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

image_size = (260, 260)  # Adjust this to match your model's input size
model = None  # Global variable to hold the model

def load_best_model():
    global model
    try:
        logging.info("Loading the best model.")
        model = keras.models.load_model("models/best_model.keras")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def load_best_model_and_predict(image_path):
    try:
        logging.info(f"Loading and preprocessing the image from {image_path}.")
        img = keras.preprocessing.image.load_img(image_path, target_size=image_size)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        logging.info("Making prediction.")
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions, axis=1)[0]

        return predicted_class, confidence
    except Exception as e:
        logging.error(f"Error in load_best_model_and_predict: {e}")
        raise

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    load_best_model()

@app.get("/")
async def read_root():
    return {"message": "Welcome to the image classification API!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_location = f"temp/{file.filename}"
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        class_names = sorted(os.listdir("data"))
        logging.info(f"Received image: {file.filename}")

        # Load the best model and make a prediction
        predicted_class, confidence = load_best_model_and_predict(file_location)
        predicted_class = class_names[predicted_class]

        # Clean up the temporary file
        os.remove(file_location)

        return JSONResponse(content={"predicted_class": predicted_class, "confidence": round(float(confidence)*100, 2)})
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)