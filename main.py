from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io
import numpy as np

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
        <html>
            <body>
                <form action="/predictions/" method="post" enctype="multipart/form-data">
                    <input type="file" name="file">
                    <input type="submit">
                </form>
            </body>
        </html>
    """

@app.post("/predictions/")
async def predict_image(file: UploadFile = File(...)):
    # Load the Keras model
    model = keras.models.load_model('https://www.dropbox.com/scl/fi/urs8l4gyyp23qkkbtingj/from_state_to_auc_fine_tune_all_layers_mobilenetv3_adam_50epochs_imagenet_weights.h5?rlkey=jq3eyovv61poykvae0fc35rkh&dl=1')
    
    # Read the image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Preprocess the image
    image = image.resize((256, 256))
    image = keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, 0)
    image = keras.applications.mobilenet_v3.preprocess_input(image)
    
    # Make predictions
    predictions = model.predict(image)
    pred_index = np.argmax(model.predict(image))
    
    tags = {0: "safe driving", 1: "texting - right", 2: "talking on the phone - right", 3: "texting - left", 4: "talking on the phone - left", 5: "operating the radio", 6: "drinking", 7: "reaching behind", 8: "hair and makeup", 9: "talking to passenger"}
    predicted_tag = tags.get(pred_index, "Unknown")

    # Return the predictions
    return {"predictions": predictions.tolist(), "predicted_tag": predicted_tag}
