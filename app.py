from flask import Flask, render_template, request
import os
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# ---------- CONFIG ----------
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------- LOAD MODEL ----------
model = tf.keras.applications.MobileNetV2(
    weights="imagenet"
)

# ---------- IMAGE PREPROCESS ----------
def prepare_image(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# ---------- ROUTES ----------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("image")

        if file:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            img = Image.open(image_path).convert("RGB")
            processed = prepare_image(img)

            preds = model.predict(processed)
            decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)
            prediction = decoded[0][0][1]

    return render_template("index.html", prediction=prediction, image=image_path)

# ---------- RENDER ENTRY POINT ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
