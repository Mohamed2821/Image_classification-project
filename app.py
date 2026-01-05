import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # FORCE CPU (VERY IMPORTANT)

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load model ONCE (smooth + fast)
model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=True
)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files["image"]

    image = Image.open(file).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    predictions = model.predict(image)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)

    results = []
    for item in decoded[0]:
        results.append({
            "label": item[1],
            "confidence": f"{item[2] * 100:.2f}%"
        })

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)

