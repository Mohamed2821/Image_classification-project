from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = Flask(__name__)

# Load model once (CPU safe)
model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=True
)

def prepare_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty file"})

    try:
        image = Image.open(io.BytesIO(file.read()))
        processed = prepare_image(image)
        preds = model.predict(processed)
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=3)[0]

        results = [
            {"label": label, "confidence": float(confidence * 100)}
            for (_, label, confidence) in decoded
        ]

        return jsonify({"predictions": results})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=False)
