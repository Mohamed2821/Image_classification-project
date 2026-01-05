from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load model ONCE (important)
model = tf.keras.applications.MobileNetV2(
    weights="imagenet"
)

def prepare_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        image = Image.open(request.files["image"])
        processed = prepare_image(image)
        preds = model.predict(processed)
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0][0]

        return jsonify({
            "label": decoded[1],
            "confidence": f"{decoded[2]*100:.2f}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()
