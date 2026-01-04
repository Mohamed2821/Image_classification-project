from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
model = tf.keras.applications.MobileNetV2(weights='imagenet')
preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
decode = tf.keras.applications.mobilenet_v2.decode_predictions
def prepare_image(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    return preprocess(img)
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)
            img = Image.open(image_path).convert('RGB')
            processed = prepare_image(img)
            preds = model.predict(processed)
            prediction = decode(preds, top=1)[0][0][1]
    return render_template('index.html', prediction=prediction, image=image_path)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)