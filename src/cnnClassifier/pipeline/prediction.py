import numpy as np
from tensorflow.keras.models import load_model
import os
import base64
import io
from PIL import Image


class PredictionPipeline:
    def __init__(self, filename="inputImage.jpg"):
        self.filename = filename
        model_path = os.path.join("model", "model.h5")
        if not os.path.exists(model_path):
            model_path = os.path.join("artifacts", "training", "model.h5")
        self.model = load_model(model_path, compile=False)

    def predict_base64(self, base64_str):
        img_bytes = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize((224, 224))
        test_image = np.array(img, dtype=np.float32) / 255.0
        test_image = np.expand_dims(test_image, axis=0)

        preds = self.model(test_image, training=False).numpy()
        result = np.argmax(preds, axis=1)

        if result[0] == 1:
            return [{"image": "Tumor"}]
        else:
            return [{"image": "Normal"}]

    def predict(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')
            return self.predict_base64(b64)
        return [{"image": "Normal"}]