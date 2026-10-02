import numpy as np
from tensorflow.keras.models import load_model
import os
import base64
import io
from PIL import Image

from cnnClassifier.constants import PARAMS_FILE_PATH
from cnnClassifier.utils import preprocessing
from cnnClassifier.utils.inference import predict_probabilities
from cnnClassifier.utils.common import read_yaml


class PredictionPipeline:
    def __init__(self, filename="inputImage.jpg"):
        self.filename = filename
        # Prefer the freshly trained model; fall back to the packaged copy.
        model_path = os.path.join("artifacts", "training", "model.h5")
        if not os.path.exists(model_path):
            model_path = os.path.join("model", "model.h5")
        self.model = load_model(model_path, compile=False)
        # Inference must use the backbone training used, because it decides
        # whether pixels are rescaled here or inside the model
        # (see utils/preprocessing.py). params.yaml is the source of truth;
        # MODEL_BACKBONE lets a container override it without editing files.
        params = read_yaml(PARAMS_FILE_PATH)
        self.backbone = os.getenv("MODEL_BACKBONE") or params.get(
            "BACKBONE", preprocessing.DEFAULT_BACKBONE
        )

    def predict_base64(self, base64_str):
        img_bytes = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_bytes))
        # Builds (1, 224, 224, 3) float32 exactly like the training and
        # evaluation generators do (bilinear resize + backbone scaling mode).
        test_image = preprocessing.prepare_image(img, backbone=self.backbone)

        # Same probability computation as stage-04 evaluation (utils/inference.py):
        # flip-TTA averaged softmax, argmax at 0.5. Evaluation and serving must
        # not drift apart, or the score in scores.json stops describing the
        # model that answers here.
        preds = predict_probabilities(self.model, test_image)
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