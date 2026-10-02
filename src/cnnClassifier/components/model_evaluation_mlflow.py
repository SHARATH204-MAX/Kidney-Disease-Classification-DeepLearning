import numpy as np
import tensorflow as tf
from pathlib import Path
import mlflow
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils import preprocessing
from cnnClassifier.utils.inference import predict_probabilities
from cnnClassifier.utils.common import read_yaml, create_directories, save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):

        # Same helper the training component uses: identical pixel scaling,
        # geometry, interpolation and split -- otherwise this score is either
        # leaked (split too small) or measured off-distribution (wrong rescale).
        datagenerator_kwargs = preprocessing.generator_kwargs(
            backbone=self.config.params_backbone
        )

        dataflow_kwargs = preprocessing.dataflow_kwargs(
            batch_size=self.config.params_batch_size
        )

        print(
            f"[evaluation] backbone={self.config.params_backbone} | "
            f"rescale={datagenerator_kwargs['rescale']} | "
            f"split={datagenerator_kwargs['validation_split']}"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    def load_model(self, path: Path) -> tf.keras.Model:
        # compile=False: a legacy .h5 round trip restores an optimizer built
        # with an empty variable set, and evaluate()/predict() then crash with
        # "Unknown variable: <Variable path=dense/kernel ...>".
        model = tf.keras.models.load_model(path, compile=False)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                # The optimizer is irrelevant to the metrics below -- it only
                # has to exist for compile() to succeed.
                learning_rate=float(self.config.all_params.get("LEARNING_RATE", 1e-4))
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()

        y_true = np.asarray(self.valid_generator.classes)
        # shuffle=False on the generator => rows align with y_true.
        # Manual batching instead of model.predict(generator) so the flip-TTA
        # pass (utils/inference.py -- shared with the Flask service) sees the
        # exact same batches. This is the number reported in scores.json.
        steps = int(np.ceil(
            self.valid_generator.samples / self.valid_generator.batch_size
        ))
        chunks = []
        self.valid_generator.reset()
        for i in range(steps):
            x, _ = next(self.valid_generator)
            chunks.append(predict_probabilities(self.model, x))
            if (i + 1) % 10 == 0:
                print(f"[evaluation] predicted {i + 1}/{steps} batches (TTA)")
        probs = np.concatenate(chunks, axis=0)
        y_pred = probs.argmax(axis=1)

        num_classes = probs.shape[1]
        onehot = tf.keras.utils.to_categorical(y_true, num_classes=num_classes)

        # Single pass: cross-entropy computed from the same predictions that
        # drive the per-class metrics (saves a full second evaluation pass).
        loss = float(
            tf.keras.losses.categorical_crossentropy(onehot, probs).numpy().mean()
        )
        accuracy = float((y_pred == y_true).mean())

        indices = {v: k for k, v in self.valid_generator.class_indices.items()}
        confusion = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            confusion[int(t), int(p)] += 1

        print("\n" + "=" * 60)
        print("EVALUATION")
        print("=" * 60)
        print("confusion matrix (rows = actual, cols = predicted)")
        header = "        " + "".join(f"{indices[i]:>10}" for i in range(num_classes))
        print(header)
        for i in range(num_classes):
            print(
                f"{indices[i]:>8}"
                + "".join(f"{confusion[i, j]:>10}" for j in range(num_classes))
            )

        scores = {"loss": loss, "accuracy": accuracy}
        for i in range(num_classes):
            tp = int(confusion[i, i])
            support = int(confusion[i].sum())
            predicted = int(confusion[:, i].sum())
            recall = tp / support if support else 0.0
            precision = tp / predicted if predicted else 0.0
            label = str(indices[i]).lower()
            scores[f"precision_{label}"] = float(precision)
            scores[f"recall_{label}"] = float(recall)
            print(
                f"  {indices[i]:>8}: precision={precision:.4f} "
                f"recall={recall:.4f} support={support}"
            )

        scores["confusion_matrix"] = confusion.tolist()
        print(f"\n  loss={loss:.4f}  accuracy={accuracy:.4f}")
        print("=" * 60 + "\n")

        # Kept for log_into_mlflow(), which indexes score[0] / score[1].
        self.score = [loss, accuracy]
        self.scores = scores
        self.save_score()

    def save_score(self):
        save_json(path=Path("scores.json"), data=self.scores)

    def log_into_mlflow(self):
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_registry_uri(self.config.mlflow_uri)

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": float(self.score[0]), "accuracy": float(self.score[1])}
            )
            mlflow.log_artifact(str(self.config.path_of_model), artifact_path="model")
