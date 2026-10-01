import math
import tensorflow as tf
from pathlib import Path

from cnnClassifier.entity.config_entity import TrainingConfig


class Training:

    def __init__(self, config: TrainingConfig):
        self.config = config

    # -------------------------------------------------------------
    # LOAD BASE MODEL
    # -------------------------------------------------------------

    def get_base_model(self):

        # Load WITHOUT the optimizer restored from disk: the legacy .h5 round
        # trip leaves the optimizer built with an empty variable set, and
        # model.fit() then crashes with
        #   "Unknown variable: <Variable path=dense/kernel ...>"
        # Compile a fresh optimizer on the loaded model instead.
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path,
            compile=False
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.params_learning_rate
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    # -------------------------------------------------------------
    # TRAINING AND VALIDATION DATA GENERATORS
    # -------------------------------------------------------------

    def train_valid_generator(self):

        # ---------------------------------------------------------
        # VALIDATION CONFIGURATION
        # ---------------------------------------------------------

        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.20
        )

        # ---------------------------------------------------------
        # IMAGE CONFIGURATION
        # ---------------------------------------------------------

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # ---------------------------------------------------------
        # VALIDATION DATA GENERATOR
        # ---------------------------------------------------------

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        # ---------------------------------------------------------
        # TRAINING DATA AUGMENTATION
        # ---------------------------------------------------------

        if self.config.params_is_augmentation:

            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(

                rescale=1.0 / 255,

                validation_split=0.20,

                rotation_range=20,

                width_shift_range=0.10,

                height_shift_range=0.10,

                zoom_range=0.15,

                horizontal_flip=True
            )

        else:

            train_datagenerator = valid_datagenerator

        # ---------------------------------------------------------
        # TRAINING DATA GENERATOR
        # ---------------------------------------------------------

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    # -------------------------------------------------------------
    # SAVE MODEL
    # -------------------------------------------------------------

    @staticmethod
    def save_model(
        path: Path,
        model: tf.keras.Model
    ):

        model.save(path)

    # -------------------------------------------------------------
    # TRAIN MODEL
    # -------------------------------------------------------------

    def train(self):

        # ---------------------------------------------------------
        # CALCULATE STEPS
        # ---------------------------------------------------------

        steps_per_epoch = math.ceil(
            self.train_generator.samples /
            self.train_generator.batch_size
        )

        validation_steps = math.ceil(
            self.valid_generator.samples /
            self.valid_generator.batch_size
        )

        print("\n")
        print("=" * 60)
        print("TRAINING INFORMATION")
        print("=" * 60)

        print(
            f"Training images   : {self.train_generator.samples}"
        )

        print(
            f"Validation images : {self.valid_generator.samples}"
        )

        print(
            f"Batch size        : {self.train_generator.batch_size}"
        )

        print(
            f"Steps per epoch   : {steps_per_epoch}"
        )

        print(
            f"Validation steps  : {validation_steps}"
        )

        print("=" * 60)
        print("\n")

        # ---------------------------------------------------------
        # CALLBACKS
        # ---------------------------------------------------------

        callbacks = [

            # Stop training if validation loss stops improving
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),

            # Reduce learning rate when validation loss stops improving
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=2,
                min_lr=1e-7,
                verbose=1
            ),

            # Save the best model
            tf.keras.callbacks.ModelCheckpoint(
                filepath="artifacts/training/best_model.h5",
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1
            )
        ]

        # ---------------------------------------------------------
        # TRAIN MODEL
        # ---------------------------------------------------------

        history = self.model.fit(

            self.train_generator,

            epochs=self.config.params_epochs,

            steps_per_epoch=steps_per_epoch,

            validation_data=self.valid_generator,

            validation_steps=validation_steps,

            callbacks=callbacks
        )

        # ---------------------------------------------------------
        # SAVE FINAL MODEL
        # ---------------------------------------------------------

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

        print("\n")
        print("=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)