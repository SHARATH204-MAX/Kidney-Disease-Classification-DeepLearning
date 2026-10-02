import json
import math
import os
import numpy as np
import tensorflow as tf
from pathlib import Path

from cnnClassifier.entity.config_entity import TrainingConfig
from cnnClassifier.utils import preprocessing

# Written by BestModelCheckpoint; remembers the best metric ACROSS runs so a
# resumed run cannot silently overwrite a better checkpoint.
BEST_MODEL_PATH = Path("artifacts/training/best_model.h5")
BEST_STATE_PATH = Path("artifacts/training/best_model.metrics.json")


class BestModelCheckpoint(tf.keras.callbacks.Callback):
    """Save the model whenever ``monitor`` improves -- and remember the value.

    Keras' own ModelCheckpoint only knows the best value from the *current*
    fit() call, so after a crash/resume it would immediately overwrite a better
    checkpoint written before the crash. We persist the metric next to the
    weights instead.
    """

    def __init__(self, filepath: Path, monitor: str = "val_accuracy"):
        super().__init__()
        self.filepath = Path(filepath)
        self.state_path = BEST_STATE_PATH
        self.monitor = monitor

    def _best(self) -> float:
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            return float(state.get(self.monitor, float("-inf")))
        except (OSError, ValueError, TypeError):
            return float("-inf")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        current = float(current)
        best = self._best()
        if current > best:
            self.model.save(str(self.filepath))
            state = {
                self.monitor: current,
                "epoch": int(epoch) + 1,
                "val_loss": float(logs.get("val_loss", float("nan"))),
            }
            self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            print(
                f"\n{self.monitor} improved {best:.4f} -> {current:.4f} "
                f"(epoch {epoch + 1}); saved {self.filepath}"
            )
        else:
            print(
                f"\n{self.monitor} {current:.4f} <= best {best:.4f}; "
                f"keeping {self.filepath}"
            )


class Training:

    def __init__(self, config: TrainingConfig):
        self.config = config

    # -------------------------------------------------------------
    # LOAD BASE MODEL
    # -------------------------------------------------------------

    def get_base_model(self):

        if self.config.params_resume and BEST_MODEL_PATH.exists():
            # Crash-resume: continue from the best checkpoint of the run that
            # was interrupted, instead of restarting from epoch 1.
            print(f"[training] RESUME_TRAINING=True -> loading {BEST_MODEL_PATH}")
            self.model = tf.keras.models.load_model(
                str(BEST_MODEL_PATH),
                compile=False
            )
        else:
            # A fresh run must not inherit the previous run's checkpoint
            # state: BestModelCheckpoint refuses to save anything worse than
            # the recorded best, which would block checkpointing an entire
            # new run (e.g. after switching backbone) until it happened to
            # beat the old number.
            for stale in (BEST_MODEL_PATH, BEST_STATE_PATH):
                if Path(stale).exists():
                    Path(stale).unlink()
                    print(
                        f"[training] fresh run -> removed stale checkpoint {stale}"
                    )

            # Load WITHOUT the optimizer restored from disk: the legacy .h5 round
            # trip leaves the optimizer built with an empty variable set, and
            # model.fit() then crashes with
            #   "Unknown variable: <Variable path=dense/kernel ...>"
            # Compile a fresh optimizer on the loaded model instead.
            self.model = tf.keras.models.load_model(
                self.config.updated_base_model_path,
                compile=False
            )

        # LABEL_SMOOTHING: soften hard 0/1 targets. The phase-2 model was
        # confidently wrong on 35 of its 169 errors (train acc 0.97 vs val
        # 0.885) -- classic overconfidence; smoothing trades a little training
        # accuracy for better-calibrated, better-generalising probabilities.
        self.label_smoothing = float(
            getattr(self.config, "params_label_smoothing", 0.0) or 0.0
        )
        if self.label_smoothing > 0:
            print(f"[training] label_smoothing={self.label_smoothing:g}")

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.params_learning_rate
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=self.label_smoothing
            ),
            metrics=["accuracy"]
        )

        # ---------------------------------------------------------
        # PHASE-2 FINE-TUNING (UNFREEZE_LAST > 0)
        #
        # Head-only training plateaus with a frozen backbone. Unfreezing the
        # last N backbone layers and continuing at a much lower learning rate
        # lets the high-level features adapt to CT anatomy. Only meaningful
        # when resuming from an existing checkpoint (RESUME_TRAINING: True).
        # ---------------------------------------------------------

        unfreeze = int(getattr(self.config, "params_unfreeze_last", 0) or 0)

        if unfreeze > 0:

            layers = list(self.model.layers)
            head_types = {"GlobalAveragePooling2D", "Dense", "Dropout"}

            # The head built by PrepareBaseModel._prepare_full_model is
            # [GlobalAveragePooling2D, Dense, Dropout, Dense] at the END of the
            # graph, so walk back from the final softmax to find its start.
            #
            # Do NOT search for "the first GlobalAveragePooling2D": EfficientNet
            # has one inside every squeeze-and-excite block (blockNa_se_squeeze
            # sits at index 11 here). The old code found that one, treated the
            # stem as "the backbone", and fine-tuned the FIRST layers -- exactly
            # the opposite of what UNFREEZE_LAST is for (+1,280 params instead
            # of the last blocks).
            head_start = len(layers)
            while head_start > 1 and type(layers[head_start - 1]).__name__ in head_types:
                head_start -= 1

            if (
                head_start >= len(layers)
                or type(layers[head_start]).__name__ != "GlobalAveragePooling2D"
            ):
                raise ValueError(
                    "Cannot locate the classification head: expected a trailing "
                    "[GlobalAveragePooling2D, Dense, Dropout, Dense] run at the "
                    "end of self.model.layers."
                )

            backbone_layers = layers[:head_start]
            unfrozen = backbone_layers[-unfreeze:]

            for layer in unfrozen:
                layer.trainable = True

            finetune_lr = float(
                getattr(self.config, "params_finetune_learning_rate", 1e-5) or 1e-5
            )
            trainable = sum(
                int(np.prod(w.shape)) for w in self.model.trainable_weights
            )

            print(
                f"[training] fine-tuning: unfroze last {len(unfrozen)} of "
                f"{len(backbone_layers)} backbone layers "
                f"({unfrozen[0].name} .. {unfrozen[-1].name}) "
                f"-> {trainable:,} trainable params @ lr={finetune_lr:g}"
            )

            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=finetune_lr),
                loss=tf.keras.losses.CategoricalCrossentropy(
                    label_smoothing=self.label_smoothing
                ),
                metrics=["accuracy"]
            )

    # -------------------------------------------------------------
    # TRAINING AND VALIDATION DATA GENERATORS
    # -------------------------------------------------------------

    def train_valid_generator(self):

        backbone = self.config.params_backbone

        # ---------------------------------------------------------
        # SHARED PREPROCESSING CONTRACT (utils/preprocessing.py)
        #
        # Training, evaluation and inference all pull their pixel scaling
        # from the same helper so they can never drift apart again.
        # ---------------------------------------------------------

        datagenerator_kwargs = preprocessing.generator_kwargs(backbone=backbone)

        dataflow_kwargs = preprocessing.dataflow_kwargs(
            batch_size=self.config.params_batch_size
        )

        print(
            f"[training] backbone={backbone} | "
            f"rescale={datagenerator_kwargs['rescale']} | "
            f"target={dataflow_kwargs['target_size']} | "
            f"interpolation={dataflow_kwargs['interpolation']}"
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

                # Same scaling/split as validation -- only the geometric
                # transforms differ between the two splits.
                **datagenerator_kwargs,

                rotation_range=20,

                width_shift_range=0.10,

                height_shift_range=0.10,

                zoom_range=0.15,

                # Shear + photometric jitter: the train/val gap (0.97 vs 0.885)
                # showed the model was memorising exact pixel layouts. Shear
                # varies apparent kidney orientation; brightness jitter mimics
                # scanner/windowing variation between scans. Both are applied
                # through a uint8 PIL round trip, so outputs stay in [0, 255]
                # and never exceed what EfficientNet's internal Rescaling(1/255)
                # expects.
                shear_range=0.10,

                brightness_range=(0.8, 1.2),

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
        # CLASS WEIGHTS -- counter the Normal/Tumor imbalance
        # ---------------------------------------------------------

        # The dataset is ~69% Normal / 31% Tumor. Unweighted, the optimizer
        # buys accuracy by leaning on the majority class; an unweighted run
        # measured just 0.197 tumor recall at 0.736 overall accuracy.
        #
        # CLASS_WEIGHT_POWER tempers the weighting:
        #   1.0 = full inverse frequency (e.g. 0.72 / 1.61) -- maximises
        #         minority recall but shifts the decision boundary so far
        #         toward "Tumor" that ~198 normals are mislabelled.
        #   0.5 = square-root tempering (0.85 / 1.27) -- keeps a meaningful
        #         minority bias while staying close to the natural prior.
        #   0.0 = no weighting.
        counts = np.bincount(
            np.asarray(self.train_generator.classes), minlength=2
        )
        total = int(counts.sum())
        # 0.0 is a legal value meaning "no weighting", so the default must be
        # applied on None only -- `or 1.0` would coerce an explicit 0.0 to 1.0.
        power_raw = getattr(self.config, "params_class_weight_power", None)
        power = 1.0 if power_raw is None else float(power_raw)
        class_weight = {
            int(i): (total / (len(counts) * int(c))) ** power
            for i, c in enumerate(counts)
        }

        indices = {v: k for k, v in self.train_generator.class_indices.items()}

        print("=" * 60)
        print(f"CLASS WEIGHTS (train split, power={power:g})")
        print("=" * 60)
        for i, c in enumerate(counts):
            label = indices.get(i, str(i))
            print(f"  {label:>8}: {int(c):>5} images -> weight {class_weight[i]:.4f}")
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

            # Save the best model (remembers the best value across resumed
            # runs -- see BestModelCheckpoint)
            BestModelCheckpoint(
                filepath=BEST_MODEL_PATH,
                monitor="val_accuracy"
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

            # Down-weight Normal / up-weight Tumor so the model cannot reach a
            # good loss by ignoring tumors.
            class_weight=class_weight,

            callbacks=callbacks
        )

        # ---------------------------------------------------------
        # RESTORE THE BEST-EVER CHECKPOINT
        # ---------------------------------------------------------

        # EarlyStopping's restore_best_weights only knows the CURRENT fit()
        # call. After a resumed run (phase 2 fine-tuning) it would restore the
        # best *phase-2* epoch even when that epoch is worse than the
        # checkpoint from phase 1 -- and we would then ship that regression as
        # model.h5. Reload the checkpoint that BestModelCheckpoint blessed
        # instead, so the promoted weights are the best seen across all runs.
        if BEST_MODEL_PATH.exists():
            try:
                best = tf.keras.models.load_model(
                    str(BEST_MODEL_PATH), compile=False
                )
                self.model.set_weights(best.get_weights())
                print(
                    f"[training] restored best-ever checkpoint "
                    f"({BEST_MODEL_PATH}) before saving the final model"
                )
            except (ValueError, OSError) as exc:
                print(
                    f"[training] WARNING: could not restore "
                    f"{BEST_MODEL_PATH} ({exc}); saving the final epoch instead"
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