from cnnClassifier.constants import *
import os
from cnnClassifier.utils.common import read_yaml, create_directories,save_json
from cnnClassifier.utils.preprocessing import DEFAULT_BACKBONE
from cnnClassifier.entity.config_entity import (DataIngestionConfig,
                                                PrepareBaseModelConfig,
                                                TrainingConfig,
                                                EvaluationConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        # Optional params keep old params.yaml files working.
        self.backbone = self.params.get("BACKBONE", DEFAULT_BACKBONE)
        self.resume = bool(self.params.get("RESUME_TRAINING", False))

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    


    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
            params_backbone=self.backbone
        )

        return prepare_base_model_config
    



    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "DataSet")
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE,
            params_backbone=self.backbone,
            params_resume=self.resume,
            params_unfreeze_last=int(self.params.get("UNFREEZE_LAST", 0) or 0),
            params_finetune_learning_rate=float(
                self.params.get("FINETUNE_LEARNING_RATE", 1e-5) or 1e-5
            ),
            # NOTE: no `or <default>` here -- CLASS_WEIGHT_POWER/LABEL_SMOOTHING
            # are meaningful at exactly 0.0 (no weighting / no smoothing), and
            # `0.0 or 1.0` would silently turn an explicit 0 into the default.
            params_class_weight_power=(
                1.0 if self.params.get("CLASS_WEIGHT_POWER") is None
                else float(self.params["CLASS_WEIGHT_POWER"])
            ),
            params_label_smoothing=(
                0.0 if self.params.get("LABEL_SMOOTHING") is None
                else float(self.params["LABEL_SMOOTHING"])
            )
        )

        return training_config
    


    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model="artifacts/training/model.h5",
            training_data="artifacts/data_ingestion/DataSet",
            mlflow_uri=os.getenv(
                "MLFLOW_TRACKING_URI",
                "https://dagshub.com/SHARATH204-MAX/Kidney-Disease-Classification-DeepLearning.mlflow",
            ),
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
            params_backbone=self.backbone
        )
        return eval_config