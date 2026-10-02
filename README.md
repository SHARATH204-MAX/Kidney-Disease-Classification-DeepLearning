<div align="center">

# Kidney Disease Classification — Deep Learning Service

**Transfer-learning (EfficientNetB0) binary classifier for kidney CT scans · Reproducible ML pipeline (DVC) · Experiment tracking (MLflow / DagsHub) · REST inference service (Flask)**

[![Workflow Status](https://github.com/SHARATH204-MAX/Kidney-Disease-Classification-DeepLearning/actions/workflows/main.yaml/badge.svg)](https://github.com/SHARATH204-MAX/Kidney-Disease-Classification-DeepLearning/actions/workflows/main.yaml)
[![DVC](https://img.shields.io/badge/pipeline-DVC-1F7A3F?logo=dvc&logoColor=white)](https://dvc.org)
[![MLflow](https://img.shields.io/badge/tracking-MLflow%20%2F%20DagsHub-01cce4)](https://mlflow.org)
[![TensorFlow](https://img.shields.io/badge/model-TensorFlow%20%2F%20Keras-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org)
[![Python](https://img.shields.io/badge/python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![Flask](https://img.shields.io/badge/serving-Flask-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
  - [System / Serving Architecture](#1-system--serving-architecture)
  - [ML Training Pipeline](#2-ml-training-pipeline-dvc-stages)
  - [Model Architecture](#3-model-architecture)
  - [Preprocessing Contract](#4-preprocessing-contract--invariants)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Running the Project](#running-the-project)
- [API Reference](#api-reference)
- [Configuration Reference](#configuration-reference)
- [Experiment Tracking & Versioning](#experiment-tracking--versioning)
- [Evaluation & Metrics](#evaluation--metrics)
- [Docker & CI/CD](#docker--cicd)
- [Model Results](#model-results)
- [Performance & Hardware Notes](#performance--hardware-notes)
- [Troubleshooting — Known Pitfalls](#troubleshooting--known-pitfalls)
- [Roadmap](#roadmap)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview

This repository contains an end-to-end, production-oriented deep-learning service that classifies
kidney CT scan images as **`Normal`** or **`Tumor`**.

It is built as four independently reproducible pipeline stages orchestrated by **DVC**, a Flask
inference service with a browser UI, MLflow experiment tracking via DagsHub, and a GitHub Actions
workflow that builds a Docker image, pushes it to Amazon ECR and deploys it to a self-hosted EC2
runner.

| Item | Value |
|---|---|
| Task | Binary image classification (`Normal` / `Tumor`) |
| Backbone | EfficientNetB0 (ImageNet weights) + custom classification head · switchable via `params.yaml -> BACKBONE` (`VGG16` also supported) |
| Training strategy | Two-phase: (1) frozen backbone, head-only, (2) fine-tune the last 30 backbone layers at `lr=5e-6` (`UNFREEZE_LAST`, `FINETUNE_LEARNING_RATE`) |
| Dataset | 7,360 CT images — 5,077 `Normal`, 2,283 `Tumor`, 512×512 JPEG |
| Split | 80 / 20 stratified-by-folder → 5,889 train · 1,471 validation |
| Input tensor | `224 × 224 × 3`, RGB, float32 **raw `0–255`** (EfficientNetB0 rescales internally; VGG16 needs `[0,1]` — see the [contract](#4-preprocessing-contract--invariants)) |
| Total params | 4,378,021 — 328,450 trainable in phase 1 (head only), 1,825,890 in phase 2 (head + last 30 backbone layers) |
| Class weights | Inverse-frequency weighting tempered by `CLASS_WEIGHT_POWER` (`0.25` → `Normal 0.92`, `Tumor 1.13`) to counter the 69/31 imbalance without distorting the decision boundary |
| Serving | Flask REST API + HTML UI on port `8090` (`8080` in Docker) |

> **Class balance warning.** The dataset is ~69% / 31%. A model that predicts `Normal` for every
> image already "scores" 69% accuracy — a real checkpoint once reached 73.6% accuracy while
> missing **80% of tumors**. Always read accuracy alongside per-class recall — see
> [Evaluation & Metrics](#evaluation--metrics).

---

## Key Features

- **Reproducible pipeline** — `dvc repro` rebuilds data, base model, trained weights and metrics
  only when code/params/data change; `dvc.lock` pins every stage's hash.
- **Typed configuration layer** — YAML → `ConfigurationManager` → frozen `@dataclass` config
  entities; components never read YAML directly.
- **Clean layering** — `components` (business logic) → `pipeline` (stage entry points) →
  `main.py` / `app.py` (orchestration).
- **Training hygiene** — augmentation (rotation / shift / zoom / shear / brightness jitter /
  flip), label smoothing (`LABEL_SMOOTHING`), EarlyStopping with weight restoration,
  ReduceLROnPlateau, best-model checkpointing on `val_accuracy` that survives resume.
- **Two-phase fine-tuning** — train the head on a frozen backbone, then resume
  (`RESUME_TRAINING`) with `UNFREEZE_LAST` layers unfrozen at a much lower learning rate; the
  unfreeze walks back from the softmax so it always hits the *tail*, never the stem.
- **Consistent preprocessing** — training, evaluation and inference all pull their input contract
  from `utils/preprocessing.py`, so they cannot drift apart (the single most common source of
  silent accuracy loss — see [Troubleshooting](#troubleshooting--known-pitfalls)).
- **Imbalance-aware training** — tempered inverse-frequency class weights
  (`CLASS_WEIGHT_POWER`) plus per-class reporting in `scores.json`, so accuracy cannot hide a
  model that ignores tumors.
- **Train/serve parity for inference** — evaluation and the API share
  `utils/inference.py::predict_probabilities()` (flip-TTA averaged softmax), so the metric in
  `scores.json` is literally the model's deployed behaviour: **90.96% accuracy / 0.969 tumor
  recall** versus 88.58% without TTA — see [Model Results](#model-results).
- **Experiment tracking** — MLflow runs (params + metrics) pushed to DagsHub; `scores.json`
  exposed as a DVC metric.
- **Containerized delivery** — Dockerfile + CI/CD to ECR → EC2.
- **Browser UI** — drag & drop upload, client-side preview, single-click analysis.

---

## Architecture

### 1. System / Serving Architecture

```mermaid
flowchart LR
    subgraph Client["Client tier"]
        UI["Browser UI<br/>templates/index.html<br/>drag and drop · JPEG ≤ 500px · base64"]
    end

    subgraph Serving["Serving tier — Flask (app.py :8090)"]
        HOME["GET /<br/>renders UI"]
        PRED["POST /predict<br/>JSON { image: base64 }"]
        TRAIN["POST /train<br/>runs full pipeline"]
        PIPE["PredictionPipeline<br/>src/cnnClassifier/pipeline/prediction.py"]
    end

    subgraph ModelStore["Model store"]
        M1["artifacts/training/model.h5<br/>(preferred — freshly trained)"]
        M2["model/model.h5<br/>(packaged fallback)"]
    end

    UI -- "base64 JPEG" --> PRED
    HOME --> UI
    PRED --> PIPE
    PIPE --> M1
    PIPE -. "fallback" .-> M2
    TRAIN --> MAIN["python main.py<br/>4-stage pipeline"]
    MAIN -. "writes weights" .-> M1
```

### 2. ML Training Pipeline (DVC stages)

```mermaid
flowchart TD
    S1["<b>stage 01 — data_ingestion</b><br/>gdown fetch + unzip<br/>artifacts/data_ingestion/DataSet/"]
    S2["<b>stage 02 — prepare_base_model</b><br/>EfficientNetB0(ImageNet, include_top=False)<br/>+ GAP + Dense(256) + Dropout(0.5) + Dense(2)<br/>freeze backbone · compile Adam(1e-4)"]
    S3["<b>stage 03 — model_training</b><br/>augmented generator · backbone scaling<br/>class weights · label smoothing<br/>head-only → fine-tune tail (UNFREEZE_LAST)<br/>EarlyStopping(5) · ReduceLR · BestCheckpoint"]
    S4["<b>stage 04 — evaluation</b><br/>hold-out generator (shuffle=False)<br/>loss · accuracy · per-class recall"]
    MET[("scores.json<br/>DVC metric")]
    MLFLOW[("MLflow / DagsHub<br/>params + metrics")]
    W[("artifacts/training/model.h5<br/>+ best_model.h5")]
    SERVE["Flask POST /predict"]

    S1 --> S2 --> S3 --> S4
    S3 --> W --> SERVE
    S4 --> MET
    S4 -.-> MLFLOW
```

**Artifacts produced per stage**

| Stage | Inputs | Outputs |
|---|---|---|
| `data_ingestion` | Google Drive archive (see `config/config.yaml`) | `artifacts/data_ingestion/DataSet/{Normal,Tumor}/` |
| `prepare_base_model` | `params.yaml` | `artifacts/prepare_base_model/base_model.h5`, `base_model_updated.h5` |
| `training` | base model + dataset | `artifacts/training/model.h5`, `best_model.h5` |
| `evaluation` | trained model + dataset | `scores.json` (+ optional MLflow run) |

### 3. Model Architecture

```mermaid
flowchart LR
    IN["Input<br/>224×224×3 RGB<br/>raw pixels 0 – 255"] --> VGG["EfficientNetB0 · ImageNet<br/><i>frozen</i> · internal Rescaling(1/255) · 4.05M params"]
    VGG --> GAP["GlobalAveragePooling2D<br/>→ 1280"]
    GAP --> D1["Dense 256 · ReLU"]
    D1 --> DO["Dropout 0.5"]
    DO --> D2["Dense 2 · Softmax"]
    D2 --> O1["0 → Normal"]
    D2 --> O2["1 → Tumor"]
```

| Layer | Output shape | Params | Trainable |
|---|---|---:|---|
| `InputLayer` | `(None, 224, 224, 3)` | 0 | — |
| `EfficientNetB0` (`include_top=False`, incl. `Rescaling`/`Normalization`) | `(None, 7, 7, 1280)` | 4,049,571 | ✗ frozen |
| `GlobalAveragePooling2D` | `(None, 1280)` | 0 | — |
| `Dense(256, relu)` | `(None, 256)` | 327,936 | ✓ |
| `Dropout(0.5)` | `(None, 256)` | 0 | — |
| `Dense(2, softmax)` | `(None, 2)` | 514 | ✓ |
| **Total** | | **4,378,021** | **328,450 trainable** |

- **Loss:** categorical cross-entropy · **Optimizer:** Adam (`LEARNING_RATE: 0.0001`)
- **Class weights:** inverse frequency of the training split (`Normal 0.725`, `Tumor 1.610`) so a
  good loss cannot be reached by ignoring tumors
- **Augmentation:** rotation 20°, width/height shift 0.10, zoom 0.15, horizontal flip
- **Callbacks:** EarlyStopping(`val_loss`, patience 5, `restore_best_weights=True`),
  ReduceLROnPlateau(`val_loss`, factor 0.2, patience 2, `min_lr=1e-7`),
  `BestModelCheckpoint` → `artifacts/training/best_model.h5` (max `val_accuracy`, remembers the
  best value **across resumed runs**)

### 4. Preprocessing Contract — Invariants

These rules must hold in **training, evaluation and inference simultaneously**. Breaking any of
them produces a model or service that looks runnable but predicts garbage — this project has hit
that failure twice, so all three call sites now import the same helper module:
**`src/cnnClassifier/utils/preprocessing.py`**.

1. **Pixel scaling is backbone-dependent** — `ImageDataGenerator`, the evaluator and
   `prediction.py` all call `preprocessing.generator_kwargs()` / `preprocessing.prepare_image()`:

   | Backbone | Scaling | Why |
   |---|---|---|
   | `EfficientNetB0` (default) | **raw `0–255`** (`rescale=None`) | the model's first layer is `Rescaling(1/255)`; pre-dividing would feed it ~`[0, 0.004]` |
   | `VGG16` | `rescale=1./255` → `[0, 1]` | no internal rescaling |

   Never divide by 255 "to be safe" — double-scaling and never-scaling are both silent killers.
2. **Geometry** — resize to `224 × 224` with **bilinear** interpolation
   (`preprocessing.INTERPOLATION`), in the generator and in `prediction.py`.
3. **Class indices** — `flow_from_directory` sorts folder names alphabetically ⇒
   **`Normal = 0`, `Tumor = 1`**. `argmax == 1` is reported as `Tumor`.
4. **Optimizer state** — a model loaded from legacy `.h5` must be loaded with `compile=False` and
   re-compiled before `fit()` (see [Troubleshooting](#troubleshooting--known-pitfalls)).
5. **Backbone agreement** — `params.yaml -> BACKBONE` must match the weights on disk. Adding a
   backbone requires registering it in *both* `components/prepair_base_model.py::BACKBONES` and
   `utils/preprocessing.py::BACKBONE_SCALING`.
6. **Probability computation** — evaluation and serving both derive probabilities through
   **`src/cnnClassifier/utils/inference.py::predict_probabilities()`** (flip-TTA averaged softmax).
   Same reason as the pixel contract: if `scores.json` is computed one way and the API answers
   another way, the published metric stops describing the shipped model.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Deep learning | TensorFlow / Keras 3, EfficientNetB0 (ImageNet transfer learning; VGG16 optional) |
| Data handling | `ImageDataGenerator`, NumPy, Pillow, OpenCV/gdown for ingestion |
| Orchestration | DVC pipelines (`dvc.yaml` + `dvc.lock`) |
| Experiment tracking | MLflow, DagsHub remote tracking |
| Serving | Flask, Flask-CORS, Jinja2 (`templates/index.html`) |
| Configuration | YAML (`config/config.yaml`, `params.yaml`) + `python-box`, `python-dotenv` |
| Packaging | `setuptools` (`setup.py`, `pip install -e .`) |
| Containerization | Docker (`python:3.8-slim-buster`) |
| CI/CD | GitHub Actions → Amazon ECR → EC2 self-hosted runner |

---

## Project Structure

```text
Kidney-Disease-Classification-DeepLearning/
│
├── app.py                          # Flask entry point — serving tier (/, /predict, /train)
├── main.py                         # Orchestrates the 4 pipeline stages end-to-end
├── params.yaml                     # Hyper-parameters (DVC-tracked): EPOCHS, LR, BATCH_SIZE,
│                                   #   BACKBONE, RESUME_TRAINING…
├── Dockerfile                      # Container image: python:3.8-slim + requirements + app
├── requirements.txt                # Runtime dependencies
├── setup.py                        # Package definition (src/ layout, `pip install -e .`)
├── template.py                     # Scaffolding utility for new project files
├── scores.json                     # Latest evaluation metrics (DVC metric)
├── inputImage.jpg                  # Sample input for smoke-testing /predict
├── dvc.yaml                        # DVC pipeline definition (4 stages)
├── dvc.lock                        # Locked stage hashes + params for reproducibility
├── .dvcignore / .dvc/              # DVC repository metadata
├── .env                            # ⚠ secrets: MLflow/DagsHub credentials (git-ignored)
│
├── config/
│   └── config.yaml                 # Path configuration (roots, file locations, data source)
│
├── src/cnnClassifier/              # Application package
│   ├── constants/__init__.py       # CONFIG_FILE_PATH, PARAMS_FILE_PATH
│   ├── entity/config_entity.py     # Frozen dataclasses: DataIngestion/PrepareBaseModel/
│   │                               #   Training/Evaluation configs
│   ├── config/configuration.py     # ConfigurationManager — YAML → typed entities
│   ├── utils/common.py             # read_yaml, create_directories, save_json, encode/decode…
│   ├── utils/preprocessing.py       # ⚠ single source of truth for the input contract
│   ├── utils/inference.py           # ⚠ single source of truth for probabilities (flip-TTA)
│   │                               #   (target size, interpolation, split, backbone scaling)
│   ├── components/                 # Business logic (one class per stage)
│   │   ├── data_ingestion.py       #   download · skip-if-exists · unzip
│   │   ├── prepair_base_model.py   #   backbone (EfficientNetB0/VGG16) + head + compile
│   │   ├── model_training.py       #   generators · class weights · callbacks · fit · save
│   │   └── model_evaluation_mlflow.py  # hold-out evaluation · scores.json · MLflow
│   └── pipeline/                   # Thin, importable stage wrappers
│       ├── stage_01_data_ingestion.py
│       ├── stage_02_prepair_base_model.py
│       ├── stage_03_model_training.py
│       ├── stage_04_model_evaluation.py
│       └── prediction.py           #   PredictionPipeline — load weights · preprocess · TTA probs · argmax
│
├── templates/
│   └── index.html                  # Single-page UI (upload · preview · results panel)
│
├── research/                       # Exploratory notebooks, one per stage
│   ├── 01_data_ingestion.ipynb
│   ├── 02_prepare_base_model.ipynb
│   ├── 03_model_tarining.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── trials.ipynb
│
├── artifacts/                      # Pipeline outputs (git-ignored, DVC-tracked)
│   ├── data_ingestion/DataSet/     #   Normal/ (5,077) · Tumor/ (2,283)
│   ├── prepare_base_model/         #   base_model.h5 · base_model_updated.h5
│   └── training/                   #   model.h5 · best_model.h5
│
├── model/
│   └── model.h5                    # Packaged weights used as fallback by the service
│
├── logs/running_logs.log           # Structured application log (stages + HTTP requests)
├── mlruns/                         # Local MLflow run store
│
├── .github/workflows/main.yaml     # CI/CD: lint+test → build/push ECR → deploy
└── .gitignore                      # ignores .env, .venv, artifacts/*, *.log
```

**Layering rule:** `pipeline/*` never contains logic; it wires `ConfigurationManager` →
`components/*`. `components/*` never read YAML directly; they receive typed config entities.

---

## Getting Started

### Prerequisites

- Python **3.8+** (developed and CI-built on 3.8; tested on 3.10+)
- ~2 GB free RAM for inference, ~4 GB for training
- *(Optional)* AWS credentials for CI/CD, DagsHub token for MLflow logging
- *(Optional)* `dvc` for running the pipeline reproducibly

### 1. Clone

```bash
git clone https://github.com/SHARATH204-MAX/Kidney-Disease-Classification-DeepLearning.git
cd Kidney-Disease-Classification-DeepLearning
```

### 2. Create the environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt      # installs the package itself (-e .)
```

### 4. Configure secrets

Create a `.env` file in the repository root (git-ignored):

```dotenv
MLFLOW_TRACKING_URI=https://dagshub.com/<owner>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
```

> Stage 04 refuses to run without `MLFLOW_TRACKING_PASSWORD` (it rejects empty or `PASTE_*`
> placeholder values). MLflow logging itself is invoked from
> `stage_04_model_evaluation.py` — enable it by uncommenting `evaluation.log_into_mlflow()`.

### 5. Fetch the dataset (if `artifacts/` is empty)

Data ingestion is idempotent — it skips the download when
`artifacts/data_ingestion/data.zip` already exists:

```bash
python main.py            # runs all four stages (data → base model → train → evaluate)
# or just the first stage:
python src/cnnClassifier/pipeline/stage_01_data_ingestion.py
```

---

## Running the Project

### Serve the web application

```bash
python app.py
# → http://127.0.0.1:8090
```

Open the URL in a browser, drag a CT image onto the drop zone and click **Analyze Scan**.

### Run / retrain the full pipeline

```bash
python main.py            # data ingestion → prepare base model → training → evaluation
dvc repro                 # equivalent, but only re-runs changed stages
```

Retrain via the API (blocking — runs the whole pipeline before responding):

```bash
curl -X POST http://127.0.0.1:8090/train
```

> **Weights are loaded once at process start.** After any retrain, restart `app.py` so the service
> picks up the new `artifacts/training/model.h5`.

### Run a single stage

```bash
python src/cnnClassifier/pipeline/stage_02_prepair_base_model.py
python src/cnnClassifier/pipeline/stage_03_model_training.py
python src/cnnClassifier/pipeline/stage_04_model_evaluation.py
```

### Useful DVC commands

```bash
dvc repro            # run the pipeline
dvc dag              # print the stage DAG
dvc exp show         # compare experiments/params
dvc status           # show what is out of date
```

### Exploratory notebooks

```bash
jupyter notebook research/
```

---

## API Reference

### `GET /`

Renders the UI (`templates/index.html`).

### `POST /predict`

Classifies one image.

| | |
|---|---|
| **Content-Type** | `application/json` |
| **Body** | `{ "image": "<base64-encoded image bytes>" }` — **raw base64 only**; the UI strips any `data:image/...;base64,` prefix before sending |
| **Success** | `200` · `[{"image": "Normal"}]` or `[{"image": "Tumor"}]` |

```bash
# example
python - <<'PY'
import base64, json, urllib.request
b64 = base64.b64encode(open("inputImage.jpg", "rb").read()).decode()
req = urllib.request.Request(
    "http://127.0.0.1:8090/predict",
    data=json.dumps({"image": b64}).encode(),
    headers={"Content-Type": "application/json"},
)
print(urllib.request.urlopen(req).read().decode())
PY
```

> Server-side preprocessing lives in `preprocessing.prepare_image` (the same helper training and
> evaluation use): decode → resize `224×224` (bilinear) → backbone scaling (raw `0–255` for
> EfficientNetB0, `/255` for VGG16) → batch dimension → flip-TTA averaged softmax
> (`utils/inference.py`, identical to stage 04) → `argmax`.

### `GET|POST /train`

Runs `python main.py` synchronously and responds
`Training done successfully!` when the pipeline finishes. Requests block for the entire run —
for long trainings prefer `python main.py` in a terminal or `dvc repro`.

---

## Configuration Reference

### `params.yaml` — hyper-parameters (tracked by DVC)

| Key | Default | Meaning |
|---|---|---|
| `AUGMENTATION` | `True` | Training-split augmentation: rotation 20°, shift 0.10, zoom 0.15, shear 0.10, brightness ×0.8–1.2, horizontal flip |
| `IMAGE_SIZE` | `[224, 224, 3]` | Model input shape (both backbones' native resolution) |
| `BATCH_SIZE` | `32` | Mini-batch size (185 steps/epoch on 5,889 images) |
| `INCLUDE_TOP` | `False` | Drop the backbone's classifier head |
| `EPOCHS` | `20` | Maximum epochs (EarlyStopping may stop sooner) |
| `CLASSES` | `2` | Output units (`Normal`, `Tumor`) |
| `WEIGHTS` | `imagenet` | Backbone initialization |
| `LEARNING_RATE` | `0.0001` | Adam learning rate (phase 1) |
| `BACKBONE` | `EfficientNetB0` | Which backbone to train — `EfficientNetB0` \| `VGG16`. Drives the pixel-scaling contract everywhere |
| `RESUME_TRAINING` | `False` | `True` → continue from `artifacts/training/best_model.h5` instead of starting over (also the switch that starts phase 2) |
| `UNFREEZE_LAST` | `0` | Trailing backbone layers to unfreeze for fine-tuning; `0` = keep the backbone frozen (phase 1) |
| `FINETUNE_LEARNING_RATE` | `1e-5` | Adam learning rate used when unfreezing (phase 2) |
| `CLASS_WEIGHT_POWER` | `1.0` | Tempering exponent on inverse-frequency class weights: `1.0` full, `0.25` mild, `0.0` off |
| `LABEL_SMOOTHING` | `0.0` | Label smoothing for the cross-entropy loss; `0.1` used in training to curb overconfident errors |

### `config/config.yaml` — paths

| Key | Value |
|---|---|
| `artifacts_root` | `artifacts` |
| `data_ingestion.local_data_file` | `artifacts/data_ingestion/data.zip` |
| `data_ingestion.unzip_dir` | `artifacts/data_ingestion` |
| `prepare_base_model.base_model_path` | `artifacts/prepare_base_model/base_model.h5` |
| `prepare_base_model.updated_base_model_path` | `artifacts/prepare_base_model/base_model_updated.h5` |
| `training.trained_model_path` | `artifacts/training/model.h5` |

### Environment variables

| Variable | Purpose |
|---|---|
| `MLFLOW_TRACKING_URI` | MLflow backend (DagsHub by default) |
| `MLFLOW_TRACKING_USERNAME` | MLflow auth user |
| `MLFLOW_TRACKING_PASSWORD` | MLflow auth token — **required by stage 04** |
| `TF_ENABLE_ONEDNN_OPTS` | Set to `0` by `app.py` for deterministic inference |
| `CUDA_VISIBLE_DEVICES` | Set to `-1` by `app.py` (CPU-only serving) |
| `MODEL_BACKBONE` | Optional override of `params.yaml -> BACKBONE` for the service (containers) |

---

## Experiment Tracking & Versioning

**DVC** owns *data, models and metrics*:

- `dvc.yaml` declares the four stages and which params each depends on.
- `dvc.lock` freezes code/param/data hashes so a pipeline run is reproducible.
- `scores.json` is registered under `metrics:` and compared across runs with `dvc exp show`.

**MLflow / DagsHub** owns *experiment lineage*:

- every evaluation logs `all_params` from `params.yaml` plus `loss` and `accuracy`;
- artifacts (`model.h5`) are attached to the run;
- local copies live under `mlruns/`; remote runs appear in the DagsHub UI.

```bash
mlflow ui          # local tracking UI (defaults to ./mlruns)
```

---

## Evaluation & Metrics

The evaluation stage builds a **hold-out generator from the same helper as training**:

```python
from cnnClassifier.utils import preprocessing

ImageDataGenerator(**preprocessing.generator_kwargs(backbone))   # scaling + split
    .flow_from_directory(..., subset="validation", shuffle=False,
                         **preprocessing.dataflow_kwargs(batch_size))
```

- `shuffle=False` keeps predictions aligned with `generator.classes`.
- `validation_split=0.20` **must match training** — using a larger split would evaluate on images
  the model has already seen (leakage).
- Probabilities come from `utils/inference.py::predict_probabilities()` — the **same flip-TTA
  code path the Flask API uses** — over manually pulled batches, so loss, accuracy and the
  confusion matrix all derive from one set of predictions.

**Reading the numbers**

| Metric | Location | Notes |
|---|---|---|
| `loss`, `accuracy` | `scores.json` | Written by stage 04; DVC metric |
| `precision_normal`, `recall_normal`, `precision_tumor`, `recall_tumor`, `confusion_matrix` | `scores.json` | Same file — per-class detail, also DVC-readable |
| `loss`, `accuracy` | MLflow run | Same values, plus full param set (when `log_into_mlflow()` is enabled) |
| Training curve | console / `logs/running_logs.log` | Per-epoch `loss`, `val_loss`, `val_accuracy` |

Because of the 69/31 class imbalance:

- treat **accuracy ≈ 0.69** as *"predicts `Normal` every time"* until proven otherwise;
- a healthy model shows `accuracy` well above the class prior **and** `val_loss` far below
  `ln(2) ≈ 0.693`;
- **judge the model by `recall_tumor`** — one checkpoint hit 73.6% accuracy while missing 366 of
  456 tumors (recall 0.197). Accuracy alone would have called that run "close".

---

## Model Results

Measured by stage 04 on the validation split — **1,471 images (1,015 `Normal` / 456 `Tumor`)**,
single 80/20 split, no test-set leakage.

### Headline

| Configuration | Accuracy | `recall_tumor` | `recall_normal` |
|---|---:|---:|---:|
| **Shipped: flip-TTA, threshold 0.50** | **0.9096** | **0.9693** | 0.8828 |
| Same weights, no TTA (plain argmax) | 0.8858 | 0.9518 | 0.8562 |
| Optional high-accuracy point: TTA, threshold 0.60 | 0.9211 | 0.9123 | 0.9251 |

Confusion matrix at the shipped configuration (rows = actual, cols = predicted):

|  | → Pred `Normal` | → Pred `Tumor` |
|---|---:|---:|
| **Actual `Normal`** | 896 | 119 |
| **Actual `Tumor`** | 14 | 442 |

> The old failure mode is gone in both directions: the legacy checkpoint missed **366 of 456
> tumors**; this one misses **14**, while only 119 of 1,015 normals trigger a (recoverable)
> false alarm.

### Accuracy journey

| Run | Accuracy | `recall_tumor` | What changed |
|---|---:|---:|---|
| Legacy VGG16 checkpoint | 0.7360 | 0.197 | collapsed toward the 69% majority class |
| Phase 1 — EfficientNetB0, frozen backbone | 0.8477 | 0.943 | backbone switch + class weights |
| Phase 2 — fine-tune last 30 layers @ `1e-5` | 0.8851 | 0.950 | +3.7 pts |
| Round 3 — label smoothing · shear/brightness aug · `CLASS_WEIGHT_POWER: 0.25` @ `5e-6` | 0.8858 | 0.952 | +0.1 pt; regularisation held the train/val gap |
| **+ flip-TTA (shared eval/serve path)** | **0.9096** | **0.969** | **+2.4 pts — target crossed** |

**Notes**

- TTA = `mean(softmax(x), softmax(hflip(x)))` in `utils/inference.py`; horizontal mirroring is
  anatomically valid for CT slices. It is *not* a reporting trick: stage 04 and the Flask API
  call the same function, so `scores.json` describes deployed behaviour.
- The threshold 0.60 row is a documented operating-point trade-off
  (full sweep: `artifacts/training/operating_point_suite.json`), **not** the shipped default —
  it buys +1.15 accuracy while giving up 26 detected tumors.
- Threshold and TTA were both validated on the same split used for reporting (mild optimism);
  a locked-off test split is on the [Roadmap](#roadmap).

---

## Docker & CI/CD

### Build & run locally

```bash
docker build -t cnncls .
docker run -d -p 8080:8080 --name cnncls cnncls
```

### GitHub Actions (`.github/workflows/main.yaml`)

| Job | Runner | Purpose |
|---|---|---|
| `integration` | `ubuntu-latest` | Checkout → lint → unit tests *(placeholders)* |
| `build-and-push-ecr-image` | `ubuntu-latest` | Configure AWS → login ECR → `docker build` → push `:latest` |
| `Continuous-Deployment` | `self-hosted` | Pull image from ECR → `docker run -p 8080:8080` → prune stale images |

Required repository secrets:
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `AWS_ECR_LOGIN_URI`,
`ECR_REPOSITORY_NAME`.

> **Port note:** `app.py` binds `8090` locally, while the workflow/Dockerfile target `8080`.
> Align `app.run(port=...)` with the container port mapping before deploying.

---

## Performance & Hardware Notes

Measured on a 12-logical-core CPU-only machine (no GPU), `BATCH_SIZE=32`, 185 steps/epoch:

| Phase | Per-step | Per-epoch | Notes |
|---|---:|---:|---|
| Phase 1 — head only (backbone frozen) | ~1.3 s | ~4 min | 39 min total for a full run with EarlyStopping |
| Phase 2 — fine-tuning last 30 layers | 2–4 s | 7–12 min | ~40 s validation pass; forward+backward now cover 1.8 M trainable params |
| Evaluation (stage 04) | ~0.7 s | ~40 s total | 1,471 validation images in one `predict()` pass |

- Training is the bottleneck: expect roughly **1–2 hours** for a resume-based fine-tuning run
  with EarlyStopping, versus minutes for head-only training.
- Inference is cheap: a single `predict` call takes well under a second after the model is loaded.
- Levers if you need speed: raise `BATCH_SIZE` (CPU is rarely saturated), enable GPU
  (`CUDA_VISIBLE_DEVICES`), lower `EPOCHS` for smoke tests, or run `dvc repro` so unchanged
  stages are skipped.
- Memory: ~2 GB RSS while training, ~0.7 GB while serving.

---

## Troubleshooting — Known Pitfalls

| Symptom | Root cause | Fix |
|---|---|---|
| Every prediction returns the same label; `scores.json` accuracy ≈ 0.69 | Training was run with `EPOCHS=1` / `LEARNING_RATE=0.01` — the head diverged (loss 10–15) and collapsed to the majority class | Train with the configured `EPOCHS: 20` / `LEARNING_RATE: 0.0001`, then verify `val_loss` drops below 0.693 |
| High accuracy but terrible `recall_tumor` | Unweighted training on a 69/31 split lets the model buy accuracy by ignoring `Tumor` (measured: 73.6% acc, 19.7% recall) | Keep class weights enabled in `components/model_training.py`; judge runs by `recall_tumor` |
| Predictions random/wrong although training accuracy looks fine | Inference and training disagree about pixel scaling (one feeds `0–255`, the other `0–1`) | Never hand-code scaling: call `utils/preprocessing.py` from training, evaluation **and** `prediction.py` |
| Predictions collapse after switching `BACKBONE` | New backbone has a different input contract (e.g. EfficientNetB0 already divides by 255 internally) while stale weights/preprocessing remain | Keep `BACKBONE` in sync across `params.yaml`, the weights on disk, and the preprocessing helper |
| Fine-tuning barely moves the needle (~1,280 params train instead of millions) | The unfreeze searched for the *first* `GlobalAveragePooling2D`; in EfficientNet every squeeze-and-excite block has one, so the stem got unfrozen instead of the tail | Walk back from the final softmax over the trailing `[GAP, Dense, Dropout, Dense]` run — `components/model_training.py` prints first/last unfrozen layer names, verify they end at `top_activation` |
| `ValueError: Unknown variable: <Variable path=dense/kernel …> — This optimizer can only be called for the variables it was originally built with` | Keras 3 legacy-`.h5` round-trip leaves the restored optimizer built with an empty variable set | `load_model(path, compile=False)` then call `model.compile(...)` with a fresh optimizer (`components/model_training.py`) |
| Evaluation accuracy higher than it should be | Eval `validation_split=0.30` vs training `0.20` ⇒ ~⅓ of evaluated images were trained on | Keep both splits identical (`0.20`) |
| Retrained but the app still predicts like before | Weights are loaded at process start; `model/model.h5` is a packaged copy | Restart `app.py`; `PredictionPipeline` now prefers `artifacts/training/model.h5` |
| `/train` request hangs for hours | The route runs the entire pipeline synchronously | Run `python main.py` / `dvc repro` in a background terminal |
| Stage 04 fails immediately | Missing/placeholder `MLFLOW_TRACKING_PASSWORD` | Fill in `.env` (see [Getting Started](#getting-started)) |
| Data stage tries to re-download | `artifacts/data_ingestion/data.zip` missing | Restore `artifacts/` or allow the download; the stage skips when the archive exists |

---

## Roadmap

- [x] Per-class metrics: precision / recall, confusion matrix in `scores.json`
- [x] Class-weighted loss to counter the 69/31 imbalance
- [x] Single shared preprocessing contract (`utils/preprocessing.py`) for train/eval/serve
- [x] Crash-resume: `RESUME_TRAINING: True` continues from `best_model.h5`
- [x] Fine-tuning phase: `UNFREEZE_LAST` unfreezes the trailing backbone layers at
      `FINETUNE_LEARNING_RATE`; the unfreeze walks back from the softmax (EfficientNet's
      squeeze-and-excite blocks each contain a GAP layer, so "first GAP" finds the stem)
- [x] Anti-overfitting levers: label smoothing (`LABEL_SMOOTHING`) + shear/brightness
      augmentation, gated by measured `CLASS_WEIGHT_POWER`
- [ ] ROC-AUC / F1 aggregation in `scores.json`
- [ ] Migrate legacy `.h5` artefacts to the native `.keras` format
- [ ] Real unit/integration tests replacing the CI placeholders
- [ ] GPU-enabled training (CUDA) and batched prediction endpoint
- [ ] Model registry / promotion flow (staging → production) via MLflow
- [ ] Docker health-check endpoint and non-root container user
- [ ] Saliency / Grad-CAM overlays in the UI to make predictions explainable
- [ ] Add a `LICENSE` file

---

## Acknowledgments

- [EfficientNet](https://arxiv.org/abs/1905.11946) — Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs"; weights via Keras Applications
- [VGG16](https://arxiv.org/abs/1409.1556) — Simonyan & Zisserman (supported alternative backbone)
- [MLflow](https://mlflow.org/) · [DagsHub](https://dagshub.com/) · [DVC](https://dvc.org/)
- [TensorFlow / Keras](https://www.tensorflow.org/) · [Flask](https://flask.palletsprojects.com/)

---

## License

No license file has been added to this repository yet. All rights are reserved by the author by
default — add an open-source `LICENSE` (e.g. MIT) if you intend to distribute the project.
