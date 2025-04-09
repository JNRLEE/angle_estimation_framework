# LDV Reorientation

A framework for angle estimation using deep learning.

## Project Setup

This project uses Docker for environment setup and PyTorch for deep learning.

### Prerequisites

- Docker
- Git

### Running with Docker

1. Build the Docker image:
```
docker build -t ldv-reorientation .
```

2. Run the container:
```
docker run -it ldv-reorientation
```

## Dependencies

This project requires:

- Python 3.9
- PyTorch 2.0+
- torchvision
- numpy
- pillow
- librosa
- PyYAML
- transformers (for using SwinTransformerBackbone)

See `requirements.txt` for the complete list of dependencies.

# Angle Estimation Framework

This framework provides a modular structure for training and evaluating models for angle estimation tasks based on spectrogram data.

## Structure

- `configs/`: Configuration files for experiments.
- `data/`: Data loading and preprocessing modules.
- `losses/`: Custom loss functions.
- `models/`: Model definitions, separated into backbones and heads.
- `optimizers/`: Optimizer and scheduler configurations.
- `scripts/`: Main training and evaluation scripts.
- `trainers/`: Training loop logic.
- `utils/`: Utility functions. 

    angle_estimation_framework/
    ├── configs/
    │   └── swin_regression_18deg.yaml  # Example config
    ├── data/
    │   ├── __init__.py
    │   ├── datasets.py         # AudioSpectrogramDataset
    │   └── ranking.py          # RankingPairDataset
    ├── losses/
    │   ├── __init__.py         # Loss registry (get_loss_function)
    │   └── ranking_losses.py   # OrderedMarginRankingLoss, etc.
    ├── models/
    │   ├── __init__.py         # Exposes AngleEstimationModel, backbones, heads
    │   ├── backbones/
    │   │   ├── __init__.py
    │   │   ├── simple_cnn.py       # SimpleCNNBackbone
    │   │   └── swin_transformer.py # SwinTransformerBackbone
    │   ├── heads/
    │   │   ├── __init__.py
    │   │   └── regression_head.py  # RegressionHead
    │   └── meta_model.py       # AngleEstimationModel (combines backbone+head)
    ├── optimizers/
    │   └── __init__.py         # Optimizer/Scheduler registry (get_optimizer, etc.)
    ├── scripts/
    │   ├── __init__.py
    │   └── train.py            # Main training script
    ├── trainers/
    │   ├── __init__.py
    │   └── base_trainer.py     # BaseTrainer class
    ├── utils/
    │   ├── __init__.py
    │   └── audio_utils.py      # Audio processing functions
    ├── README.md
    ├── requirements.txt        # Python dependencies
    └──tests/                   # Preserve all the test script