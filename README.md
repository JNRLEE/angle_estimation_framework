# 角度估測框架 (Angle Estimation Framework)

基於深度學習的聲音訊號角度估測框架。

## 專案設置

本專案使用 PyTorch 進行深度學習模型訓練和評估。

### 先決條件

- Python 3.12+
- Git

### 從原始碼啟動

1. 克隆此儲存庫:
```
git clone https://github.com/JNRLEE/angle_estimation_framework.git
```

2. 安裝依賴:
```
pip install -r requirements.txt
```

3. 執行訓練:
```
# 分類模型
python -m angle_estimation_framework.scripts.train --config angle_estimation_framework/configs/swin_classification_18deg.yaml --seed 42

# 回歸模型
python -m angle_estimation_framework.scripts.train --config angle_estimation_framework/configs/swin_regression_18deg.yaml --seed 42
```

### 使用 Docker

1. 構建 Docker 映像:
```
docker build -t angle-estimation .
```

2. 運行容器:
```
docker run -it angle-estimation
```

## 依賴

本專案需要:

- Python 3.12+
- PyTorch 2.0+
- torchvision
- numpy
- pillow
- librosa
- PyYAML
- transformers (用於 SwinTransformerBackbone)

詳細依賴列表請參見 `requirements.txt`。

# 角度估測框架架構

本框架為基於聲音頻譜圖的角度估測任務提供了模組化結構，支援分類和回歸兩種模式。

## 結構

- `configs/`: 實驗配置文件
  - `swin_classification_18deg.yaml`: 分類模型配置範例
  - `swin_regression_18deg.yaml`: 回歸模型配置範例
- `data/`: 資料載入與預處理模組
- `losses/`: 自定義損失函數
- `models/`: 模型定義，分為骨幹網路和頭部
- `optimizers/`: 優化器與學習率調度器配置
- `scripts/`: 主要訓練與評估腳本
- `trainers/`: 訓練迴圈邏輯
- `utils/`: 工具函數
- `tests/`: 測試腳本

    angle_estimation_framework/
    ├── configs/
    │   ├── swin_classification_18deg.yaml # 分類模型配置
    │   └── swin_regression_18deg.yaml     # 回歸模型配置
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
    │   │   ├── classification_head.py # ClassificationHead
    │   │   └── regression_head.py     # RegressionHead
    │   └── meta_model.py       # AngleEstimationModel (combines backbone+head)
    ├── optimizers/
    │   └── __init__.py         # Optimizer/Scheduler registry
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
    └── tests/                  # Test scripts
    
## 使用方法

### 資料格式

預期的資料組織方式為分層結構:
```
step_018_sliced/
├── deg000/
│   ├── plastic/
│   │   ├── plastic_deg000_500hz_00.wav
│   │   ├── ...
│   └── box/
│       ├── box_deg000_500hz_00.wav
│       ├── ...
├── deg018/
└── ...
```

檔案名稱格式應包含角度、材質、頻率和索引等信息。

### 訓練模型

分類模型訓練範例:
```bash
python -m angle_estimation_framework.scripts.train --config angle_estimation_framework/configs/swin_classification_18deg.yaml --seed 42
```

回歸模型訓練範例:
```bash
python -m angle_estimation_framework.scripts.train --config angle_estimation_framework/configs/swin_regression_18deg.yaml --seed 42
```

訓練結果將保存在 `training_runs` 目錄下，包含模型檢查點、配置和訓練歷史。

### 配置文件

配置文件是 YAML 格式，包括數據、模型、損失函數、優化器和訓練參數等設定。
可根據需求自定義這些參數。