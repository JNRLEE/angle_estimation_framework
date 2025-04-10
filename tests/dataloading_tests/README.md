# 音訊角度估測系統測試目錄

這個目錄包含用於測試 Angle Estimation Framework 各個組件的測試腳本。

## 測試檔案說明

- `test_hierarchical_data_loading.py` - 測試新的層級式資料結構的資料讀取邏輯
- `test_hierarchical_dataset.py` - 測試 AudioSpectrogramDataset 類別處理新資料結構的能力

## 如何執行測試

### 資料結構與讀取邏輯測試

```bash
# 預設使用 /Users/sbplab/jnrle/sinetone_link/step_018_sliced 資料目錄
python tests/test_hierarchical_data_loading.py

# 指定資料目錄
python tests/test_hierarchical_data_loading.py --data_dir "/path/to/data"
```

### 資料集類別測試

```bash
# 預設使用 /Users/sbplab/jnrle/sinetone_link/step_018_sliced 資料目錄
python tests/test_hierarchical_dataset.py

# 指定資料目錄和輸出目錄
python tests/test_hierarchical_dataset.py --data_dir "/path/to/data" --output_dir "tests/output"
```

## 輸出結果

測試腳本會輸出以下結果：

1. 測試結果和統計資訊會顯示在終端上
2. `test_hierarchical_dataset.py` 會在 `tests/output` 目錄下產生頻譜圖，用於視覺化檢查資料集處理效果

## 自動測試腳本

這些測試腳本包含完整的 docstring，記錄了測試目的，並可以直接運行進行自動化測試。測試結果會顯示在終端，並且會將視覺化結果保存在輸出目錄中。 