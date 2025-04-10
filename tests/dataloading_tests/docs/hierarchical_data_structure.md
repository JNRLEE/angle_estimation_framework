# 層級式資料結構更新說明

## 概述

我們已更新資料讀取邏輯，以處理新的層級式資料夾結構。舊結構將所有音訊檔案存放在單一目錄下，而新結構按角度和材質分類，形成以下層次結構：

```
/sinetone_link/step_018_sliced/
    /deg000/
        /box/
            box_deg000_500hz_00.wav
            box_deg000_1000hz_00.wav
            ...
        /plastic/
            plastic_deg000_500hz_00.wav
            plastic_deg000_1000hz_00.wav
            ...
    /deg018/
        /box/
            ...
        /plastic/
            ...
    ...
    /deg180/
        /box/
            ...
        /plastic/
            ...
```

## 主要更新

1. 新增了 `get_files_from_hierarchical_structure` 函數，遞迴搜尋層級式結構中的檔案
2. 增強了元數據提取功能，可從目錄路徑中提取角度和材質信息：
   - `extract_angle_from_path`
   - `extract_material_from_path`
3. 更新了 `extract_metadata_from_filename` 函數，優先從路徑中提取元數據，再回退到檔名提取
4. 更新了 `AudioSpectrogramDataset` 類別，使用新的檔案搜尋函數

## 優勢

1. **更佳組織** - 資料按角度和材質分類，方便管理和瀏覽
2. **提高可靠性** - 現在可從路徑和檔名中提取元數據，提供雙重保障
3. **靈活性** - 支持新舊文件結構，保持向後兼容性
4. **標準化** - 從路徑提取元數據能確保更一致的標籤提取

## 使用方法

### 配置檔範例

以下是使用新資料結構的配置檔案範例：

```yaml
data:
  data_dir: "/Users/sbplab/jnrle/sinetone_link/step_018_sliced"
  material_filter: "plastic"  # or "all", "plastic", "box"
  frequency_filter: [500]
  index_filter: null
  format_filter: ["wav"]
  angle_values: [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
  # Other data parameters...
```

### 程式碼使用範例

```python
from angle_estimation_framework.data.dataset_utils import get_files_from_hierarchical_structure
from angle_estimation_framework.data.datasets import AudioSpectrogramDataset

# 取得檔案列表
files = get_files_from_hierarchical_structure("/path/to/data", format_filter=['wav'])

# 或直接使用 AudioSpectrogramDataset
config = {
    'audio_params': {...},
    'data_filtering': {
        'material_filter': "plastic",
        'frequency_filter': [500],
        'format_filter': ["wav"],
        'angle_values': [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
    }
}
dataset = AudioSpectrogramDataset("/path/to/data", config)
```

## 測試

我們已建立兩個測試腳本，用於驗證新的資料讀取邏輯：

1. `tests/test_hierarchical_data_loading.py`：測試檔案搜尋和元數據提取功能
2. `tests/test_hierarchical_dataset.py`：測試 `AudioSpectrogramDataset` 能否正確處理新資料結構

## 維護建議

1. 繼續使用標準化的檔名模式：`{material}_deg{angle}_{frequency}hz_{index}.wav`
2. 遵循新的目錄結構：`deg{angle}/{material}/{files}`
3. 如需添加新的檔案，請將其放在對應的角度和材質目錄下
4. 建議使用測試腳本檢查任何對資料處理邏輯的修改

## 注意事項

雖然我們現在從路徑中提取角度和材質，但建議保持檔名中的元數據格式一致，以提供冗餘備份並保持向後兼容性。 