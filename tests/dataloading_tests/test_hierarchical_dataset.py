#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
這個程式檔案用來測試 AudioSpectrogramDataset 類別是否能正確處理新的層級式資料結構。
"""

import os
import sys
import argparse
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# 添加專案根目錄到 Python 路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from angle_estimation_framework.data.datasets import AudioSpectrogramDataset

def plot_spectrogram(spectrogram, title, output_file):
    """
    繪製頻譜圖並儲存。
    
    Args:
        spectrogram (torch.Tensor): 頻譜圖張量
        title (str): 圖表標題
        output_file (str): 輸出檔案路徑
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(spectrogram.numpy(), aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('時間')
    plt.ylabel('頻率')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def test_dataset(data_dir, config, experiment_id, output_dir):
    """
    測試 AudioSpectrogramDataset 類別。
    
    Args:
        data_dir (str): 資料目錄路徑
        config (dict): 設定字典
        experiment_id (str): 實驗 ID
        output_dir (str): 輸出目錄
    """
    print(f"創建資料集對象...")
    dataset = AudioSpectrogramDataset(data_dir, config)
    
    print(f"資料集大小: {len(dataset)}")
    
    if len(dataset) == 0:
        print("錯誤: 資料集為空，測試終止")
        return
    
    # 檢查標籤分布
    if hasattr(dataset, 'labels') and dataset.labels:
        unique_labels = set(dataset.labels)
        label_counts = {}
        for label in unique_labels:
            count = dataset.labels.count(label)
            label_counts[label] = count
        
        print(f"標籤分布:")
        for label, count in sorted(label_counts.items()):
            print(f"  - 標籤 {label}: {count} 個樣本")
    
    # 取樣資料並視覺化
    print(f"取樣資料集中的樣本...")
    
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 從每個角度取樣一個樣本進行視覺化
    if hasattr(dataset, 'angle_values') and dataset.angle_values:
        for i, angle in enumerate(dataset.angle_values):
            # 尋找對應角度的樣本索引
            indices = [j for j, label in enumerate(dataset.labels) if label == i]
            if indices:
                idx = indices[0]  # 取第一個找到的樣本
                try:
                    # 獲取樣本
                    spectrogram, label = dataset[idx]
                    
                    # 提取文件名和材質以用於標題
                    file_path = dataset.file_paths[idx]
                    file_name = os.path.basename(file_path)
                    material = "未知"
                    if "/box/" in file_path:
                        material = "箱子"
                    elif "/plastic/" in file_path:
                        material = "塑膠"
                    
                    # 繪製並保存頻譜圖
                    title = f"{material} {angle}度 頻譜圖 (標籤: {label.item():.0f})"
                    output_file = os.path.join(output_dir, f"{experiment_id}_angle{angle}_sample.png")
                    plot_spectrogram(spectrogram.squeeze(), title, output_file)
                    print(f"  - 角度 {angle}度: 已儲存頻譜圖至 {output_file}")
                except Exception as e:
                    print(f"  - 角度 {angle}度: 處理失敗 - {str(e)}")
            else:
                print(f"  - 角度 {angle}度: 無樣本")
    else:
        # 如果沒有特定角度值，則取樣前 5 個樣本
        for i in range(min(5, len(dataset))):
            try:
                spectrogram, label = dataset[i]
                
                # 提取文件名以用於標題
                file_path = dataset.file_paths[i]
                file_name = os.path.basename(file_path)
                
                # 繪製並保存頻譜圖
                title = f"樣本 {i+1}: {file_name} (標籤: {label.item():.0f})"
                output_file = os.path.join(output_dir, f"{experiment_id}_sample{i+1}.png")
                plot_spectrogram(spectrogram.squeeze(), title, output_file)
                print(f"  - 樣本 {i+1}: 已儲存頻譜圖至 {output_file}")
            except Exception as e:
                print(f"  - 樣本 {i+1}: 處理失敗 - {str(e)}")
    
    # 測試資料加載速度
    print(f"測試資料加載速度...")
    
    # 創建一個小型 DataLoader 測試批處理加載
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=0
    )
    
    # 迭代幾個批次以測試加載速度
    start_time = datetime.now()
    num_batches = min(5, len(data_loader))
    
    for i, (spectrograms, labels) in enumerate(data_loader):
        if i >= num_batches:
            break
        
        print(f"  - 批次 {i+1}: 形狀 {spectrograms.shape}, 標籤形狀 {labels.shape}")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"  - 加載 {num_batches} 個批次耗時: {elapsed:.2f} 秒")

def main():
    parser = argparse.ArgumentParser(description="測試階層式資料結構的 AudioSpectrogramDataset")
    parser.add_argument("--data_dir", type=str, default="/Users/sbplab/jnrle/sinetone_link/step_018_sliced",
                        help="資料目錄的路徑")
    parser.add_argument("--output_dir", type=str, default="tests/output",
                        help="輸出目錄的路徑")
    args = parser.parse_args()
    
    # 實驗 ID 使用時間戳
    experiment_id = f"dataset_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"=== AudioSpectrogramDataset 測試 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===\n")
    
    # 定義用於測試的配置
    config = {
        'audio_params': {
            'sample_rate': 16000,
            'n_fft': 1024,
            'hop_length': 256,
            'n_mels': 128,
            'target_length': 128
        },
        'data_filtering': {
            'material_filter': "plastic",  # 測試只使用塑膠材質
            'frequency_filter': [500],     # 測試只使用 500Hz
            'index_filter': None,
            'format_filter': ["wav"],
            'angle_values': [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
        }
    }
    
    # 測試資料集
    test_dataset(args.data_dir, config, experiment_id, args.output_dir)
    
    print("\n=== 測試完成 ===")

if __name__ == "__main__":
    main() 