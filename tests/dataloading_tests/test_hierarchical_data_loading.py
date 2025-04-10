#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
這個程式檔案用來測試新的層級式資料結構的資料讀取邏輯是否正確，確保可以從新的資料夾結構中正確讀取音訊檔案及其元數據。
"""

import os
import sys
import argparse
from datetime import datetime

# 添加專案根目錄到 Python 路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 直接從 dataset_utils 模組導入需要的函數
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from angle_estimation_framework.data.dataset_utils import (
    get_files_from_hierarchical_structure,
    extract_angle_from_path,
    extract_material_from_path,
    extract_metadata_from_filename,
    extract_angle_from_filename,
    filter_files_by_metadata
)

def test_file_listing(data_dir):
    """
    測試從新的資料夾結構中獲取檔案列表的功能。
    
    Args:
        data_dir (str): 資料目錄的路徑
        
    Returns:
        list: 找到的檔案列表
    """
    print(f"測試從 {data_dir} 獲取檔案列表...")
    
    # 獲取所有 wav 檔案
    all_files = get_files_from_hierarchical_structure(data_dir, format_filter=['wav'])
    
    print(f"總共找到 {len(all_files)} 個 .wav 檔案")
    
    # 列出前 5 個檔案
    if all_files:
        print("前 5 個檔案範例:")
        for i, file in enumerate(all_files[:5]):
            print(f"  {i+1}. {file}")
            
    return all_files

def test_metadata_extraction(files):
    """
    測試從檔案路徑中提取元數據的功能。
    
    Args:
        files (list): 檔案路徑列表
        
    Returns:
        dict: 統計結果
    """
    print("\n測試元數據提取...")
    
    stats = {
        'angle_extraction_success': 0,
        'material_extraction_success': 0,
        'frequency_extraction_success': 0,
        'total_files': len(files)
    }
    
    angle_counts = {}
    material_counts = {}
    frequency_counts = {}
    
    # 測試前 20 個檔案，並打印其元數據
    test_files = files[:20] if len(files) > 20 else files
    
    for i, file in enumerate(test_files):
        metadata = extract_metadata_from_filename(file)
        
        print(f"\n檔案 {i+1}: {os.path.basename(file)}")
        for key, value in metadata.items():
            if key not in ['filename', 'full_path']:
                print(f"  - {key}: {value}")
        
        # 統計成功提取的欄位數量
        if 'angle' in metadata:
            stats['angle_extraction_success'] += 1
            angle = metadata['angle']
            angle_counts[angle] = angle_counts.get(angle, 0) + 1
            
        if 'material' in metadata:
            stats['material_extraction_success'] += 1
            material = metadata['material']
            material_counts[material] = material_counts.get(material, 0) + 1
            
        if 'frequency' in metadata:
            stats['frequency_extraction_success'] += 1
            frequency = metadata['frequency']
            frequency_counts[frequency] = frequency_counts.get(frequency, 0) + 1
    
    # 計算統計百分比
    total = stats['total_files']
    stats['angle_extraction_rate'] = (stats['angle_extraction_success'] / len(test_files)) * 100 if len(test_files) > 0 else 0
    stats['material_extraction_rate'] = (stats['material_extraction_success'] / len(test_files)) * 100 if len(test_files) > 0 else 0
    stats['frequency_extraction_rate'] = (stats['frequency_extraction_success'] / len(test_files)) * 100 if len(test_files) > 0 else 0
    
    print("\n元數據提取統計:")
    print(f"  - 樣本數量: {len(test_files)}")
    print(f"  - 角度提取成功率: {stats['angle_extraction_rate']:.1f}%")
    print(f"  - 材質提取成功率: {stats['material_extraction_rate']:.1f}%")
    print(f"  - 頻率提取成功率: {stats['frequency_extraction_rate']:.1f}%")
    
    print("\n角度分布:")
    for angle, count in sorted(angle_counts.items()):
        print(f"  - {angle}度: {count} 個檔案")
        
    print("\n材質分布:")
    for material, count in material_counts.items():
        print(f"  - {material}: {count} 個檔案")
        
    print("\n頻率分布:")
    for frequency, count in sorted(frequency_counts.items()):
        print(f"  - {frequency}Hz: {count} 個檔案")
    
    return stats

def test_file_filtering(files, filter_criteria):
    """
    測試檔案過濾功能。
    
    Args:
        files (list): 檔案路徑列表
        filter_criteria (dict): 過濾條件
        
    Returns:
        tuple: (過濾後的檔案列表, 對應的標籤列表, 統計資訊)
    """
    print("\n測試檔案過濾...")
    print(f"過濾條件: {filter_criteria}")
    
    filtered_files, filtered_labels, stats = filter_files_by_metadata(files, filter_criteria)
    
    print(f"過濾結果:")
    print(f"  - 總檔案數: {stats['total']}")
    print(f"  - 接受的檔案數: {stats['accepted']}")
    print(f"  - 拒絕的檔案數: {sum(stats['rejected'].values())}")
    
    # 顯示拒絕的原因
    for reason, count in stats['rejected'].items():
        if count > 0:
            print(f"    - 因 {reason} 拒絕: {count} 個檔案")
    
    # 顯示前 5 個過濾後的檔案
    if filtered_files:
        print("\n過濾後的前 5 個檔案範例:")
        for i, (file, label) in enumerate(zip(filtered_files[:5], filtered_labels[:5])):
            print(f"  {i+1}. {os.path.basename(file)} (標籤: {label})")
    
    return filtered_files, filtered_labels, stats

def test_angle_extraction_methods(files):
    """
    分別測試並比較從路徑和檔名中提取角度的方法。
    
    Args:
        files (list): 檔案路徑列表
        
    Returns:
        dict: 比較結果統計
    """
    print("\n測試角度提取方法比較...")
    
    stats = {
        'both_methods_match': 0,
        'path_only_success': 0,
        'filename_only_success': 0,
        'both_failed': 0,
        'total': len(files)
    }
    
    # 測試前 20 個檔案
    test_files = files[:20] if len(files) > 20 else files
    
    for file in test_files:
        path_angle = extract_angle_from_path(file)
        filename_angle = extract_angle_from_filename(file)
        
        if path_angle is not None and filename_angle is not None:
            if path_angle == filename_angle:
                stats['both_methods_match'] += 1
            else:
                print(f"警告: 檔案路徑與檔名中的角度不一致 - {file}")
                print(f"  - 路徑中的角度: {path_angle}")
                print(f"  - 檔名中的角度: {filename_angle}")
        elif path_angle is not None:
            stats['path_only_success'] += 1
        elif filename_angle is not None:
            stats['filename_only_success'] += 1
        else:
            stats['both_failed'] += 1
    
    # 計算統計百分比
    total = len(test_files)
    stats['both_methods_match_rate'] = (stats['both_methods_match'] / total) * 100 if total > 0 else 0
    stats['path_only_success_rate'] = (stats['path_only_success'] / total) * 100 if total > 0 else 0
    stats['filename_only_success_rate'] = (stats['filename_only_success'] / total) * 100 if total > 0 else 0
    stats['both_failed_rate'] = (stats['both_failed'] / total) * 100 if total > 0 else 0
    
    print("\n角度提取方法比較結果:")
    print(f"  - 樣本數量: {total}")
    print(f"  - 兩種方法結果一致: {stats['both_methods_match_rate']:.1f}%")
    print(f"  - 僅路徑中提取成功: {stats['path_only_success_rate']:.1f}%")
    print(f"  - 僅檔名中提取成功: {stats['filename_only_success_rate']:.1f}%")
    print(f"  - 兩種方法均失敗: {stats['both_failed_rate']:.1f}%")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="測試層級式資料結構的資料讀取邏輯")
    parser.add_argument("--data_dir", type=str, default="/Users/sbplab/jnrle/sinetone_link/step_018_sliced",
                        help="資料目錄的路徑")
    args = parser.parse_args()
    
    print(f"=== 資料讀取邏輯測試 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===\n")
    
    # 測試 1: 檔案列表
    files = test_file_listing(args.data_dir)
    
    if not files:
        print("錯誤: 找不到檔案，測試終止")
        sys.exit(1)
    
    # 測試 2: 元數據提取
    metadata_stats = test_metadata_extraction(files)
    
    # 測試 3: 角度提取方法比較
    angle_extraction_stats = test_angle_extraction_methods(files)
    
    # 測試 4: 檔案過濾 (塑膠材質)
    filter_criteria_plastic = {
        'material_filter': 'plastic',
        'angle_values': [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
    }
    
    plastic_files, plastic_labels, plastic_stats = test_file_filtering(files, filter_criteria_plastic)
    
    # 測試 5: 檔案過濾 (箱子材質，僅 500Hz)
    filter_criteria_box_500hz = {
        'material_filter': 'box',
        'frequency_filter': [500],
        'angle_values': [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
    }
    
    box_files, box_labels, box_stats = test_file_filtering(files, filter_criteria_box_500hz)
    
    print("\n=== 測試完成 ===")
    
    # 測試摘要
    print("\n測試摘要:")
    print(f"  - 總檔案數: {len(files)}")
    print(f"  - 元數據提取成功率 (角度): {metadata_stats['angle_extraction_rate']:.1f}%")
    print(f"  - 元數據提取成功率 (材質): {metadata_stats['material_extraction_rate']:.1f}%")
    print(f"  - 元數據提取成功率 (頻率): {metadata_stats['frequency_extraction_rate']:.1f}%")
    print(f"  - 兩種角度提取方法一致率: {angle_extraction_stats['both_methods_match_rate']:.1f}%")
    print(f"  - 塑膠材質檔案過濾接受率: {(plastic_stats['accepted'] / len(files)) * 100:.1f}%")
    print(f"  - 箱子材質 500Hz 檔案過濾接受率: {(box_stats['accepted'] / len(files)) * 100:.1f}%")

if __name__ == "__main__":
    main() 