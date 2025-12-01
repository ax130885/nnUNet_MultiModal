# nnunetv2/inference/nnunet_predictor_multimodal.py

import inspect
import itertools
import multiprocessing
import os
from copy import deepcopy
from queue import Queue
from threading import Thread
from time import sleep
from typing import Tuple, Union, List, Optional

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import nnunetv2
from nnunetv2.configuration import default_num_processes

# 如果創建了新文件，則導入路徑需要調整
from nnunetv2.inference.data_iterators_multimodal import preprocessing_iterator_fromfiles_multimodal
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels # 確保導入


from nnunetv2.inference.export_prediction import export_prediction_from_logits, \
    convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, \
    compute_steps_for_sliding_window
from nnunetv2.utilities.file_path_utilities import get_output_folder, check_workers_alive_and_busy
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.label_handling.label_handling import LabelManager

# 導入原始的 nnUNetPredictor
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


from nnunetv2.training.nnUNetTrainer.multitask_segmentation_model import MyModel
from nnunetv2.training.nnUNetTrainer.multitask_model import MyMultiModel

import csv
from nnunetv2.preprocessing.clinical_data_label_encoder import ClinicalDataLabelEncoder
import time
import json

class nnUNetPredictorMultimodal(nnUNetPredictor):
    """
    擴展 nnUNetPredictor，使其能夠處理多模態數據 (影像 + 臨床資料) 的預測。
    主要修改是調整預測函數，以接受並將臨床特徵傳遞給底層模型。
    """
    def __init__(self,
                 tile_step_size: float = 0.5, # 滑窗預測時的重疊步長比例
                 use_gaussian: bool = True,   # 是否使用高斯權重對滑窗結果加權
                 use_mirroring: bool = True,  # 是否使用測試時數據增強（鏡像）
                 perform_everything_on_device: bool = True, # 是否全程在 GPU 執行
                 device: torch.device = torch.device('cuda'), # 指定運算設備
                 verbose: bool = True,       # 是否輸出詳細日誌
                 verbose_preprocessing: bool = False, # 是否輸出預處理詳細日誌
                 allow_tqdm: bool = True,     # 是否允許顯示進度條
                 clinical_data_dir: str = None,
                 no_use_input_cli_data: bool = False): # 推論時 不輸入臨床資料):
        super().__init__(tile_step_size, use_gaussian, use_mirroring,
                         perform_everything_on_device, device, verbose,
                         verbose_preprocessing, allow_tqdm)
        self.clinical_data_label_encoder = None
        self.reverse_mappings = None
        self.missing_flags = None
        self.clinical_data_dir = clinical_data_dir  # 新增臨床數據目錄參數
        self.no_use_input_cli_data = no_use_input_cli_data  # 推論時不輸入臨床資料


        print("nnUNetPredictorMultimodal 初始化完成。")

        self.clinical_data_label_encoder = None
        self.reverse_location_mapping = None
        self.reverse_t_stage_mapping = None
        self.reverse_n_stage_mapping = None
        self.reverse_m_stage_mapping = None
        self.reverse_dataset_mapping = None

    # 初始化臨床資料編碼器 取得反向 mapping
    def initialize_clinical_encoder(self):
        """初始化臨床數據編碼器並設置反向映射"""
        debug_log_path = "/mnt/data1/graduate/yuxin/Lab/model/UNet_base/nnunet_ins_data/nnunet_debug.txt"
        
        # 寫入debug信息到文件
        def write_debug_log(message):
            try:
                with open(debug_log_path, "a") as f:
                    f.write(f"{message}\n")
            except Exception as e:
                print(f"無法寫入debug日誌: {e}")
                
        if self.clinical_data_dir:
            try:
                import pandas as pd
                import traceback
                
                write_debug_log(f"[PREDICTOR] 正在初始化臨床數據編碼器，目錄: {self.clinical_data_dir}")
                
                self.clinical_data_label_encoder = ClinicalDataLabelEncoder(self.clinical_data_dir)
                self.num_clinical_classes = {
                    'location': self.clinical_data_label_encoder.num_location_classes,
                    't_stage': self.clinical_data_label_encoder.num_t_stage_classes,
                    'n_stage': self.clinical_data_label_encoder.num_n_stage_classes,
                    'm_stage': self.clinical_data_label_encoder.num_m_stage_classes,
                    'dataset': self.clinical_data_label_encoder.num_dataset_classes
                }
                
                # 記錄缺失標記的索引
                self.missing_flags = {
                    'location': self.clinical_data_label_encoder.missing_flag_location,
                    't_stage': self.clinical_data_label_encoder.missing_flag_t_stage,
                    'n_stage': self.clinical_data_label_encoder.missing_flag_n_stage,
                    'm_stage': self.clinical_data_label_encoder.missing_flag_m_stage,
                    'dataset': self.clinical_data_label_encoder.missing_flag_dataset
                }
                
                # 保存反向映射以供檢視結果使用
                self.reverse_location_mapping = self.clinical_data_label_encoder.reverse_location_mapping
                self.reverse_t_stage_mapping = self.clinical_data_label_encoder.reverse_t_stage_mapping
                self.reverse_n_stage_mapping = self.clinical_data_label_encoder.reverse_n_stage_mapping
                self.reverse_m_stage_mapping = self.clinical_data_label_encoder.reverse_m_stage_mapping
                self.reverse_dataset_mapping = self.clinical_data_label_encoder.reverse_dataset_mapping
                
                # 從編碼器獲取反向映射
                self.reverse_mappings = {
                    'location': self.clinical_data_label_encoder.reverse_location_mapping,
                    't_stage': self.clinical_data_label_encoder.reverse_t_stage_mapping,
                    'n_stage': self.clinical_data_label_encoder.reverse_n_stage_mapping,
                    'm_stage': self.clinical_data_label_encoder.reverse_m_stage_mapping,
                    'dataset': self.clinical_data_label_encoder.reverse_dataset_mapping
                }
                
                # 檢查是否成功載入臨床數據
                if hasattr(self.clinical_data_label_encoder, 'clinical_encoded_df'):
                    df = self.clinical_data_label_encoder.clinical_encoded_df
                    write_debug_log(f"[PREDICTOR] 成功載入臨床數據 CSV，共 {len(df)} 行")
                    write_debug_log(f"[PREDICTOR] 臨床數據表頭: {df.columns.tolist()}")
                    write_debug_log(f"[PREDICTOR] 臨床數據案例ID範例: {df['Case_Index'].tolist()[:5]}")
                    
                    print(f"已載入臨床數據CSV，形狀: {df.shape}，列: {df.columns.tolist()}")
                    print(f"包含 {len(df['Case_Index'].unique())} 個唯一案例ID")
                else:
                    write_debug_log("[PREDICTOR警告] clinical_data_label_encoder 沒有 clinical_encoded_df 屬性")
                
                write_debug_log("[PREDICTOR] 臨床數據編碼器初始化成功")
                print(f"臨床數據編碼器初始化成功，缺失標記: {self.missing_flags}")
                
            except Exception as e:
                print(f"[警告] 無法初始化 ClinicalDataLabelEncoder: {e}")
                write_debug_log(f"[PREDICTOR錯誤] 初始化臨床數據編碼器時發生錯誤: {e}")
                write_debug_log(traceback.format_exc())
                traceback.print_exc()
                self.clinical_data_label_encoder = None
                self.reverse_mappings = None
                self.missing_flags = None

    def _generate_text_description(self, clinical_data_dict, clinical_mask_dict):
        """
        根據可用的臨床特徵生成文字描述
        
        Args:
            clinical_data_dict: 包含各特徵值的字典
            clinical_mask_dict: 包含各特徵是否有效的字典
        
        Returns:
            str: 生成的文字描述
        """
        if not hasattr(self, 'clinical_data_label_encoder') or self.clinical_data_label_encoder is None:
            return "A computerized tomography scan reveals a colorectal cancer."
        
        # 獲取反向映射表中的類別名稱
        location_mapping = self.clinical_data_label_encoder.reverse_location_mapping
        t_stage_mapping = self.clinical_data_label_encoder.reverse_t_stage_mapping
        n_stage_mapping = self.clinical_data_label_encoder.reverse_n_stage_mapping
        m_stage_mapping = self.clinical_data_label_encoder.reverse_m_stage_mapping
        dataset_mapping = self.clinical_data_label_encoder.reverse_dataset_mapping
        
        # 基礎描述
        base_text = "A computerized tomography scan reveals a colorectal cancer"
        
        # 收集有效的特徵描述
        feature_descriptions = []
        
        # Location 描述
        if clinical_mask_dict['location']:
            loc_idx = clinical_data_dict['location']
            location_name = location_mapping[loc_idx]
            if location_name != 'Missing':
                feature_descriptions.append(f"located in the {location_name} region")
        
        # T Stage 描述
        if clinical_mask_dict['t_stage']:
            t_idx = clinical_data_dict['t_stage']
            t_stage_name = t_stage_mapping[t_idx]
            if t_stage_name != 'Missing':
                feature_descriptions.append(f"with T stage {t_stage_name}")
        
        # N Stage 描述
        if clinical_mask_dict['n_stage']:
            n_idx = clinical_data_dict['n_stage']
            n_stage_name = n_stage_mapping[n_idx]
            if n_stage_name != 'Missing':
                feature_descriptions.append(f"N stage {n_stage_name}")
        
        # M Stage 描述
        if clinical_mask_dict['m_stage']:
            m_idx = clinical_data_dict['m_stage']
            m_stage_name = m_stage_mapping[m_idx]
            if m_stage_name != 'Missing':
                metastasis_desc = "with distant metastasis" if m_stage_name == "M1" else "without distant metastasis"
                feature_descriptions.append(metastasis_desc)

        # Dataset 描述
        if clinical_mask_dict['dataset']:
            dataset_idx = clinical_data_dict['dataset']
            dataset_name = self.clinical_data_label_encoder.reverse_dataset_mapping[dataset_idx]
            if dataset_name != 'Missing':
                feature_descriptions.append(f"from the {dataset_name} dataset")
        
        # 組合描述
        if feature_descriptions:
            full_text = base_text + " " + ", ".join(feature_descriptions) + "."
        else:
            full_text = base_text + "."
        
        return full_text

    # 推論主函式
    @torch.inference_mode()
    def predict_from_data_iterator(self,
                                data_iterator,
                                save_probabilities: bool = False, # 是否儲存預測的機率圖。
                                num_processes_segmentation_export: int = default_num_processes, # 用於分割結果導出的進程數。
                                profiler=None): # 是否啟用於性能分析(功能未完成 別用)
        """
        對 iterator 提供的資料 進行推論
        迭代器返回的內容為
        item = {
                'data': data,  # 影像數據
                'target': seg,  # 分割標籤
                'clinical_data': clinical_data,  # 臨床數據字典
                'clinical_mask': clinical_mask,  # 臨床掩碼字典
                'properties': data_properties,  # 數據屬性
                'ofile': output_filenames_truncated[idx] if output_filenames_truncated is not None else None  # 輸出文件路徑
            }
        """
        # 初始化臨床數據編碼器
        if self.clinical_data_dir:
            self.initialize_clinical_encoder()

        # 將臨床資料的預測結果 進行 argmax 接著轉換為實際標籤
        def clinical_logits_to_labels(clinical_attr_preds):
            # 使用反向映射將索引轉回標籤
            labels = {}
            
            # 調試: 檢查關鍵變量是否存在
            if not hasattr(self, 'reverse_mappings') or self.reverse_mappings is None:
                if self.verbose:
                    print("警告: reverse_mappings 未初始化，使用預設值")
            if not hasattr(self, 'missing_flags') or self.missing_flags is None:
                if self.verbose:
                    print("警告: missing_flags 未初始化，使用預設值")
            
            if self.reverse_mappings and clinical_attr_preds:
                try:
                    # 獲取最大概率的類別索引
                    loc_idx = int(torch.argmax(clinical_attr_preds['location']).item())
                    t_idx = int(torch.argmax(clinical_attr_preds['t_stage']).item())
                    n_idx = int(torch.argmax(clinical_attr_preds['n_stage']).item())
                    m_idx = int(torch.argmax(clinical_attr_preds['m_stage']).item())
                    dataset_idx = int(torch.argmax(clinical_attr_preds['dataset']).item())
                    
                    # 計算各特徵的分布概率，用於多樣性檢查
                    loc_probs = torch.softmax(clinical_attr_preds['location'], dim=0).cpu().numpy()
                    t_probs = torch.softmax(clinical_attr_preds['t_stage'], dim=0).cpu().numpy()
                    n_probs = torch.softmax(clinical_attr_preds['n_stage'], dim=0).cpu().numpy()
                    m_probs = torch.softmax(clinical_attr_preds['m_stage'], dim=0).cpu().numpy()
                    dataset_probs = torch.softmax(clinical_attr_preds['dataset'], dim=0).cpu().numpy()
                    
                    # 檢查預測的多樣性 - 如果最高概率低於閾值，可能表示模型不確定
                    loc_uncertain = np.max(loc_probs) < 0.7
                    t_uncertain = np.max(t_probs) < 0.7
                    n_uncertain = np.max(n_probs) < 0.7
                    m_uncertain = np.max(m_probs) < 0.7
                    dataset_uncertain = np.max(dataset_probs) < 0.7
                    
                    # 記錄原始索引和缺失值標記
                    loc_missing_idx = self.missing_flags.get('location')
                    t_missing_idx = self.missing_flags.get('t_stage')
                    n_missing_idx = self.missing_flags.get('n_stage')
                    m_missing_idx = self.missing_flags.get('m_stage')
                    dataset_missing_idx = self.missing_flags.get('dataset')
                    
                    # 如果 missing_flag 為 True 或特徵索引等於缺失值索引，則標記為 "Missing"
                    loc_label = "Missing" if (loc_idx == loc_missing_idx) else self.reverse_mappings['location'].get(loc_idx, 'Unknown')
                    t_label = "Missing" if (t_idx == t_missing_idx) else self.reverse_mappings['t_stage'].get(t_idx, 'Unknown')
                    n_label = "Missing" if (n_idx == n_missing_idx) else self.reverse_mappings['n_stage'].get(n_idx, 'Unknown')
                    m_label = "Missing" if (m_idx == m_missing_idx) else self.reverse_mappings['m_stage'].get(m_idx, 'Unknown')
                    dataset_label = "Missing" if (dataset_idx == dataset_missing_idx) else self.reverse_mappings['dataset'].get(dataset_idx, 'Unknown')
                    
                    # 組織最終結果
                    labels = {
                        "Case_Index": "",  # 會在外部設置
                        "Location": loc_label,
                        "T_stage": t_label,
                        "N_stage": n_label,
                        "M_stage": m_label,
                        "Dataset": dataset_label,
                        # 添加額外診斷資訊以便分析
                        "Debug_Info": {
                            "loc_probs": loc_probs.tolist(),
                            "t_probs": t_probs.tolist(),
                            "n_probs": n_probs.tolist(),
                            "m_probs": m_probs.tolist(),
                            "dataset_probs": dataset_probs.tolist(),
                            "loc_uncertain": loc_uncertain,
                            "t_uncertain": t_uncertain,
                            "n_uncertain": n_uncertain,
                            "m_uncertain": m_uncertain,
                            "dataset_uncertain": dataset_uncertain,
                        }
                    }
                    
                except Exception as e:
                    print(f"反向映射標籤時出錯: {e}")
                    # 預設值
                    labels = {
                        "Case_Index": "",
                        "Location": "Unknown",
                        "T_stage": "Unknown",
                        "N_stage": "Unknown",
                        "M_stage": "Unknown",
                        "Dataset": "Unknown",
                        "Missing_flags": [False, False, False, False, False]
                    }
            else:
                # 預設值
                labels = {
                    "Case_Index": "",
                    "Location": "Unknown",
                    "T_stage": "Unknown",
                    "N_stage": "Unknown",
                    "M_stage": "Unknown",
                    "Dataset": "Unknown",
                    "Missing_flags": [False, False, False, False, False]
                }
                
            return labels


        # 保存臨床預測結果 txt
        def save_clinical_prediction_txt(ofile, clinical_attr_preds):
            txt_path = ofile + ".txt"
            labels = clinical_logits_to_labels(clinical_attr_preds)
            with open(txt_path, "w") as f:
                for k, v in labels.items():
                    if k != "Case_Index":  # 跳過案例ID，因為它是從文件名中生成的
                        f.write(f"{k}: {v}\n")

        # 將 numpy bool 轉換為原生 bool，以便 JSON 序列化
        def convert_numpy_bool(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_bool(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_bool(v) for v in obj]
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return obj

        # 保存臨床預測結果 csv
        def save_clinical_prediction_csv(csv_path, all_clinical_results):
            if not all_clinical_results:
                return
                
            # 分析臨床預測的多樣性
            class_counts = {
                'Location': {},
                'T_stage': {},
                'N_stage': {},
                'M_stage': {},
                'Dataset': {}
            }
            
            # 計算每個類別的出現次數
            for result in all_clinical_results:
                for key in class_counts.keys():
                    if key in result:
                        value = result[key]
                        if value not in class_counts[key]:
                            class_counts[key][value] = 0
                        class_counts[key][value] += 1
            
            # 打印類別分佈
            print("\n=== 臨床預測類別分佈 ===")
            for key, counts in class_counts.items():
                print(f"{key}: {counts}")
            
            # 如果某個類別的預測結果過於單一，發出警告
            for key, counts in class_counts.items():
                if len(counts) == 1:
                    print(f"警告: {key} 的預測結果全部相同 ({list(counts.keys())[0]})")
                elif len(counts) <= 2 and len(all_clinical_results) > 10:
                    print(f"警告: {key} 的預測結果多樣性較低，僅有 {len(counts)} 個不同值")
            
            # 移除調試信息以便CSV保存
            cleaned_results = []
            for result in all_clinical_results:
                cleaned_result = {k: v for k, v in result.items() if k != 'Debug_Info'}
                cleaned_results.append(cleaned_result)
                
            # 保存CSV
            fieldnames = list(cleaned_results[0].keys())
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in cleaned_results:
                    writer.writerow(row)
                    

            # 另外保存一個調試信息版本（如果存在）
            if 'Debug_Info' in all_clinical_results[0]:
                debug_csv_path = csv_path.replace('.csv', '_debug.json')
                cleaned_debug_results = [convert_numpy_bool(r) for r in all_clinical_results]
                with open(debug_csv_path, 'w') as f:
                    json.dump(cleaned_debug_results, f, indent=4)
                print(f"診斷資訊已保存到: {debug_csv_path}")

        # 從這邊開始 才比較類似原生的程式
        # 使用 'spawn' 上下文創建進程池，以避免 CUDA 相關問題
        all_clinical_results = []

        # # !!!!!!!!!!!!   打開以後 會等所有iterator處理完 GPU才開始推論  超慢 !!!!!!!!!!!!
        # # DEBUG: 檢查 iterator 內容與長度
        # data_list = list(data_iterator)
        # print(f"[DEBUG] iterator 資料總數: {len(data_list)}")
        # for idx, item in enumerate(data_list):
        #     case_id = item.get('ofile', None) or (item.get('properties', {}).get('case_identifier', None) if 'properties' in item else None)
        #     print(f"[DEBUG] iterator 第{idx+1}筆: ofile={case_id}")
        # # 重新用 list 迭代
        # data_iterator = iter(data_list)

        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool] # 獲取工作進程列表
            r = [] # 用於儲存異步操作的結果物件

            # 遍歷預處理後的數據
            print("開始處理預處理後的數據...")
            for i, preprocessed_item in enumerate(data_iterator):
                # --- 性能分析：在每次處理一個案例前，調用 prof.step() ---
                # 注意：profiler.step() 的調用會根據 schedule 來決定是否記錄。
                if profiler is not None:
                    profiler.step()
                # --- 性能分析結束 ---

                # 從預處理結果中提取影像數據和臨床特徵
                image_data = preprocessed_item['data']  # 影像數據，已經預處理好的 numpy 陣列
                clinical_data = preprocessed_item['clinical_data']  # 臨床特徵字典
                clinical_mask = preprocessed_item['clinical_mask']  # 臨床特徵掩碼字典
                
                # 將 numpy 轉換為 tensor (本來在iterator就已經是 tensor 了)
                # image_tensor = torch.from_numpy(image_data).float()
                
                # 將臨床特徵轉換為 tensor 並移動到正確的設備上
                if self.no_use_input_cli_data:
                    clinical_features = {
                        'location': torch.tensor([self.missing_flags['location']], dtype=torch.long),
                        't_stage': torch.tensor([self.missing_flags['t_stage']], dtype=torch.long),
                        'n_stage': torch.tensor([self.missing_flags['n_stage']], dtype=torch.long),
                        'm_stage': torch.tensor([self.missing_flags['m_stage']], dtype=torch.long),
                        'dataset': torch.tensor([clinical_data['dataset']], dtype=torch.long) # 就算不使用臨床資料， dataset 還是要給
                    } 
                else:
                    clinical_features = {
                        'location': torch.tensor([clinical_data['location']], dtype=torch.long),
                        't_stage': torch.tensor([clinical_data['t_stage']], dtype=torch.long),
                        'n_stage': torch.tensor([clinical_data['n_stage']], dtype=torch.long),
                        'm_stage': torch.tensor([clinical_data['m_stage']], dtype=torch.long),
                        'dataset': torch.tensor([clinical_data['dataset']], dtype=torch.long)
                    }
                
                # 生成是否有臨床數據的標記，用於之後的處理
                has_clinical_data = any(clinical_mask.values())
                has_clinical_data_tensor = torch.tensor([has_clinical_data], dtype=torch.bool)

                ofile = preprocessed_item['ofile']  # 輸出文件名 (如果存在)
                properties = preprocessed_item['properties']  # 原始影像屬性

                # 紀錄時間
                print("當前時間:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                if ofile is not None:
                    print(f'\n正在預測 {os.path.basename(ofile)}:')
                else:
                    print(f'\n正在預測形狀為 {image_data.shape} 的影像:')
                print(f'全程在設備執行: {self.perform_everything_on_device}')

                # 流量控制：避免 GPU 預測過快導致導出進程阻塞
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=4)
                while not proceed:
                    sleep(0.1) # 等待 100 毫秒
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=4)
                
                # 呼叫修改後的預測邏輯，同時傳入影像數據和臨床特徵
                # 它會返回分割的 logits 和臨床屬性預測的 logits
                prediction_seg_logits, clinical_attr_preds = self.predict_logits_and_attributes_from_preprocessed_data(
                    image_data, clinical_features
                )
                
                # 將分割預測結果的匯出工作放入背景行程池中執行 (非同步)
                prediction_seg_logits = prediction_seg_logits.cpu().detach().numpy()

                # 保存臨床預測 txt
                if ofile is not None:
                    save_clinical_prediction_txt(ofile, clinical_attr_preds)

                # 收集到 all_clinical_results
                case_id = os.path.basename(ofile) if ofile is not None else f"case_{i}"
                labels = clinical_logits_to_labels(clinical_attr_preds)
                labels["Case_Index"] = case_id  # 設置案例ID
                all_clinical_results.append(labels)

                # segmentation 匯出
                if ofile is not None:
                    print('將分割預測發送給後台進程進行重採樣和導出')
                    r.append(
                        export_pool.starmap_async(
                            export_prediction_from_logits, # 導出分割結果的函數
                            ((prediction_seg_logits, properties, self.configuration_manager, self.plans_manager,
                              self.dataset_json, ofile, save_probabilities),)
                        )
                    )
                else:
                    print('將分割預測發送給後台進程進行重採樣')
                    # 如果不需要保存文件，則直接返回分割結果和臨床預測結果
                    r.append(
                        export_pool.starmap_async(
                            self._collate_prediction_outputs_for_return, # 新增一個輔助函數來收集所有返回
                            ((prediction_seg_logits, self.plans_manager, self.configuration_manager, self.label_manager,
                              properties, save_probabilities, clinical_attr_preds, has_clinical_data_tensor),) # 傳遞臨床預測結果及標誌
                        )
                    )

                if ofile is not None:
                    print(f'完成 {os.path.basename(ofile)}')
                else:
                    print(f'\n完成形狀為 {image_data.shape} 的影像預測:')

            # 等待所有異步操作完成並獲取結果
            print("等待所有分割結果導出完成...")
            ret = [i.get() for i in r]
            print("所有分割結果導出完成！")

            # 保存所有臨床預測到 csv
            if all_clinical_results and ofile is not None:
                csv_path = os.path.join(os.path.dirname(ofile), "clinical_predictions.csv")
                save_clinical_prediction_csv(csv_path, all_clinical_results)
            # 清理數據迭代器資源 (如果有使用多線程增強器)
            if isinstance(data_iterator, MultiThreadedAugmenter):
                data_iterator._finish()

            # 清理緩存和釋放 GPU 記憶體
            compute_gaussian.cache_clear()
            empty_cache(self.device)
            return ret

    @staticmethod
    def _collate_prediction_outputs_for_return(prediction_logits_seg: np.ndarray, 
                                                 plans_manager: PlansManager, 
                                                 configuration_manager: ConfigurationManager, 
                                                 label_manager: LabelManager,
                                                 properties: dict, 
                                                 return_probabilities: bool, 
                                                 clinical_attr_preds: dict, 
                                                 has_clinical_data_flag: torch.Tensor):
        """
        輔助函數：在不需要保存文件時，收集分割預測結果（包含機率圖）和臨床預測結果。
        這個函數會在子進程中運行，將所有預測相關結果打包成一個字典返回。
        
        Args:
            prediction_logits_seg (np.ndarray): 分割模型的 logits 輸出 (NumPy 陣列)。
            plans_manager (PlansManager): 計劃管理器。
            configuration_manager (ConfigurationManager): 配置管理器。
            label_manager (LabelManager): 標籤管理器。
            properties (dict): 原始影像的屬性字典。
            return_probabilities (bool): 是否返回分割的機率圖。
            clinical_attr_preds (dict): 臨床屬性模型的 logits 輸出字典。
            has_clinical_data_flag (torch.Tensor): 標記該案例是否有臨床資料的布林張量。
            
        Returns:
            dict: 包含分割結果、機率圖、臨床屬性 logits 和臨床資料標誌的字典。
        """
        # 轉換分割 logits 為分割圖和機率圖
        segmentation, probabilities = convert_predicted_logits_to_segmentation_with_correct_shape(
            prediction_logits_seg, plans_manager, configuration_manager, label_manager,
            properties, return_probabilities
        )
        return {
            'segmentation': segmentation,
            'probabilities': probabilities,
            'clinical_attr_preds': clinical_attr_preds, # 臨床屬性預測 logits (Torch 張量)
            'has_clinical_data_flag': has_clinical_data_flag.item() # 布林值 (如果批次為1，直接取 item)
        }

    @torch.inference_mode()
    def predict_logits_and_attributes_from_preprocessed_data(self, data: torch.Tensor, 
                                                             clinical_features: dict) -> Tuple[torch.Tensor, dict]:
        """
        重要！如果是級聯模型，前階段分割必須已以 one-hot 編碼形式堆疊在影像頂部！
        此方法同時返回分割的 logits 和臨床屬性預測的 logits。
        
        Args:
            data (torch.Tensor): 預處理後的影像數據 (4D 張量)。
            clinical_features (dict): 預處理後的臨床特徵字典，包含五個特徵：
                                     {'location': tensor, 't_stage': tensor, 'n_stage': tensor, 'm_stage': tensor, 'dataset': tensor}
            
        Returns:
            Tuple[torch.Tensor, dict]: 包含分割 logits 和臨床屬性 logits 字典的元組。
        """
        n_threads = torch.get_num_threads()
        torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)

        prediction_seg = None
        # 初始化臨床屬性預測的累加器
        prediction_cli = {k: None for k in ['location', 't_stage', 'n_stage', 'm_stage', 'dataset', 'missing_flags']}

        # 生成文字描述（在推論階段根據臨床特徵動態生成）
        text_descriptions = None
        if hasattr(self, 'clinical_data_label_encoder') and self.clinical_data_label_encoder is not None:
            # 將 tensor 轉換為標量值
            clinical_data_dict = {
                'location': clinical_features['location'].item(),
                't_stage': clinical_features['t_stage'].item(),
                'n_stage': clinical_features['n_stage'].item(),
                'm_stage': clinical_features['m_stage'].item(),
                'dataset': clinical_features['dataset'].item()
            }
            
            # 生成臨床掩碼（在推論階段，我們使用原始數據）
            clinical_mask_dict = {
                'location': clinical_data_dict['location'] != self.missing_flags['location'],
                't_stage': clinical_data_dict['t_stage'] != self.missing_flags['t_stage'],
                'n_stage': clinical_data_dict['n_stage'] != self.missing_flags['n_stage'],
                'm_stage': clinical_data_dict['m_stage'] != self.missing_flags['m_stage'],
                'dataset': clinical_data_dict['dataset'] != self.missing_flags['dataset']
            }
            
            # 生成文字描述
            text_description = self._generate_text_description(clinical_data_dict, clinical_mask_dict)
            text_descriptions = [text_description]  # 單個樣本的文字描述列表

        # ensemble 用， list_of_parameters 其實是 list of 模型權重 (長度為 fold 數量)
        for params in self.list_of_parameters:
            # 加載模型參數
            if not isinstance(self.network, OptimizedModule):
                self.network.load_state_dict(params)
            else:
                self.network._orig_mod.load_state_dict(params) # 處理編譯後模型

            # 呼叫修改後的 predict_sliding_window_return_logits，傳入影像和臨床特徵
            seg_logits_single_model, cli_logits_single_model = self.predict_sliding_window_return_logits(
                data,
                clinical_features['location'],
                clinical_features['t_stage'],
                clinical_features['n_stage'],
                clinical_features['m_stage'],
                clinical_features['dataset'],
                text_descriptions=text_descriptions
            )

            # 累加預測結果
            if prediction_seg is None:
                prediction_seg = seg_logits_single_model.to('cpu')
                for k, v in cli_logits_single_model.items():
                    prediction_cli[k] = v.to('cpu')
            else:
                prediction_seg += seg_logits_single_model.to('cpu')
                for k, v in cli_logits_single_model.items():
                    prediction_cli[k] += v.to('cpu')
        
        # 將多個 fold 計算的結果 進行平均
        if len(self.list_of_parameters) > 1:
            prediction_seg /= len(self.list_of_parameters)
            for k in prediction_cli.keys():
                if prediction_cli[k] is not None: # 確保屬性存在
                    prediction_cli[k] /= len(self.list_of_parameters)

        if self.verbose:
            print('預測完成')
        torch.set_num_threads(n_threads) # 恢復線程設置
        return prediction_seg, prediction_cli

    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor, clinical_features: dict, text_descriptions=None) -> Tuple[torch.Tensor, dict]:
        """
        執行鏡像增強預測（內部方法），同時考慮臨床特徵和文本描述。
        根據配置決定是否使用測試時數據增強。臨床特徵不進行鏡像操作。
        
        Args:
            x (torch.Tensor): 輸入影像批次。
            clinical_features (dict): 輸入臨床特徵字典，包含五個特徵：
                                     {'location': tensor, 't_stage': tensor, 'n_stage': tensor, 'm_stage': tensor, 'dataset': tensor}
            text_descriptions (list or None): 文本描述列表，可選參數，如果提供將用於文本編碼
            
        Returns:
            Tuple[torch.Tensor, dict]: 鏡像增強後的分割 logits 和臨床屬性 logits 字典。
        """
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        
        # 初始預測 (使用原始影像和臨床特徵)
        # MyMultiModel 的 forward 方法需要 text_descriptions 參數，如果沒有提供則使用 None
        if text_descriptions is None:
            text_descriptions = [None] * x.shape[0]  # 為批次中的每個樣本提供 None
        
        seg_prediction, cli_prediction = self.network(
            x,
            clinical_features['location'].to(self.device),
            clinical_features['t_stage'].to(self.device),
            clinical_features['n_stage'].to(self.device),
            clinical_features['m_stage'].to(self.device),
            clinical_features['dataset'].to(self.device),
            text_descriptions=text_descriptions
        )

        # 處理 deep_supervision 返回的列表
        # 若 seg_prediction 是列表，取第一個元素（最高解析度的輸出）
        if isinstance(seg_prediction, list):
            seg_prediction = seg_prediction[0]
        
        # 若 cli_prediction 包含列表，取每個列表的第一個元素
        if isinstance(cli_prediction, dict):
            for k in cli_prediction.keys():
                if isinstance(cli_prediction[k], list) and len(cli_prediction[k]) > 0:
                    cli_prediction[k] = cli_prediction[k][0]

        # 鏡像增強處理
        if mirror_axes is not None:
            # 驗證鏡像軸有效性
            assert max(mirror_axes) <= x.ndim - 3, '鏡像軸與輸入維度不匹配！'
            # 調整軸索引，使其與影像的空間維度對齊 (跳過批次和通道維度)
            mirror_axes = [m + 2 for m in mirror_axes]
            
            # 生成所有鏡像組合 (例如軸0、軸1、軸0+軸1)
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]

            # 執行鏡像預測並累加
            for axes in axes_combinations:
                mirrored_x = torch.flip(x, axes)

                # 臨床特徵不進行鏡像，直接傳遞
                if text_descriptions is not None:
                    # 如果有文本描述，調用支援文本的模型
                    m_seg_pred, m_cli_pred = self.network(
                        mirrored_x,
                        clinical_features['location'].to(self.device),
                        clinical_features['t_stage'].to(self.device),
                        clinical_features['n_stage'].to(self.device),
                        clinical_features['m_stage'].to(self.device),
                        clinical_features['dataset'].to(self.device),
                        text_descriptions=text_descriptions
                    )
                else:
                    # 使用原來的調用方式
                    m_seg_pred, m_cli_pred = self.network(
                        mirrored_x,
                        clinical_features['location'].to(self.device),
                        clinical_features['t_stage'].to(self.device),
                        clinical_features['n_stage'].to(self.device),
                        clinical_features['m_stage'].to(self.device),
                        clinical_features['dataset'].to(self.device)
                    )
                
                # 處理 deep_supervision 返回的列表
                if isinstance(m_seg_pred, list):
                    m_seg_pred = m_seg_pred[0]
                
                if isinstance(m_cli_pred, dict):
                    for k in m_cli_pred.keys():
                        if isinstance(m_cli_pred[k], list) and len(m_cli_pred[k]) > 0:
                            m_cli_pred[k] = m_cli_pred[k][0]

                # 分割結果需要反向鏡像後累加
                seg_prediction += torch.flip(m_seg_pred, axes)

                # 臨床屬性結果不需要反向鏡像，直接累加
                for k in cli_prediction.keys():
                    if cli_prediction[k] is not None: # 確保屬性存在
                        cli_prediction[k] += m_cli_pred[k]

            # 計算加權平均
            seg_prediction /= (len(axes_combinations) + 1)
            for k in cli_prediction.keys():
                if cli_prediction[k] is not None: # 確保屬性存在
                    cli_prediction[k] /= (len(axes_combinations) + 1)

        return seg_prediction, cli_prediction

    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits(self,
                                                       data: torch.Tensor,
                                                       clinical_features: dict, # 接收臨床特徵字典
                                                       slicers,
                                                       do_on_device: bool = True,
                                                       text_descriptions=None):
        """
        滑動窗口預測核心實現（內部方法），支援多模態輸入。
        使用生產者-消費者模式高效處理大影像。
        
        Args:
            data (torch.Tensor): 填充後的影像數據。
            clinical_features (dict): 該案例的臨床特徵字典，包含五個臨床特徵。
                                     {'location': tensor, 't_stage': tensor, 'n_stage': tensor, 'm_stage': tensor, 'dataset': tensor}
            slicers: 滑動窗口切片器列表。
            do_on_device (bool): 是否在設備 (GPU) 上執行所有操作。
            text_descriptions (list or None): 文本描述列表，可選參數，如果提供將用於文本編碼
            
        Returns:
            Tuple[torch.Tensor, dict]: 分割 logits 和臨床屬性 logits 字典。
        """
        predicted_logits_seg = n_predictions = prediction_seg = gaussian = workon_image = None
        predicted_logits_cli = {k: None for k in ['location', 't_stage', 'n_stage', 'm_stage', 'dataset', 'missing_flags']}
        
        results_device = self.device if do_on_device else torch.device('cpu') # 結果儲存設備

        def producer(d, slh, q, cli_features, text_descs):
            """生產者函數：將影像切片和臨床特徵放入隊列"""
            for s in slh:
                # 將影像切片和臨床特徵複製到 GPU 並放入隊列
                # 臨床特徵是對整個 case 有效，所以每個 patch 都傳遞相同的臨床特徵
                q.put((
                    torch.clone(d[s][None], memory_format=torch.contiguous_format).to(self.device),
                    cli_features, # 將所有臨床特徵直接傳遞
                    text_descs,   # 傳遞文本描述
                    s
                ))
            q.put('end') # 結束標記
            try:
                empty_cache(self.device) # 嘗試清理生產者線程的 GPU 緩存
            except RuntimeError as e:
                # 忽略因設備繁忙導致的清理失敗 (例如在 DDP 環境下)
                if "No CUDA context found" not in str(e):
                    raise e

        try:
            if self.verbose:
                print(f'移動影像到設備 {results_device}')
            data = data.to(results_device) # 將整個影像移動到結果設備

            queue = Queue(maxsize=4) # 限制隊列大小，避免記憶體溢出
            # 啟動生產者線程
            t = Thread(target=producer, args=(data, slicers, queue, clinical_features, text_descriptions))
            t.start()

            if self.verbose:
                print(f'在設備 {results_device} 預分配結果數組')
            # 預分配分割結果的記憶體空間
            predicted_logits_seg = torch.zeros(
                (self.label_manager.num_segmentation_heads, *data.shape[1:]),
                dtype=torch.half, # 使用半精度以節省記憶體
                device=results_device
            )
            # 預分配預測次數的記憶體空間，用於加權平均
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            # 為臨床屬性預測分配空間
            # 我們從模型的 init_kwargs 中獲取類別數量或使用預設值
            # 注意：這裡的 `network` 可能是 DDP 的 `module` 或 `_orig_mod`
            if isinstance(self.network, (nn.parallel.DistributedDataParallel, OptimizedModule)):
                model_base = self.network.module if isinstance(self.network, DistributedDataParallel) else self.network._orig_mod
                attr_init_kwargs = getattr(model_base, 'init_kwargs', None)
            else:
                attr_init_kwargs = getattr(self.network, 'init_kwargs', None)

            # 如果無法獲取 init_kwargs，使用預設值
            cli_output_shapes = {
                'location': (self.num_clinical_classes['location'],),  # 預設 8 個類別
                't_stage': (self.num_clinical_classes['t_stage'],),   # 預設 6 個類別
                'n_stage': (self.num_clinical_classes['n_stage'],),   # 預設 4 個類別
                'm_stage': (self.num_clinical_classes['m_stage'],),   # 預設 3 個類別
                'dataset': (self.num_clinical_classes['dataset'],),  # 預設 3 個類別
                'missing_flags': (4,)  # 預設 4 個標誌
            }
            
            # 如果有 attr_init_kwargs，從中獲取類別數量
            if attr_init_kwargs:
                cli_output_shapes = {
                    'location': (attr_init_kwargs.get('num_location_classes'),),
                    't_stage': (attr_init_kwargs.get('num_t_stage_classes'),),
                    'n_stage': (attr_init_kwargs.get('num_n_stage_classes'),),
                    'm_stage': (attr_init_kwargs.get('num_m_stage_classes'),),
                    'dataset': (attr_init_kwargs.get('num_dataset_classes'),),
                    'missing_flags': (attr_init_kwargs.get('missing_flags_dim'),)
                }
            
            # 初始化臨床屬性 logits 字典
            predicted_logits_cli = {}
            for k, shape in cli_output_shapes.items():
                predicted_logits_cli[k] = torch.zeros(shape, dtype=torch.float32, device=results_device) # 使用 float32 保持精度

            # 高斯加權設置
            if self.use_gaussian:
                gaussian = compute_gaussian(
                    tuple(self.configuration_manager.patch_size),
                    sigma_scale=1. / 8, # 高斯核標準差
                    value_scaling_factor=10, # 值縮放因子
                    device=results_device
                )
            else:
                gaussian = 1 # 不使用高斯加權

            # 進度條設置
            if not self.allow_tqdm and self.verbose:
                print(f'運行預測: {len(slicers)} 步')
            
            # 使用 tqdm 進度條
            with tqdm(desc=None, total=len(slicers), disable=not self.allow_tqdm) as pbar:
                while True:
                    item = queue.get() # 從隊列中獲取數據
                    if item == 'end': # 如果遇到結束標記
                        queue.task_done()
                        break
                    
                    workon_image, workon_cli_features, workon_text_descriptions, sl = item # 解包數據：影像切片、臨床特徵、文本描述、切片器
                    
                    # 真正使用鏡像增強預測
                    # 返回的 prediction_seg 應為 (1, num_classes, D, H, W)
                    # 返回的 cli_prediction 應為 {'location': (1, num_loc_classes), ...}
                    prediction_seg_patch, cli_prediction_patch = self._internal_maybe_mirror_and_predict(
                        workon_image, workon_cli_features, workon_text_descriptions
                    )
                    prediction_seg_patch = prediction_seg_patch.to(results_device) # 移到結果設備

                    # 移除 batch 維度
                    if isinstance(prediction_seg_patch, list):
                        prediction_seg_patch = prediction_seg_patch[0]
                    if prediction_seg_patch.shape[0] == 1:
                        prediction_seg_patch = prediction_seg_patch.squeeze(0)

                    # 應用高斯加權到分割結果
                    if self.use_gaussian:
                        prediction_seg_patch *= gaussian
                    
                    # 累加分割結果和預測次數
                    predicted_logits_seg[sl] += prediction_seg_patch
                    n_predictions[sl[1:]] += gaussian

                    # --------處理臨床屬性結果：採用基於前景比例的加權平均策略--------
                    # 計算當前 patch 中的前景比例作為權重
                    # 假設分割前景類別通常為 1+ (0 為背景)
                    with torch.no_grad():
                        # 取得預測的分割結果
                        seg_pred = torch.argmax(prediction_seg_patch, dim=0)
                        # 計算前景像素的比例
                        foreground_ratio = (seg_pred > 0).float().mean().item()
                        
                        # 對於腫瘤預測，前景比例通常很小，需要提升權重的對比度
                        # 使用輕微的非線性函數 (如 sqrt) 來增強小值的影響，同時避免極端權重
                        weight = np.sqrt(foreground_ratio * 5) if foreground_ratio > 0 else 0.01
                        
                        # 防止極端值 (權重最大值為 5.0，最小值為 0.01)
                        weight = min(max(weight, 0.01), 5.0)
                        
                        # 如果是首次處理，初始化累加器
                        if not hasattr(self, 'cli_predictions_weighted_sum'):
                            self.cli_predictions_weighted_sum = {k: None for k in predicted_logits_cli.keys()}
                            self.foreground_weights_sum = 0
                            # 額外跟踪各個臨床特徵的權重累加過程
                            self.debug_weights = []
                            
                        # 記錄此次patch的前景比例,權重,目前累積的權重用於調試
                        if hasattr(self, 'debug_weights'):
                            self.debug_weights.append({
                                'foreground_ratio': foreground_ratio,
                                'weight': weight,
                                'total_weight_so_far': self.foreground_weights_sum
                            })
                        
                        # 加權累加臨床預測結果
                        for k, v in cli_prediction_patch.items(): # cli_prediction_patch 是一個字典，v遍歷所有特徵預測Logits
                            if predicted_logits_cli[k] is not None and v is not None:
                                v_device = v.to(results_device)
                                # 如果有 batch 維度，則移除
                                if v_device.dim() > 1 and v_device.shape[0] == 1:
                                    v_device = v_device.squeeze(0)
                                
                                # 如果是首次處理，初始化累加器
                                if self.cli_predictions_weighted_sum[k] is None:
                                    # 將預測出來的 logits * 當前 patch 的權重
                                    self.cli_predictions_weighted_sum[k] = v_device * weight 
                                else:
                                    self.cli_predictions_weighted_sum[k] += v_device * weight
                        
                        # 紀錄目前權重累積了多少
                        self.foreground_weights_sum += weight
                        
                    queue.task_done()
                    pbar.update() # 更新進度條

            queue.join() # 等待隊列處理完成
            
            # 計算分割結果的加權平均 ( patch 會有重疊 所以會計算每個 voxel 被預測了幾次 下去取平均)
            torch.div(predicted_logits_seg, n_predictions, out=predicted_logits_seg)

            # 計算臨床預測的加權平均
            if hasattr(self, 'cli_predictions_weighted_sum') and self.foreground_weights_sum > 0:
                for k in predicted_logits_cli.keys():
                    if self.cli_predictions_weighted_sum[k] is not None:
                        predicted_logits_cli[k] = self.cli_predictions_weighted_sum[k] / self.foreground_weights_sum
                
                # 輸出調試信息
                if self.verbose and hasattr(self, 'debug_weights'):
                    print(f"臨床預測加權信息: 總權重={self.foreground_weights_sum:.4f}, 片段數={len(self.debug_weights)}")
                    if len(self.debug_weights) > 0:
                        max_weight = max(self.debug_weights, key=lambda x: x['weight'])
                        min_weight = min(self.debug_weights, key=lambda x: x['weight'])
                        print(f"  最大權重: {max_weight['weight']:.4f} (前景比例: {max_weight['foreground_ratio']:.4f})")
                        print(f"  最小權重: {min_weight['weight']:.4f} (前景比例: {min_weight['foreground_ratio']:.4f})")
                
                # 清理，避免影響下一次調用
                del self.cli_predictions_weighted_sum
                del self.foreground_weights_sum
                if hasattr(self, 'debug_weights'):
                    del self.debug_weights
            
            # 檢查分割預測結果中是否存在無窮值 (通常指示數值不穩定)
            if torch.any(torch.isinf(predicted_logits_seg)):
                raise RuntimeError(
                    '在分割預測數組中檢測到inf值！如果持續出現此問題，'
                    '請降低 compute_gaussian 中的 value_scaling_factor 或將 predicted_logits 的 dtype 提高為 fp32'
                )

        except Exception as e:
            # 異常時清理資源
            del predicted_logits_seg, n_predictions, prediction_seg, gaussian, workon_image
            del predicted_logits_cli
            # 清理加權平均的臨時變數
            if hasattr(self, 'cli_predictions_weighted_sum'):
                del self.cli_predictions_weighted_sum
            if hasattr(self, 'foreground_weights_sum'):
                del self.foreground_weights_sum
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        
        # 返回分割結果和臨床屬性結果
        return predicted_logits_seg, predicted_logits_cli

    @torch.inference_mode()
    def predict_sliding_window_return_logits(self, data: torch.Tensor, loc: torch.Tensor, t: torch.Tensor, n: torch.Tensor, m: torch.Tensor, dataset: torch.Tensor, text_descriptions=None):
        """
        使用滑動窗口方法進行預測，返回原始 logits（用於驗證）。
        支持多模態數據輸入（影像 + 臨床特徵 + 文本描述）。
        
        Args:
            data (torch.Tensor): 影像數據，形狀為 [C, X, Y, Z]
            loc (torch.Tensor): 腫瘤位置特徵，形狀為 [1]
            t (torch.Tensor): T 分期特徵，形狀為 [1]
            n (torch.Tensor): N 分期特徵，形狀為 [1]
            m (torch.Tensor): M 分期特徵，形狀為 [1]
            dataset (torch.Tensor): 資料集特徵，形狀為 [1]
            text_descriptions (list or None): 文本描述列表，可選參數，如果提供將用於文本編碼
            
        Returns:
            Tuple[torch.Tensor, dict]: 分割預測 logits 和臨床屬性預測 logits
        """
        # 將臨床特徵組織成字典
        clinical_features = {
            'location': loc.unsqueeze(0) if loc.dim() == 0 else loc,
            't_stage': t.unsqueeze(0) if t.dim() == 0 else t,
            'n_stage': n.unsqueeze(0) if n.dim() == 0 else n,
            'm_stage': m.unsqueeze(0) if m.dim() == 0 else m,
            'dataset': dataset.unsqueeze(0) if dataset.dim() == 0 else dataset
        }
        
        assert isinstance(data, torch.Tensor), "輸入影像必須是 torch.Tensor 類型。"
    
        self.network = self.network.to(self.device) # 確保網路在正確設備
        self.network.eval() # 設定為評估模式
        empty_cache(self.device) # 清理設備緩存

        # 自動混合精度設置（僅 CUDA 設備）
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            assert data.ndim == 4, '輸入影像必須是4D張量（通道, x, y, z）。'

            if self.verbose:
                print(f'輸入影像形狀: {data.shape}')
                print(f"步長: {self.tile_step_size}")
                print(f"鏡像軸: {self.allowed_mirroring_axes if self.use_mirroring else '無'}")

            # 邊界填充處理：當影像尺寸小於 patch_size 時，需要對影像進行填充
            padded_data, slicer_revert_padding = pad_nd_image(
                data, 
                self.configuration_manager.patch_size, # 目標尺寸 (patch_size)
                'constant', # 填充類型為常數
                {'value': 0}, # 填充值為 0
                True, # 返回切片器用於恢復原始尺寸
                None # 無需指定邊框
            )
            
            # 如果特徵不存在或為 None，則使用預設的缺失值填充
            if 'location' not in clinical_features or clinical_features['location'] is None:
                clinical_features['location'] = torch.tensor([[self.missing_flags['location']]], device=self.device)
            if 't_stage' not in clinical_features or clinical_features['t_stage'] is None:
                clinical_features['t_stage'] = torch.tensor([[self.missing_flags['t_stage']]], device=self.device)
            if 'n_stage' not in clinical_features or clinical_features['n_stage'] is None:
                clinical_features['n_stage'] = torch.tensor([[self.missing_flags['n_stage']]], device=self.device)
            if 'm_stage' not in clinical_features or clinical_features['m_stage'] is None:
                clinical_features['m_stage'] = torch.tensor([[self.missing_flags['m_stage']]], device=self.device)
            if 'dataset' not in clinical_features or clinical_features['dataset'] is None:
                clinical_features['dataset'] = torch.tensor([[self.missing_flags['dataset']]], device=self.device)

            # 記錄哪些特徵是缺失的（用於後續的 missing_flags 處理）
            is_loc_missing = clinical_features['location'].item() == self.missing_flags['location'] if clinical_features['location'].numel() == 1 else False
            is_t_missing = clinical_features['t_stage'].item() == self.missing_flags['t_stage'] if clinical_features['t_stage'].numel() == 1 else False
            is_n_missing = clinical_features['n_stage'].item() == self.missing_flags['n_stage'] if clinical_features['n_stage'].numel() == 1 else False
            is_m_missing = clinical_features['m_stage'].item() == self.missing_flags['m_stage'] if clinical_features['m_stage'].numel() == 1 else False
            is_dataset_missing = clinical_features['dataset'].item() == self.missing_flags['dataset'] if clinical_features['dataset'].numel() == 1 else False

            # 打印調試信息
            if self.verbose:
                print(f"臨床特徵: loc={clinical_features['location'].item()}, t={clinical_features['t_stage'].item()}, n={clinical_features['n_stage'].item()}, m={clinical_features['m_stage'].item()}, dataset={clinical_features['dataset'].item()}")
                print(f"缺失標記: loc={is_loc_missing}, t={is_t_missing}, n={is_n_missing}, m={is_m_missing}, dataset={is_dataset_missing}")

            # 確保所有特徵都在正確的設備上
            for k in clinical_features.keys():
                clinical_features[k] = clinical_features[k].to(self.device)

            # 獲取滑動窗口的範圍列表
            # 回傳的是 切片範圍的列表 [patch1, patch2 ...]
            # 每個 patch 裡面類似是 [(x起點,x終點), (y起點,y終點), (z起點,z終點)] 實際更複雜)
            slicers = self._internal_get_sliding_window_slicers(padded_data.shape[1:])

            # 執行預測
            try:
                predicted_seg_logits, predicted_cli_logits = self._internal_predict_sliding_window_return_logits(
                    padded_data, clinical_features, slicers, self.perform_everything_on_device, text_descriptions
                )
            except RuntimeError:
                print('在設備上預測失敗，可能是由於記憶體不足。將結果數組移至 CPU')
                empty_cache(self.device)
                predicted_seg_logits, predicted_cli_logits = self._internal_predict_sliding_window_return_logits(
                    padded_data, clinical_features, slicers, False, text_descriptions
                )

            # 移除填充區域：將分割 logits 恢復到原始影像尺寸
            predicted_seg_logits = predicted_seg_logits[(slice(None), *slicer_revert_padding[1:])]

            empty_cache(self.device) # 清理設備緩存

            return predicted_seg_logits, predicted_cli_logits


    def _internal_get_data_iterator_from_lists_of_filenames(
        self,
        input_list_of_lists: List[List[str]],
        seg_from_prev_stage_files: Union[List[str], None],
        output_filenames_truncated: Union[List[str], None],
        num_processes: int
    ):
        """改用多模態 iterator"""
        return preprocessing_iterator_fromfiles_multimodal(
            input_list_of_lists,
            seg_from_prev_stage_files,
            output_filenames_truncated,
            self.plans_manager,
            self.dataset_json,
            self.configuration_manager,
            num_processes,
            self.device.type == 'cuda',
            self.verbose_preprocessing,
            clinical_data_dir=self.clinical_data_dir
        )

    def predict_from_files(
        self,
        list_of_lists_or_source_folder: Union[str, List[List[str]]],
        output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
        save_probabilities: bool = False,
        overwrite: bool = True,
        num_processes_preprocessing: int = 3,
        num_processes_segmentation_export: int = 3,
        folder_with_segs_from_prev_stage: str = None,
        num_parts: int = 1,
        part_id: int = 0,
        max_cases: int = None  # 新增參數：限制處理的案例數量
    ):
        """
        執行流程:
            先呼叫 _manage_input_and_output_lists 整理輸入/輸出清單
            再呼叫 _internal_get_data_iterator_from_lists_of_filenames 建立 data_iterator
            最後呼叫 self.predict_from_data_iterator(...)
        """

        # 檢查 id 是否超出範圍
        assert part_id <= num_parts, ("Part ID must be smaller than num_parts. Remember that we start counting with 0. "
                                      "So if there are 3 parts then valid part IDs are 0, 1, 2")
        
        # 檢查輸入類型
        # 如果是字串，則視為資料夾；
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        # 如果是列表，則代表輸入的是 numpy 陣列的輸出檔名
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
        else:
            output_folder = None

        # cascade 模型使用 (用不到)
        if self.configuration_manager.previous_stage_name is not None:
            assert folder_with_segs_from_prev_stage is not None, \
                f'The requested configuration is a cascaded network. It requires the segmentations of the previous ' \
                f'stage ({self.configuration_manager.previous_stage_name}) as input. Please provide the folder where' \
                f' they are located via folder_with_segs_from_prev_stage'


        # 把推論參數(my_init_kwargs), dataset.json, plan,json 存到 output_folder
        ########################
        # let's store the input arguments so that its clear what was used to generate the prediction
        if output_folder is not None:
            my_init_kwargs = {}
            for k in inspect.signature(self.predict_from_files).parameters.keys():
                my_init_kwargs[k] = locals()[k]
            my_init_kwargs = deepcopy(
                my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
            recursive_fix_for_json_export(my_init_kwargs)
            maybe_mkdir_p(output_folder)
            save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

            # we need these two if we want to do things with the predictions like for example apply postprocessing
            save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
            save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)
        #######################

        # 1. 整理輸入/輸出清單（直接沿用父類邏輯）
        list_of_lists, output_filenames, seg_prev_files = self._manage_input_and_output_lists(
            list_of_lists_or_source_folder,
            output_folder_or_list_of_truncated_output_files,
            folder_with_segs_from_prev_stage,
            overwrite,
            part_id,
            num_parts,
            save_probabilities
        )
        
        # 如果設置了 max_cases，限制處理的案例數量
        if max_cases is not None and max_cases > 0:
            print(f"限制處理案例數為 {max_cases} (原有 {len(list_of_lists)} 個案例)")
            list_of_lists = list_of_lists[:max_cases]
            if output_filenames is not None:
                output_filenames = output_filenames[:max_cases]
            if seg_prev_files is not None:
                seg_prev_files = seg_prev_files[:max_cases]
        if len(list_of_lists) == 0:
            print(f'進程 {part_id}/{num_parts} 無需處理新案例，跳過')
            return []

        # 2. 建立新的多模態 iterator
        data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(
            list_of_lists,
            seg_prev_files,
            output_filenames,
            num_processes_preprocessing
        )

        # 3. 交給 predict_from_data_iterator 完成推論
        return self.predict_from_data_iterator(
            data_iterator,
            save_probabilities,
            num_processes_segmentation_export
        )

    # ----------------------------------
    # 訓練最後的驗證不會用到 真的推論時才會用到
    def initialize_from_trained_model_folder(
        self,
        model_training_output_dir: str,
        use_folds: Union[Tuple[Union[int, str]], None],
        checkpoint_name: str = 'checkpoint_final.pth'
    ):
        """
        從訓練好的模型目錄初始化多模態預測器。
        與父類唯一差別：讀取額外維度 prompt_dim 並建立 MyMultiModel。
        """
        # 0. 自動偵測折數（與父類相同）
        if use_folds is None:
            use_folds = self.auto_detect_available_folds(model_training_output_dir,
                                                        checkpoint_name)

        # 1. 讀取資料集與計劃檔
        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        # 2. 讀取檢查點
        if isinstance(use_folds, str):
            use_folds = [use_folds]

        # ensemble 用
        parameters = [] # 保存總共有哪些 fold
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(
                join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                map_location=torch.device('cpu'), weights_only=False
            )
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint.get(
                    'inference_allowed_mirroring_axes', None
                )

            parameters.append(checkpoint['network_weights'])

        # 3. 建立計劃與配置管理器
        configuration_manager = plans_manager.get_configuration(configuration_name)
        num_input_channels_img = determine_num_input_channels(
            plans_manager, configuration_manager, dataset_json
        )

        # 4. 動態載入trainer並建立網路
        trainer_class = recursive_find_python_class(
            join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
            trainer_name,
            'nnunetv2.training.nnUNetTrainer'
        )
        print(f"載入訓練器類別: {trainer_name}")
        if trainer_class is None:
            raise RuntimeError(f'找不到訓練器類別 {trainer_name}')

        # 建立模型
        my_model_init_kwargs = {
            'input_channels': num_input_channels_img,
            'num_classes': plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            'deep_supervision': False,
            'clinical_csv_dir': self.clinical_data_dir
        }
        
        # 打印模型初始化參數
        print(f"模型初始化參數: {my_model_init_kwargs}")
        
        network = trainer_class.build_network_architecture(MyMultiModel, my_model_init_kwargs).to(self.device)

        # 加載網絡權重
        network.load_state_dict(parameters[0])
        self.network = network

        # 6. 儲存其餘必要屬性
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters # ensemble 用 紀錄總共有幾個 fold 的模型權重
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)

        # 7. torch.compile 加速（若環境變數開啟）
        if ('nnUNet_compile' in os.environ and
                os.environ['nnUNet_compile'].lower() in ('true', '1', 't') and
                not isinstance(self.network, OptimizedModule)):
            print('啟用 torch.compile 加速（排除文字編碼器）')
            self.network = torch.compile(self.network)

    # 直接 super 原生 predictor 方法 (沒改)
    def _manage_input_and_output_lists(
        self,
        list_of_lists_or_source_folder: Union[str, List[List[str]]],
        output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
        folder_with_segs_from_prev_stage: str = None,
        overwrite: bool = True,  # 是否覆蓋已存在的檔案 (如果 false，則只保留不存在的檔案索引)
        part_id: int = 0,        # 目前是第幾張 GPU
        num_parts: int = 1,      # 總共幾張 GPU
        save_probabilities: bool = False,
    ) -> Tuple[List[List[str]], List[str], List[str]]:
        """
        統一格式:
        Args:
        - list_of_lists_or_source_folder: 可能是"輸入"資料夾路徑或影像檔案清單
        - output_folder_or_list_of_truncated_output_files: 同樣可能是"輸出"資料夾或輸出檔案清單
        - folder_with_segs_from_prev_stage: 前階段分割檔案資料夾

        returns:
        - list_of_lists: 整理後的影像路徑清單
        - output_filenames: 整理後的輸出檔名清單
        - seg_prev_files: 整理後的前階段分割檔案清單
        """
        list_of_lists, output_filenames, seg_prev_files = super()._manage_input_and_output_lists(
            list_of_lists_or_source_folder,
            output_folder_or_list_of_truncated_output_files,
            folder_with_segs_from_prev_stage,
            overwrite,
            part_id,
            num_parts,
            save_probabilities,
        )

        return list_of_lists, output_filenames, seg_prev_files

import argparse
import torch
import os

import argparse
import torch
import os
import multiprocessing
from nnunetv2.utilities.file_path_utilities import get_output_folder, maybe_mkdir_p

def predict_entry_point_multimodal():
    parser = argparse.ArgumentParser(
        description='nnUNetv2 多模態推論入口點 (支援臨床資料，支援 -d -c -tr -p 自動尋找模型資料夾)'
    )
    parser.add_argument('-i', type=str, required=True, help='輸入影像資料夾')
    parser.add_argument('-o', type=str, required=True, help='輸出資料夾')
    parser.add_argument('-d', type=str, required=True, help='數據集名稱或ID')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans', help='plans 名稱')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainerMultimodal', help='訓練器名稱')
    parser.add_argument('-c', type=str, required=True, help='configuration 名稱')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='指定用於預測的模型折數。預設: (0, 1, 2, 3, 4)。可用"all"')
    parser.add_argument('-step_size', type=float, required=False, default=0.5, help='滑動窗口預測步長。預設: 0.5')
    parser.add_argument('--disable_tta', action='store_true', default=False, help='停用鏡像增強')
    parser.add_argument('--save_probabilities', action='store_true', help='導出預測機率圖')
    parser.add_argument('--continue_prediction', action='store_true', help='繼續先前中斷的預測')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth', help='檢查點檔名')
    parser.add_argument('-npp', type=int, required=False, default=3, help='預處理進程數')
    parser.add_argument('-nps', type=int, required=False, default=3, help='分割導出進程數')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None, help='前階段預測資料夾')
    parser.add_argument('-num_parts', type=int, required=False, default=1, help='總分片數')
    parser.add_argument('-part_id', type=int, required=False, default=0, help='當前分片ID')
    parser.add_argument('-device', type=str, default='cuda', required=False, help="推論設備: 'cuda', 'cpu', 'mps'")
    parser.add_argument('--disable_progress_bar', action='store_true', default=False, help='停用進度條')
    parser.add_argument("--clinical_data_dir", type=str, default=None,
                        help="包含臨床數據的目錄路徑")
    parser.add_argument('--verbose', action='store_true', default=False, help='啟用詳細輸出')
    parser.add_argument('--max_cases', type=int, required=False, default=None, 
                        help='限制處理的案例數量（用於測試或調試）')
    parser.add_argument('--no_use_input_cli_data', action='store_true', default=False, 
                        help='不使用輸入的臨床資料')


    print(
        "\n#######################################################################\n"
        "使用nnU-Net時請引用以下論文:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n"
        "#######################################################################\n"
    )

    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    # 自動尋找模型資料夾
    model_folder = get_output_folder(args.d, args.tr, args.p, args.c)
    if not os.path.isdir(args.o):
        maybe_mkdir_p(args.o)

    # 設定設備
    assert args.device in ['cpu', 'cuda', 'mps'], f'-device 必須是 cpu, mps 或 cuda。當前值: {args.device}'
    if args.device == 'cpu':
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    from nnunetv2.inference.nnunet_predictor_multimodal import nnUNetPredictorMultimodal

    # 初始化多模態預測器
    predictor = nnUNetPredictorMultimodal(
        tile_step_size=args.step_size,
        use_gaussian=True,
        use_mirroring=not args.disable_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=args.verbose,
        verbose_preprocessing=args.verbose,
        allow_tqdm=not args.disable_progress_bar,
        clinical_data_dir=args.clinical_data_dir,
        no_use_input_cli_data=args.no_use_input_cli_data
    )

    predictor.initialize_from_trained_model_folder(
        model_folder,
        args.f,
        checkpoint_name=args.chk
    )

    # 執行推論
    predictor.predict_from_files(
        args.i,
        args.o,
        save_probabilities=args.save_probabilities,
        overwrite=not args.continue_prediction,
        num_processes_preprocessing=args.npp,
        num_processes_segmentation_export=args.nps,
        folder_with_segs_from_prev_stage=args.prev_stage_predictions,
        num_parts=args.num_parts,
        part_id=args.part_id,
        max_cases=args.max_cases
    )