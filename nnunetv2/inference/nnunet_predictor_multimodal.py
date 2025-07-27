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
# 我們不再需要原始的 preprocessing_iterator_fromfiles/fromnpy，因為 DataLoaderMultimodal 會將數據組織成所需格式
# from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy, preprocessing_iterator_fromfiles, \
#     preprocessing_iterator_fromnpy

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
import csv
import torch.profiler


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
                 verbose: bool = False,       # 是否輸出詳細日誌
                 verbose_preprocessing: bool = False, # 是否輸出預處理詳細日誌
                 allow_tqdm: bool = True):    # 是否允許顯示進度條
        super().__init__(tile_step_size, use_gaussian, use_mirroring,
                         perform_everything_on_device, device, verbose,
                         verbose_preprocessing, allow_tqdm)
        print("nnUNetPredictorMultimodal 初始化完成。")

    @torch.inference_mode()
    def predict_from_data_iterator(self,
                                data_iterator,
                                save_probabilities: bool = False,
                                num_processes_segmentation_export: int = default_num_processes,
                                profiler=None): # 新增可選參數
        """
        從數據迭代器進行預測（核心預測引擎），支援多模態數據。
        每個迭代器返回的元素必須是包含 'data'、'ofile' 和 'data_properties' 鍵的字典。
        其中 'data' 是一個字典，包含 'data' (影像張量)、'clinical_features' (臨床特徵張量)。
        如果 'ofile' 為 None，結果將直接返回而不是寫入文件。

        Args:
            data_iterator: 產生預處理後數據的迭代器，每個元素應為字典
                `{'data': {'data': image_tensor, 'clinical_features': clinical_tensor, 'has_clinical_data': bool_tensor}, 'ofile': ..., 'data_properties': ...}`
            save_probabilities (bool): 是否儲存預測的機率圖。
            num_processes_segmentation_export (int): 用於分割結果導出的進程數。
            profiler (torch.profiler.Profiler): 可選的 PyTorch 分析器對象，用於性能分析。

        Returns:
            list: 預測結果列表。
        """
        # 將臨床資料的預測結果 進行 argmax 接著轉換為實際標籤
        def clinical_logits_to_labels(clinical_attr_preds):
            # 這裡的 reverse dict 要根據你的實際類別順序調整
            location_reverse = {
                0: 'Missing', 1: 'ascending', 2: 'transverse', 3: 'descending',
                4: 'sigmoid', 5: 'rectal', 6: 'rectosigmoid'
            }
            t_stage_reverse = {
                0: 'T0', 1: 'T1', 2: 'T2', 3: 'T3', 4: 'T4a', 5: 'T4b', 6: 'Tx'
            }
            n_stage_reverse = {
                0: 'N0', 1: 'N1', 2: 'N2', 3: 'Nx', 4: 'Missing'
            }
            m_stage_reverse = {
                0: 'M0', 1: 'M1', 2: 'Mx', 3: 'Missing'
            }
            loc_idx = int(torch.argmax(clinical_attr_preds['location']).item())
            t_idx = int(torch.argmax(clinical_attr_preds['t_stage']).item())
            n_idx = int(torch.argmax(clinical_attr_preds['n_stage']).item())
            m_idx = int(torch.argmax(clinical_attr_preds['m_stage']).item())
            missing_flags = (torch.sigmoid(clinical_attr_preds['missing_flags']) > 0.5).int().tolist()
            return {
                "Location": location_reverse[loc_idx],
                "T_stage": t_stage_reverse[t_idx],
                "N_stage": n_stage_reverse[n_idx],
                "M_stage": m_stage_reverse[m_idx],
                "Missing_flags": missing_flags
            }


        # 保存臨床預測結果 txt
        def save_clinical_prediction_txt(ofile, clinical_attr_preds):
            txt_path = ofile + ".txt"
            labels = clinical_logits_to_labels(clinical_attr_preds)
            with open(txt_path, "w") as f:
                for k, v in labels.items():
                    f.write(f"{k}: {v}\n")

        # 保存臨床預測結果 csv
        def save_clinical_prediction_csv(csv_path, all_clinical_results):
            if not all_clinical_results:
                return
            fieldnames = list(all_clinical_results[0].keys())
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in all_clinical_results:
                    writer.writerow(row)

        # 使用 'spawn' 上下文創建進程池，以避免 CUDA 相關問題
        all_clinical_results = []
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool] # 獲取工作進程列表
            r = [] # 用於儲存異步操作的結果物件

            # 遍歷預處理後的數據
            for i, preprocessed_item in enumerate(data_iterator):
            # [DEBUG]  只推論前三筆檔案!!!!!
            #     if i >= 3:  # 只處理前兩個
            #         break

                # --- 性能分析：在每次處理一個案例前，調用 prof.step() ---
                # 注意：profiler.step() 的調用會根據 schedule 來決定是否記錄。
                if profiler is not None:
                    profiler.step()
                # --- 性能分析結束 ---

                # 從預處理結果中提取影像數據和臨床特徵
                # preprocessed_item['data'] 現在是一個字典
                image_data = preprocessed_item['data']['data']
                clinical_features = preprocessed_item['data']['clinical_features']
                has_clinical_data = preprocessed_item['data']['has_clinical_data'] # 標記是否有臨床資料

                ofile = preprocessed_item['ofile'] # 輸出文件名 (如果存在)
                properties = preprocessed_item['data_properties'] # 原始影像屬性

                if ofile is not None:
                    print(f'\n正在預測 {os.path.basename(ofile)}:')
                else:
                    print(f'\n正在預測形狀為 {image_data.shape} 的影像:')
                print(f'全程在設備執行: {self.perform_everything_on_device}')

                # 流量控制：避免 GPU 預測過快導致導出進程阻塞
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=4)
                while not proceed:
                    sleep(0.05) # 等待 100 毫秒
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=4)
                
                # 呼叫修改後的預測邏輯，同時傳入影像數據和臨床特徵
                # 它會返回分割的 logits 和臨床屬性預測的 logits
                prediction_seg_logits, clinical_attr_preds = self.predict_logits_and_attributes_from_preprocessed_data(
                    image_data, clinical_features
                )
                
                #  將分割預測結果的匯出工作放入背景行程池中執行 (非同步)
                prediction_seg_logits = prediction_seg_logits.cpu().detach().numpy()

                # 保存臨床預測 txt
                if ofile is not None:
                    save_clinical_prediction_txt(ofile, clinical_attr_preds)

                # 收集到 all_clinical_results
                case_id = os.path.basename(ofile) if ofile is not None else f"case_{i}"
                labels = clinical_logits_to_labels(clinical_attr_preds)
                result_row = {"Case_Index": case_id}
                result_row.update(labels)
                all_clinical_results.append(result_row)

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
                              properties, save_probabilities, clinical_attr_preds, has_clinical_data),) # 傳遞臨床預測結果及標誌
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
    def predict_logits_and_attributes_from_preprocessed_data(self, data: torch.Tensor, clinical_features: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        重要！如果是級聯模型，前階段分割必須已以 one-hot 編碼形式堆疊在影像頂部！
        此方法同時返回分割的 logits 和臨床屬性預測的 logits。
        
        Args:
            data (torch.Tensor): 預處理後的影像數據 (4D 張量)。
            clinical_features (torch.Tensor): 預處理後的臨床特徵數據 (批次, prompt_dim)。
            
        Returns:
            Tuple[torch.Tensor, dict]: 包含分割 logits 和臨床屬性 logits 字典的元組。
        """
        n_threads = torch.get_num_threads()
        torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)

        prediction_seg = None
        # 初始化臨床屬性預測的累加器
        prediction_attrs = {k: None for k in ['location', 't_stage', 'n_stage', 'm_stage', 'missing_flags']}

        # 模型集成：遍歷所有參數集（多折交叉驗證）
        for params in self.list_of_parameters:
            # 加載模型參數
            if not isinstance(self.network, OptimizedModule):
                self.network.load_state_dict(params)
            else:
                self.network._orig_mod.load_state_dict(params) # 處理編譯後模型

            # 呼叫修改後的 predict_sliding_window_return_logits，傳入影像和臨床特徵
            seg_logits_single_model, attr_logits_single_model = self.predict_sliding_window_return_logits(data, clinical_features)

            # 累加預測結果
            if prediction_seg is None:
                prediction_seg = seg_logits_single_model.to('cpu')
                for k, v in attr_logits_single_model.items():
                    prediction_attrs[k] = v.to('cpu')
            else:
                prediction_seg += seg_logits_single_model.to('cpu')
                for k, v in attr_logits_single_model.items():
                    prediction_attrs[k] += v.to('cpu')
        
        # 計算平均預測 (如果有多個模型)
        if len(self.list_of_parameters) > 1:
            prediction_seg /= len(self.list_of_parameters)
            for k in prediction_attrs.keys():
                if prediction_attrs[k] is not None: # 確保屬性存在
                    prediction_attrs[k] /= len(self.list_of_parameters)

        if self.verbose:
            print('預測完成')
        torch.set_num_threads(n_threads) # 恢復線程設置
        return prediction_seg, prediction_attrs

    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor, clinical_features: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        執行鏡像增強預測（內部方法），同時考慮臨床特徵。
        根據配置決定是否使用測試時數據增強。臨床特徵不進行鏡像操作。
        
        Args:
            x (torch.Tensor): 輸入影像批次。
            clinical_features (torch.Tensor): 輸入臨床特徵批次。
            
        Returns:
            Tuple[torch.Tensor, dict]: 鏡像增強後的分割 logits 和臨床屬性 logits 字典。
        """
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        
        # 初始預測 (使用原始影像和臨床特徵)
        seg_prediction, attr_prediction = self.network(x, clinical_features)
        # assert isinstance(seg_prediction, torch.Tensor), f"seg_prediction type: {type(seg_prediction)}"

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
                m_seg_pred, m_attr_pred = self.network(mirrored_x, clinical_features)

                # 分割結果需要反向鏡像後累加
                seg_prediction += torch.flip(m_seg_pred, axes)

                # 臨床屬性結果不需要反向鏡像，直接累加
                for k in attr_prediction.keys():
                    if attr_prediction[k] is not None: # 確保屬性存在
                        attr_prediction[k] += m_attr_pred[k]

            # 計算加權平均
            seg_prediction /= (len(axes_combinations) + 1)
            for k in attr_prediction.keys():
                if attr_prediction[k] is not None: # 確保屬性存在
                    attr_prediction[k] /= (len(axes_combinations) + 1)

        return seg_prediction, attr_prediction

    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits(self,
                                                       data: torch.Tensor,
                                                       clinical_features: torch.Tensor, # 新增臨床特徵輸入
                                                       slicers,
                                                       do_on_device: bool = True):
        """
        滑動窗口預測核心實現（內部方法），支援多模態輸入。
        使用生產者-消費者模式高效處理大影像。
        
        Args:
            data (torch.Tensor): 填充後的影像數據。
            clinical_features (torch.Tensor): 該案例的臨床特徵數據。
            slicers: 滑動窗口切片器列表。
            do_on_device (bool): 是否在設備 (GPU) 上執行所有操作。
            
        Returns:
            Tuple[torch.Tensor, dict]: 分割 logits 和臨床屬性 logits 字典。
        """
        predicted_logits_seg = n_predictions = prediction_seg = gaussian = workon_image = None
        predicted_logits_attrs = {k: None for k in ['location', 't_stage', 'n_stage', 'm_stage', 'missing_flags']}
        
        results_device = self.device if do_on_device else torch.device('cpu') # 結果儲存設備

        def producer(d, slh, q, clinical_f):
            """生產者函數：將影像切片和臨床特徵放入隊列"""
            for s in slh:
                # 將影像切片和臨床特徵複製到 GPU 並放入隊列
                q.put((
                    torch.clone(d[s][None], memory_format=torch.contiguous_format).to(self.device),
                    clinical_f.to(self.device), # 臨床特徵是對整個 case 有效，所以每個 patch 都傳遞
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
            t = Thread(target=producer, args=(data, slicers, queue, clinical_features))
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

            # 為臨床屬性預測分配空間 (由於臨床屬性預測通常是針對整個案例而不是單個 patch，
            # 我們只需在第一次有效預測時獲取其結果)
            # 這裡假設 `MyModel` 會在 `forward` 中根據 `patch_size` 確定輸出尺寸
            # 為了通用性，我們從 `network.init_kwargs` 中獲取類別數量
            # 注意：這裡的 `network` 可能是 DDP 的 `module` 或 `_orig_mod`
            if isinstance(self.network, (nn.parallel.DistributedDataParallel, OptimizedModule)):
                model_base = self.network.module if isinstance(self.network, DistributedDataParallel) else self.network._orig_mod
                attr_init_kwargs = model_base.init_kwargs
            else:
                attr_init_kwargs = self.network.init_kwargs # 假設 MyModel 有 init_kwargs 屬性

            attr_output_shapes = {
                'location': (attr_init_kwargs.get('location_classes', 7),),
                't_stage': (attr_init_kwargs.get('t_stage_classes', 7),),
                'n_stage': (attr_init_kwargs.get('n_stage_classes', 5),),
                'm_stage': (attr_init_kwargs.get('m_stage_classes', 4),),
                'missing_flags': (attr_init_kwargs.get('missing_flags_dim', 4),)
            }
            # 初始化臨床屬性 logits 字典，用於存儲第一次預測結果 (或平均結果)
            for k, shape in attr_output_shapes.items():
                predicted_logits_attrs[k] = torch.zeros(shape, dtype=torch.float32, device=results_device) # 使用 float32 保持精度

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
            
            # 執行推論
            with tqdm(desc=None, total=len(slicers), disable=not self.allow_tqdm) as pbar:
                first_attr_pred_processed = False # 標記是否已處理過臨床屬性預測
                while True:
                    item = queue.get() # 從隊列中獲取數據
                    if item == 'end': # 如果遇到結束標記
                        queue.task_done()
                        break
                    
                    workon_image, workon_clinical_features, sl = item # 解包數據：影像切片、臨床特徵、切片器
                    
                    # 執行預測，同時傳入影像和臨床特徵
                    # 返回的 prediction_seg 應為 (1, num_classes, D, H, W)
                    # 返回的 attr_prediction 應為 {'location': (1, num_loc_classes), ...}
                    prediction_seg_patch, attr_prediction_patch = self._internal_maybe_mirror_and_predict(
                        workon_image, workon_clinical_features
                    )
                    prediction_seg_patch = prediction_seg_patch.to(results_device) # 移除批次維度

                    # 移除 batch 維度
                    if isinstance(prediction_seg_patch, list):
                        prediction_seg_patch = prediction_seg_patch[0]
                    if prediction_seg_patch.shape[0] == 1:
                        prediction_seg_patch = prediction_seg_patch.squeeze(0)


                    # 應用高斯加權到分割結果
                    if self.use_gaussian:
                        prediction_seg_patch *= gaussian
                    
                    # 檢查兩者的shape
                    # print(f'predicted_logits[sl] shape: {predicted_logits_seg[sl].shape}, prediction shape: {prediction_seg_patch.shape}')
                    # breakpoint()
                    # predicted_logits[sl] shape: torch.Size([2, 112, 160, 128]), prediction shape: torch.Size([1, 2, 112, 160, 128])

                    # 累加分割結果和預測次數
                    predicted_logits_seg[sl] += prediction_seg_patch
                    n_predictions[sl[1:]] += gaussian

                    # 處理臨床屬性結果：由於臨床特徵針對整個案例，其預測結果不應隨每個 patch 累加。
                    # 我們只在第一次有效處理 patch 時記錄其臨床屬性預測，因為它代表了整個案例的預測。
                    if not first_attr_pred_processed:
                        for k, v in attr_prediction_patch.items():
                            if predicted_logits_attrs[k] is not None:
                                predicted_logits_attrs[k] = v.to(results_device) # 僅記錄第一個 patch 的結果 (移除批次維度)
                        first_attr_pred_processed = True
                        
                    queue.task_done()
                    pbar.update() # 更新進度條

            # # # 執行推論
            # batch_size = 4
            # with tqdm(desc=None, total=len(slicers), disable=not self.allow_tqdm) as pbar:
            #     first_attr_pred_processed = False
            #     patch_list, clinical_list, slice_list = [], [], []
            #     while True:
            #         item = queue.get()
            #         if item == 'end':
            #             # 處理最後不足 batch_size 的 patch
            #             if patch_list:
            #                 batch_tensor = torch.cat(patch_list, dim=0)
            #                 batch_clinical = torch.cat(clinical_list, dim=0)
            #                 seg_preds, attr_preds = self._internal_maybe_mirror_and_predict(batch_tensor, batch_clinical)
            #                 for idx, sl in enumerate(slice_list):
            #                     seg_patch = seg_preds[idx].to(results_device)
            #                     if self.use_gaussian:
            #                         seg_patch *= gaussian
            #                     predicted_logits_seg[sl] += seg_patch
            #                     n_predictions[sl[1:]] += gaussian
            #                     if not first_attr_pred_processed:
            #                         for k, v in attr_preds.items():
            #                             if predicted_logits_attrs[k] is not None:
            #                                 predicted_logits_attrs[k] = v[idx].to(results_device)
            #                         first_attr_pred_processed = True
            #                 pbar.update(len(patch_list))
            #             queue.task_done()
            #             break

            #         workon_image, workon_clinical_features, sl = item
            #         patch_list.append(workon_image)
            #         clinical_list.append(workon_clinical_features)
            #         slice_list.append(sl)

            #         if len(patch_list) == batch_size:
            #             batch_tensor = torch.cat(patch_list, dim=0)
            #             batch_clinical = torch.cat(clinical_list, dim=0)
            #             seg_preds, attr_preds = self._internal_maybe_mirror_and_predict(batch_tensor, batch_clinical)
            #             for idx, sl in enumerate(slice_list):
            #                 seg_patch = seg_preds[idx].to(results_device)
            #                 if self.use_gaussian:
            #                     seg_patch *= gaussian
            #                 predicted_logits_seg[sl] += seg_patch
            #                 n_predictions[sl[1:]] += gaussian
            #                 if not first_attr_pred_processed:
            #                     for k, v in attr_preds.items():
            #                         if predicted_logits_attrs[k] is not None:
            #                             predicted_logits_attrs[k] = v[idx].to(results_device)
            #                     first_attr_pred_processed = True
            #             patch_list, clinical_list, slice_list = [], [], []
            #             pbar.update(batch_size) # 更新進度條
            #         queue.task_done()

            queue.join() # 等待隊列處理完成
            
            # 計算分割結果的加權平均
            torch.div(predicted_logits_seg, n_predictions, out=predicted_logits_seg)

            # 檢查分割預測結果中是否存在無窮值 (通常指示數值不穩定)
            if torch.any(torch.isinf(predicted_logits_seg)):
                raise RuntimeError(
                    '在分割預測數組中檢測到inf值！如果持續出現此問題，'
                    '請降低 compute_gaussian 中的 value_scaling_factor 或將 predicted_logits 的 dtype 提高為 fp32'
                )

        except Exception as e:
            # 異常時清理資源
            del predicted_logits_seg, n_predictions, prediction_seg, gaussian, workon_image
            del predicted_logits_attrs
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        
        # 返回分割結果和臨床屬性結果
        return predicted_logits_seg, predicted_logits_attrs

    @torch.inference_mode()
    def predict_sliding_window_return_logits(self, input_image: torch.Tensor, clinical_features: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        滑動窗口預測主入口，返回分割 logits 和臨床屬性 logits。
        此方法會處理影像的填充、滑窗切片，並將影像和臨床特徵傳遞給底層模型進行預測。
        
        Args:
            input_image (torch.Tensor): 原始輸入影像 (4D 張量，(批次, 通道, D, H, W))。
            clinical_features (torch.Tensor): 該案例的臨床特徵數據 (2D 張量，(批次, prompt_dim))。
            
        Returns:
            Tuple[torch.Tensor, dict]: 包含分割 logits 和臨床屬性 logits 字典的元組。
        """
        assert isinstance(input_image, torch.Tensor), "輸入影像必須是 torch.Tensor 類型。"
        assert isinstance(clinical_features, torch.Tensor), "輸入臨床特徵必須是 torch.Tensor 類型。"

        self.network = self.network.to(self.device) # 確保網路在正確設備
        self.network.eval() # 設定為評估模式
        empty_cache(self.device) # 清理設備緩存

        # 自動混合精度設置（僅 CUDA 設備）
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            assert input_image.ndim == 4, '輸入影像必須是4D張量（批次, 通道, x, y, z）。'
            # 臨床特徵假定為 (批次, prompt_dim)

            if self.verbose:
                print(f'輸入影像形狀: {input_image.shape}')
                print(f'輸入臨床特徵形狀: {clinical_features.shape}')
                print(f"步長: {self.tile_step_size}")
                print(f"鏡像軸: {self.allowed_mirroring_axes if self.use_mirroring else '無'}")

            # 邊界填充處理：當影像尺寸小於 patch_size 時，需要對影像進行填充
            data_padded, slicer_revert_padding = pad_nd_image(
                input_image,
                self.configuration_manager.patch_size, # 目標尺寸 (patch_size)
                'constant', # 填充類型為常數
                {'value': 0}, # 填充值為 0
                True, # 返回切片器用於恢復原始尺寸
                None # 無需指定邊框
            )

            # 獲取滑動窗口切片器：這些切片器定義了每個 patch 的位置
            slicers = self._internal_get_sliding_window_slicers(data_padded.shape[1:])

            # 執行預測：嘗試在設備上執行所有操作，如果記憶體不足則回退到 CPU
            if self.perform_everything_on_device and self.device.type == 'cuda':
                try:
                    predicted_logits_seg, predicted_logits_attrs = self._internal_predict_sliding_window_return_logits(
                        data_padded, clinical_features, slicers, self.perform_everything_on_device
                    )
                except RuntimeError: # 通常因記憶體不足 (OOM) 引起
                    print('設備預測失敗（可能因內存不足），將結果數組移至CPU')
                    empty_cache(self.device) # 清理 GPU 緩存
                    predicted_logits_seg, predicted_logits_attrs = self._internal_predict_sliding_window_return_logits(
                        data_padded, clinical_features, slicers, False # 在 CPU 上執行
                    )
            else:
                # 不在設備上執行或設備為 CPU/MPS
                predicted_logits_seg, predicted_logits_attrs = self._internal_predict_sliding_window_return_logits(
                    data_padded, clinical_features, slicers, self.perform_everything_on_device
                )

            empty_cache(self.device) # 清理設備緩存

            # 移除填充區域：將分割 logits 恢復到原始影像尺寸
            predicted_logits_seg = predicted_logits_seg[(slice(None), *slicer_revert_padding[1:])]

            # 返回分割 logits 和臨床屬性 logits
            return predicted_logits_seg, predicted_logits_attrs

    # def predict_single_npy_array(self,
    #                              input_image: np.ndarray,
    #                              image_properties: dict,
    #                              segmentation_previous_stage: np.ndarray = None,
    #                              clinical_features: np.ndarray = None, # 新增臨床特徵
    #                              output_file_truncated: str = None,
    #                              save_or_return_probabilities: bool = False):
    #     """
    #     警告：速度較慢！僅在無法批量處理時使用此方法。
    #     此方法適用於對單個 NumPy 陣列進行預測，現在也接受臨床特徵。
    #     它會手動處理影像的預處理和預測流程。
        
    #     Args:
    #         input_image (np.ndarray): 輸入影像的 NumPy 陣列。
    #         image_properties (dict): 影像的屬性字典。
    #         segmentation_previous_stage (np.ndarray, optional): 前一階段的分割結果 (用於級聯模型)。
    #         clinical_features (np.ndarray, optional): 該案例的臨床特徵 NumPy 陣列。
    #         output_file_truncated (str, optional): 輸出文件的路徑前綴 (不含擴展名)。
    #         save_or_return_probabilities (bool): 是否保存或返回分割的機率圖。
            
    #     Returns:
    #         Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: 根據 `save_or_return_probabilities` 返回分割圖或分割圖和機率圖。
    #         對於多模態，額外返回臨床屬性 logits 字典。
    #     """
    #     # 手動執行預處理邏輯 (簡化，僅為示例)
    #     # 影像數據從 NumPy 陣列轉換為 Torch 張量
    #     input_image_tensor = torch.from_numpy(input_image).float()
        
    #     # 臨床特徵從 NumPy 陣列轉換為 Torch 張量
    #     clinical_features_tensor = None
    #     if clinical_features is not None:
    #         clinical_features_tensor = torch.from_numpy(clinical_features).float()
    #         # 確保 clinical_features_tensor 有批次維度，例如 (1, prompt_dim)
    #         if clinical_features_tensor.ndim == 1:
    #             clinical_features_tensor = clinical_features_tensor[None] # 添加批次維度
    #     else:
    #         # 如果沒有臨床資料 (例如 Stage 1 數據)，創建一個填充零的張量
    #         # 這裡需要從 configuration_manager 獲取 MyModel 的 prompt_dim
    #         # 注意：這裡的 self.configuration_manager.network_arch_init_kwargs 是 Trainer 初始化時設定的
    #         # 而在 Predictor 中，network_arch_init_kwargs 可能不包含 MyModel 的所有自訂參數
    #         # 穩健做法是從 self.network (或其 module/orig_mod) 中獲取 MyModel 的 prompt_dim
    #         # 為簡潔起見，這裡硬編碼為 17 (MyModel 預設值)
    #         prompt_dim = self.network.init_kwargs.get('prompt_dim', 17) if not isinstance(self.network, (nn.parallel.DistributedDataParallel, OptimizedModule)) \
    #             else (self.network.module.init_kwargs.get('prompt_dim', 17) if isinstance(self.network, DistributedDataParallel) else self.network._orig_mod.init_kwargs.get('prompt_dim', 17))
    #         clinical_features_tensor = torch.zeros((1, prompt_dim), dtype=torch.float32)

    #     if self.verbose:
    #         print('正在進行預測')
        
    #     # 呼叫修改後的 predict_logits_and_attributes_from_preprocessed_data
    #     # 它將返回分割 logits 和臨床屬性 logits
    #     predicted_logits_seg, predicted_logits_attrs = self.predict_logits_and_attributes_from_preprocessed_data(
    #         input_image_tensor, clinical_features_tensor
    #     )
    #     predicted_logits_seg = predicted_logits_seg.cpu() # 將分割結果移動到 CPU

    #     if self.verbose:
    #         print('重採樣到原始形狀')

    #     # 處理輸出：保存文件或直接返回結果
    #     if output_file_truncated is not None:
    #         # export_prediction_from_logits 處理分割結果的保存
    #         export_prediction_from_logits(
    #             predicted_logits_seg,
    #             image_properties,
    #             self.configuration_manager,
    #             self.plans_manager,
    #             self.dataset_json,
    #             output_file_truncated,
    #             save_or_return_probabilities
    #         )
    #         # 臨床屬性結果不保存為文件，但可以在這裡選擇返回
    #         return {
    #             'segmentation_file': output_file_truncated + self.dataset_json['file_ending'],
    #             'clinical_attributes_logits': predicted_logits_attrs # 返回臨床屬性 logits (Torch 張量)
    #         }
    #     else:
    #         # convert_predicted_logits_to_segmentation_with_correct_shape 處理分割結果的返回
    #         segmentation_result, probabilities_result = convert_predicted_logits_to_segmentation_with_correct_shape(
    #             predicted_logits_seg,
    #             self.plans_manager,
    #             self.configuration_manager,
    #             self.label_manager,
    #             image_properties,
    #             return_probabilities=save_or_return_probabilities
    #         )

    #         # 根據 save_or_return_probabilities 返回不同的元組
    #         if save_or_return_probabilities:
    #             return segmentation_result, probabilities_result, predicted_logits_attrs
    #         else:
    #             return segmentation_result, predicted_logits_attrs

# 如果需要從命令行直接調用此模組進行預測，可以擴展 predict_entry_point 等函數
# 這裡為簡潔起見，不提供命令行入口點的完整修改，因為它通常在 nnunetv2_predict 腳本中處理。


# ------------------------------------------------
    # 1. 內部 iterator 統一換成多模態版本
    # ------------------------------------------------
    def _internal_get_data_iterator_from_lists_of_filenames(
        self,
        input_list_of_lists: List[List[str]],
        seg_from_prev_stage_files: Union[List[str], None],
        output_filenames_truncated: Union[List[str], None],
        num_processes: int,
        clinical_data_folder: Union[str, None] = None  # 新增
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
            clinical_data_folder=clinical_data_folder
        )

    # ------------------------------------------------
    # 2. predict_from_files 加上 clinical_data_folder
    # ------------------------------------------------
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
        clinical_data_folder: Union[str, None] = None  # 新增
    ):
        """
        執行流程:
            先呼叫 _manage_input_and_output_lists 整理輸入/輸出清單
            再呼叫 _internal_get_data_iterator_from_lists_of_filenames 建立 data_iterator
            最後呼叫 self.predict_from_data_iterator(...)
        """
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
        if len(list_of_lists) == 0:
            print(f'進程 {part_id}/{num_parts} 無需處理新案例，跳過')
            return []

        # 2. 建立新的多模態 iterator
        data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(
            list_of_lists,
            seg_prev_files,
            output_filenames,
            num_processes_preprocessing,
            clinical_data_folder=clinical_data_folder
        )

        # 3. 交給 predict_from_data_iterator 完成推論
        return self.predict_from_data_iterator(
            data_iterator,
            save_probabilities,
            num_processes_segmentation_export
        )

    # ------------------------------------------------
    # 3. 順序版 predict_from_files_sequential
    # ------------------------------------------------
    def predict_from_files_sequential(
        self,
        list_of_lists_or_source_folder: Union[str, List[List[str]]],
        output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
        save_probabilities: bool = False,
        overwrite: bool = True,
        folder_with_segs_from_prev_stage: str = None,
        clinical_data_folder: Union[str, None] = None  # 新增
    ):
        """
        單線程版本：同樣支援 clinical_data_folder
        """
        # 1. 整理清單（固定 part_id=0, num_parts=1）
        list_of_lists, output_filenames, seg_prev_files = self._manage_input_and_output_lists(
            list_of_lists_or_source_folder,
            output_folder_or_list_of_truncated_output_files,
            folder_with_segs_from_prev_stage,
            overwrite,
            0, 1, save_probabilities
        )
        if len(list_of_lists) == 0:
            print('無需處理新案例，跳過')
            return []

        # 2. 建立 iterator（單線程）
        iterator = preprocessing_iterator_fromfiles_multimodal(
            list_of_lists,
            seg_prev_files,
            output_filenames,
            self.plans_manager,
            self.dataset_json,
            self.configuration_manager,
            num_processes=1,
            pin_memory=False,
            verbose=self.verbose,
            clinical_data_folder=clinical_data_folder
        )

        # 3. 推論
        return self.predict_from_data_iterator(iterator, save_probabilities, 1)

    # ------------------------------------------------
    # 4. predict_from_list_of_npy_arrays 增加臨床資料
    # ------------------------------------------------
    def predict_from_list_of_npy_arrays(
        self,
        image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
        segs_from_prev_stage_or_list_of_segs_from_prev_stage: Union[None, np.ndarray, List[np.ndarray]],
        properties_or_list_of_properties: Union[dict, List[dict]],
        truncated_ofname: Union[str, List[str], None],
        num_processes: int = 3,
        save_probabilities: bool = False,
        num_processes_segmentation_export: int = 3,
        clinical_data_folder: Union[str, None] = None  # 新增
    ):
        """
        直接從 numpy 陣列推論，支援臨床資料
        """
        # 1. 統一變成 list 格式
        images = [image_or_list_of_images] if not isinstance(image_or_list_of_images, list) else image_or_list_of_images
        segs = [segs_from_prev_stage_or_list_of_segs_from_prev_stage] if isinstance(segs_from_prev_stage_or_list_of_segs_from_prev_stage, np.ndarray) else segs_from_prev_stage_or_list_of_segs_from_prev_stage
        props = [properties_or_list_of_properties] if isinstance(properties_or_list_of_properties, dict) else properties_or_list_of_properties
        ofnames = [truncated_ofname] if isinstance(truncated_ofname, str) else truncated_ofname

        # 2. 建立 iterator（從 numpy）
        iterator = preprocessing_iterator_fromfiles_multimodal(
            images,  # 雖然叫 fromfiles，但內部其實支援 numpy list
            segs,
            ofnames,
            self.plans_manager,
            self.dataset_json,
            self.configuration_manager,
            num_processes,
            self.device.type == 'cuda',
            self.verbose_preprocessing,
            clinical_data_folder=clinical_data_folder
        )

        # 3. 推論
        return self.predict_from_data_iterator(
            iterator,
            save_probabilities,
            num_processes_segmentation_export
        )

    # ------------------------------------------------
    # 5. predict_single_npy_array 也補上臨床資料
    # ------------------------------------------------
    def predict_single_npy_array(
        self,
        input_image: np.ndarray,
        image_properties: dict,
        segmentation_previous_stage: np.ndarray = None,
        clinical_features: np.ndarray = None,   # 直接給 numpy 陣列
        output_file_truncated: str = None,
        save_or_return_probabilities: bool = False
    ):
        """
        單張 numpy 推論：把單筆資料包成 list 後交給上一個方法
        注意：此處 clinical_features 直接給 numpy，不從硬碟讀
        """
        # 1. 把單張包成 list
        images = [input_image]
        segs = [segmentation_previous_stage]
        props = [image_properties]
        ofnames = [output_file_truncated]

        # 2. 建立 iterator（num_processes=1）
        iterator = preprocessing_iterator_fromfiles_multimodal(
            images,
            segs,
            ofnames,
            self.plans_manager,
            self.dataset_json,
            self.configuration_manager,
            num_processes=1,
            pin_memory=False,
            verbose=self.verbose,
            clinical_data_folder=None  # 不從資料夾讀，直接給 numpy
        )

        # 3. 推論
        results = self.predict_from_data_iterator(iterator, save_or_return_probabilities, 1)

        # 4. 回傳單筆結果
        return results[0] if results else None
        
    # 訓練最後的驗證不會用到 真的推論時才會用到
    def initialize_from_trained_model_folder(
        self,
        model_training_output_dir: str,
        use_folds: Union[Tuple[Union[int, str]], None],
        checkpoint_name: str = 'checkpoint_final.pth'
    ):
        """
        從訓練好的模型目錄初始化多模態預測器。
        與父類唯一差別：讀取額外維度 prompt_dim 並建立 MyModel。
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

        parameters = []
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
                # ★★ 讀取 prompt_dim 等額外參數 ★★
                extra_kwargs = checkpoint.get('init_kwargs', {})
                prompt_dim = extra_kwargs.get('prompt_dim', 17)
                location_classes = extra_kwargs.get('location_classes', 7)
                t_stage_classes = extra_kwargs.get('t_stage_classes', 7)
                n_stage_classes = extra_kwargs.get('n_stage_classes', 5)
                m_stage_classes = extra_kwargs.get('m_stage_classes', 4)
                missing_flags_dim = extra_kwargs.get('missing_flags_dim', 4)
            parameters.append(checkpoint['network_weights'])

        # 3. 建立計劃與配置管理器
        configuration_manager = plans_manager.get_configuration(configuration_name)
        num_input_channels_img = determine_num_input_channels(
            plans_manager, configuration_manager, dataset_json
        )

        # 4. 動態載入訓練器並建立網路
        trainer_class = recursive_find_python_class(
            join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
            trainer_name,
            'nnunetv2.training.nnUNetTrainer'
        )
        if trainer_class is None:
            raise RuntimeError(f'找不到訓練器類別 {trainer_name}')

        # my_model_init_kwargs = {
        #     'input_channels': self.num_input_channels,
        #     'num_classes': self.label_manager.num_segmentation_heads,
        #     'deep_supervision': self.enable_deep_supervision,
        #     'prompt_dim': 17,
        #     'location_classes': 7,
        #     't_stage_classes': 7,
        #     'n_stage_classes': 5,
        #     'm_stage_classes': 4,
        #     'missing_flags_dim': 4
        # }
        my_model_init_kwargs = {
            'input_channels': num_input_channels_img,
            'num_classes': plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            'deep_supervision': False,
            'prompt_dim': prompt_dim,
            'location_classes': location_classes,
            't_stage_classes': t_stage_classes,
            'n_stage_classes': n_stage_classes,
            'm_stage_classes': m_stage_classes,
            'missing_flags_dim': missing_flags_dim
        }
        network = trainer_class.build_network_architecture(MyModel, my_model_init_kwargs).to(self.device)

        # 5. 建立 MyModel 並載入權重
        model = MyModel(
            input_channels=num_input_channels_img,
            num_classes=plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            deep_supervision=False,
            prompt_dim=prompt_dim,
            location_classes=location_classes,
            t_stage_classes=t_stage_classes,
            n_stage_classes=n_stage_classes,
            m_stage_classes=m_stage_classes,
            missing_flags_dim=missing_flags_dim
        )

        network.load_state_dict(parameters[0])
        self.network = network

        # 6. 儲存其餘必要屬性
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)

        # 7. torch.compile 加速（若環境變數開啟）
        if ('nnUNet_compile' in os.environ and
                os.environ['nnUNet_compile'].lower() in ('true', '1', 't') and
                not isinstance(self.network, OptimizedModule)):
            print('啟用 torch.compile 加速')
            self.network = torch.compile(self.network)


    def _manage_input_and_output_lists(
        self,
        list_of_lists_or_source_folder: Union[str, List[List[str]]],
        output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
        folder_with_segs_from_prev_stage: str = None,
        overwrite: bool = True,
        part_id: int = 0,
        num_parts: int = 1,
        save_probabilities: bool = False,
    ) -> Tuple[List[List[str]], List[str], List[str]]:
        """
        與父類同名，唯一差別：計算「總輸入通道數」時加入臨床 prompt 維度。
        """
        # 1. 先呼叫父類完成所有檔案列表整理
        list_of_lists, output_filenames, seg_prev_files = super()._manage_input_and_output_lists(
            list_of_lists_or_source_folder,
            output_folder_or_list_of_truncated_output_files,
            folder_with_segs_from_prev_stage,
            overwrite,
            part_id,
            num_parts,
            save_probabilities,
        )

        # 2. 重新計算總輸入通道數（影像 + seg-prev one-hot + prompt）
        num_img_channels = determine_num_input_channels(
            self.plans_manager, self.configuration_manager, self.dataset_json
        )
        num_seg_prev_channels = 0
        if folder_with_segs_from_prev_stage is not None:
            num_seg_prev_channels = len(
                self.label_manager.foreground_labels
            )  # one-hot 後的通道數
        prompt_dim = getattr(self.network, 'prompt_dim', 17)  # 從模型抓維度
        total_input_channels = num_img_channels + num_seg_prev_channels + prompt_dim

        # 3. 印出供確認
        if self.verbose:
            print(
                f"[多模態] 影像通道={num_img_channels}, "
                f"前階段分割通道={num_seg_prev_channels}, "
                f"臨床 prompt 維度={prompt_dim}, "
                f"總輸入通道={total_input_channels}"
            )

        return list_of_lists, output_filenames, seg_prev_files


    # 覆蓋父類同名方法
    def predict_single_npy_array(
        self,
        input_image: np.ndarray,
        image_properties: dict,
        segmentation_previous_stage: np.ndarray = None,
        clinical_features: np.ndarray = None,
        output_file_truncated: str = None,
        save_or_return_probabilities: bool = False,
    ):
        """
        單張 numpy 推論：直接把臨床 prompt 一起傳給推論函數
        """
        # 1. 轉 tensor
        image_tensor = torch.from_numpy(input_image).float()
        prompt_dim = getattr(self.network, 'prompt_dim', 17)
        if clinical_features is None:
            clinical_tensor = torch.zeros((1, prompt_dim), dtype=torch.float32)
        else:
            clinical_tensor = torch.from_numpy(clinical_features).float()
            if clinical_tensor.ndim == 1:
                clinical_tensor = clinical_tensor.unsqueeze(0)  # 加 batch

        # 2. 推論
        seg_logits, attr_logits = self.predict_logits_and_attributes_from_preprocessed_data(
            image_tensor, clinical_tensor
        )
        seg_logits = seg_logits.cpu()

        # 3. 後處理
        if output_file_truncated is not None:
            export_prediction_from_logits(
                seg_logits,
                image_properties,
                self.configuration_manager,
                self.plans_manager,
                self.dataset_json,
                output_file_truncated,
                save_or_return_probabilities,
            )
            return {
                'segmentation_file': output_file_truncated + self.dataset_json['file_ending'],
                'clinical_attributes_logits': attr_logits,
            }
        else:
            seg, prob = convert_predicted_logits_to_segmentation_with_correct_shape(
                seg_logits,
                self.plans_manager,
                self.configuration_manager,
                self.label_manager,
                image_properties,
                return_probabilities=save_or_return_probabilities,
            )
            if save_or_return_probabilities:
                return seg, prob, attr_logits
            else:
                return seg, attr_logits
        


import argparse
import torch
import os
import cProfile
import pstats

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
    parser.add_argument('--clinical_data_folder', type=str, required=False, default=None, help='臨床資料資料夾 (pkl)')

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
        verbose=False,
        allow_tqdm=not args.disable_progress_bar,
        verbose_preprocessing=False
    )

    predictor.initialize_from_trained_model_folder(
        model_folder,
        args.f,
        checkpoint_name=args.chk
    )

    # 進行效能測試
    # profiler = cProfile.Profile()
    # profiler.enable()

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
        clinical_data_folder=args.clinical_data_folder
    )

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats(20)  # 印出前20個最耗時的函數


    # # 進行效能測試
    # # 我們將使用 torch.profiler 代替 cProfile
    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,  # 如果使用 GPU
    #     ],
    #     schedule=torch.profiler.schedule(
    #         wait=1,  # 前1個 step 不分析 (例如第一次迭代的數據加載)
    #         warmup=0,  # 接下來1個 step 作為預熱 (讓 CUDA 內核達到穩定狀態)
    #         active=2,  # 接下來3個 step 進行分析 (分析3次核心預測)
    #         repeat=0   # 重複一次這個循環 (總共分析 3*2 = 6 次)
    #     ),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),  # 輸出到 ./log 目錄，可用 TensorBoard 查看
    #     record_shapes=True,       # 記錄張量形狀
    #     profile_memory=True,      # 分析內存使用 (較耗資源)
    #     with_stack=True,          # 記錄 Python 堆疊 (用於追溯調用來源)
    #     with_flops=True           # 計算浮點運算次數 (FLOPs)
    # ) as prof:
        
    #     # 執行推論，並將 profiler 對象傳遞給 predict_from_files
    #     # 注意：我們需要修改 predict_from_files 來接受 profiler，但為了不修改太多，
    #     # 我們直接調用 predict_from_data_iterator，因為我們已經修改了它。
    #     # 但為了保持與您原始代碼的相似性，我們可以這樣做：
        
    #     # 1. 使用 predict_from_files 的邏輯創建 list_of_lists 等
    #     list_of_lists, output_filenames, seg_prev_files = predictor._manage_input_and_output_lists(
    #         args.i,
    #         args.o,
    #         args.prev_stage_predictions,
    #         overwrite=not args.continue_prediction,
    #         part_id=args.part_id,
    #         num_parts=args.num_parts
    #     )
        
    #     if len(list_of_lists) == 0:
    #         print(f'進程 {args.part_id}/{args.num_parts} 無需處理新案例，跳過')
    #         return []

    #     # 2. 創建 data_iterator
    #     data_iterator = predictor._internal_get_data_iterator_from_lists_of_filenames(
    #         list_of_lists,
    #         seg_prev_files,
    #         output_filenames,
    #         num_processes=args.npp,
    #         clinical_data_folder=args.clinical_data_folder
    #     )

    #     # 3. 直接調用 predict_from_data_iterator 並傳入 profiler
    #     predictor.predict_from_data_iterator(
    #         data_iterator,
    #         save_probabilities=args.save_probabilities,
    #         num_processes_segmentation_export=args.nps,
    #         profiler=prof  # 關鍵：傳遞 profiler
    #     )

    # # 分析結束，輸出結果
    # print("\n" + "="*80)
    # print("效能分析結果 (按 CUDA 時間排序):")
    # print("="*80)
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # print("\n" + "="*80)
    # print("效能分析結果 (按 CPU 時間排序):")
    # print("="*80)
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # print("\n效能分析完成。TensorBoard 追蹤已保存到 ./log 目錄。")

    # # 如果您想完全保持原始代碼，可以將 profiler 傳遞給一個修改版的 predict_from_files，
    # # 但這需要修改 predict_from_files，這與修改 predict_from_data_iterator 的工作量相似。
    # # 因此，上述方法是最佳平衡點。