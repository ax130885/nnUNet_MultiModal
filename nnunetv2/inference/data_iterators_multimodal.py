# 為了讓 nnUNet 在「推論階段」也能讀取臨床資料 (prompt)
# 我們複製原有的 iterator 邏輯，並在裡面加上「讀取臨床 pkl」的步驟
import os
from typing import List, Union
import numpy as np
import torch
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from batchgenerators.dataloading.data_loader import DataLoader
from nnunetv2.training.dataloading.nnunet_data_loader_multimodal import nnUNetDataLoaderMultimodal
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_pickle, isfile, join
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.preprocessing.clinical_data_label_encoder import ClinicalDataLabelEncoder


import multiprocessing
import queue
from time import sleep
from torch.multiprocessing import Event, Queue, Manager
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot


def preprocess_fromfiles_save_to_queue_multimodal(list_of_lists: List[List[str]],
                                       list_of_segs_from_prev_stage_files: Union[None, List[str]],
                                       output_filenames_truncated: Union[None, List[str]],
                                       plans_manager: PlansManager,
                                       dataset_json: dict,
                                       configuration_manager: ConfigurationManager,
                                       target_queue: Queue,
                                       done_event: Event,
                                       abort_event: Event,
                                       verbose: bool = False,
                                       clinical_data_dir: Union[str, None] = None):
    
    """
    多模態版本的預處理函數，從檔案載入影像資料並預處理，同時載入臨床資料。
    結果會放入隊列中供主進程使用。

    Args:
        list_of_lists: 每個案例的影像檔案路徑清單
        list_of_segs_from_prev_stage_files: 前階段分割檔案清單 (可為 None)
        output_filenames_truncated: 輸出檔案名稱清單 (可為 None)
        plans_manager: nnUNet 計劃管理器
        dataset_json: 資料集 JSON
        configuration_manager: 配置管理器
        target_queue: 結果放入的目標隊列
        done_event: 完成處理的事件標記
        abort_event: 中止處理的事件標記
        verbose: 是否輸出詳細日誌
        clinical_data_dir: 臨床資料目錄路徑
    """
    total_cases = len(list_of_lists)
    print(f"該 iterator worker 收到總共 {total_cases} 筆資料")

    try:
        # 取得標籤管理器（用於分割標籤 one-hot 編碼）
        label_manager = plans_manager.get_label_manager(dataset_json)
        # 取得預處理器實例（根據 configuration_manager 設定）
        preprocessor = configuration_manager.preprocessor_class(verbose=verbose)

        # ---------------------臨床資料用 新增的部份---------------------
        # 初始化臨床數據編碼器
        clinical_data_label_encoder = None
        missing_flags = None
        if clinical_data_dir:
            try:
                from nnunetv2.preprocessing.clinical_data_label_encoder import ClinicalDataLabelEncoder
                clinical_data_label_encoder = ClinicalDataLabelEncoder(clinical_data_dir)
                missing_flags = {
                    'location': clinical_data_label_encoder.missing_flag_location,
                    't_stage': clinical_data_label_encoder.missing_flag_t_stage,
                    'n_stage': clinical_data_label_encoder.missing_flag_n_stage,
                    'm_stage': clinical_data_label_encoder.missing_flag_m_stage
                }
            except Exception as e:
                if verbose:
                    print(f"[警告] 無法初始化 ClinicalDataLabelEncoder: {e}")
        # ------------------------------------------

        # 逐一處理每個案例
        for idx in range(len(list_of_lists)):
            # 執行預處理：讀取影像與分割檔案，並回傳預處理後的資料與屬性
            data, seg, data_properties = preprocessor.run_case(
                list_of_lists[idx],
                list_of_segs_from_prev_stage_files[idx] if list_of_segs_from_prev_stage_files is not None else None,
                plans_manager,
                configuration_manager,
                dataset_json
            )
            # 如果有前階段分割，則將分割標籤做 one-hot 編碼並合併到影像資料
            if list_of_segs_from_prev_stage_files is not None and list_of_segs_from_prev_stage_files[idx] is not None:
                seg_onehot = convert_labelmap_to_one_hot(seg[0], label_manager.foreground_labels, data.dtype)
                data = np.vstack((data, seg_onehot))

            # 將 numpy array 轉換為 torch tensor，並設為 float32 及 contiguous 格式
            data = torch.from_numpy(data).to(dtype=torch.float32, memory_format=torch.contiguous_format)


            # ---------------------臨床資料用 新增的部份---------------------
            # 讀取臨床資料
            clinical_data = {
                'location': missing_flags.get('location', -1) if missing_flags else -1,
                't_stage': missing_flags.get('t_stage', -1) if missing_flags else -1,
                'n_stage': missing_flags.get('n_stage', -1) if missing_flags else -1,
                'm_stage': missing_flags.get('m_stage', -1) if missing_flags else -1
            }
            
            clinical_mask = {
                'location': False,
                't_stage': False,
                'n_stage': False,
                'm_stage': False
            }

            # 從臨床資料檔案讀取數據
            case_id = data_properties.get('case_identifier', None)
            if case_id and clinical_data_label_encoder:
                try:
                    df = clinical_data_label_encoder.clinical_full_df
                    row = df[df['Case_Index'] == case_id]
                    
                    if not row.empty:
                        clinical_data['location'] = row['Location'].values[0]
                        clinical_data['t_stage'] = row['T_stage'].values[0]
                        clinical_data['n_stage'] = row['N_stage'].values[0]
                        clinical_data['m_stage'] = row['M_stage'].values[0]
                        
                        # 設置掩碼，表示哪些特徵有效
                        clinical_mask['location'] = clinical_data['location'] != missing_flags.get('location', -1)
                        clinical_mask['t_stage'] = clinical_data['t_stage'] != missing_flags.get('t_stage', -1)
                        clinical_mask['n_stage'] = clinical_data['n_stage'] != missing_flags.get('n_stage', -1)
                        clinical_mask['m_stage'] = clinical_data['m_stage'] != missing_flags.get('m_stage', -1)
                except Exception as e:
                    if verbose:
                        print(f"[警告] 讀取臨床數據失敗: {e}")
            # ------------------------------------------


            # # 組成一個 dict，包含預處理後的資料、屬性、輸出檔名
            # item = {
            #     'data': data,
            #     'data_properties': data_properties,
            #     'ofile': output_filenames_truncated[idx] if output_filenames_truncated is not None else None
            # }

            # 組成一個 dict，包含預處理後的資料、臨床資料、屬性、輸出檔名
            item = {
                'data': data,  # 影像數據
                'target': seg,  # 分割標籤
                'clinical_data': clinical_data,  # 臨床數據字典
                'clinical_mask': clinical_mask,  # 臨床掩碼字典
                'properties': data_properties,  # 數據屬性
                'ofile': output_filenames_truncated[idx] if output_filenames_truncated is not None else None  # 輸出文件路徑
            }

            # 嘗試將 item 放入 queue，若 queue 滿則重試，直到成功或 abort_event 被觸發
            success = False
            while not success:
                try:
                    if abort_event.is_set():
                        # 若有 abort 訊號則直接結束
                        return
                    target_queue.put(item, timeout=0.01)
                    success = True
                except queue.Full:
                    # queue 滿時等待重試
                    pass
        # 所有案例處理完畢，設定完成事件
        done_event.set()
    except Exception as e:
        # 若發生例外，觸發 abort_event 並拋出例外
        abort_event.set()
        raise e
    




# 具體的資料載入邏輯在 preprocess_fromfiles_save_to_queue
# 此函式只是一個多進程包裝器 內容與原生 preprocessing_iterator_fromfiles 幾乎相同
# 只差在 資料載入邏輯 改成 preprocess_fromfiles_save_to_queue_multimodal 還有多接收 clinical_data_dir 參數
def preprocessing_iterator_fromfiles_multimodal(list_of_lists: List[List[str]],
                                     list_of_segs_from_prev_stage_files: Union[None, List[str]],
                                     output_filenames_truncated: Union[None, List[str]],
                                     plans_manager: PlansManager,
                                     dataset_json: dict,
                                     configuration_manager: ConfigurationManager,
                                     num_processes: int,
                                     pin_memory: bool = False,
                                     verbose: bool = False,
                                     clinical_data_dir: Union[str, None] = None):
    """
    多進程資料預處理迭代器
    主要用於 nnUNet 推論階段，將原始影像與分割資料進行預處理，並以多進程方式加速。
    每個 worker 處理一部分資料，結果透過 queue 回傳給主進程。
    """
    # 取得 multiprocessing context，使用 'spawn' 模式以避免 fork 帶來的問題
    context = multiprocessing.get_context('spawn')
    manager = Manager()
    # 決定實際啟動的進程數量（不超過資料數量）
    num_processes = min(len(list_of_lists), num_processes)
    assert num_processes >= 1

    processes = []      # 儲存所有 worker process
    done_events = []    # 每個 worker 的完成事件
    target_queues = []  # 每個 worker 的 queue，用來傳遞預處理結果
    abort_event = manager.Event()  # 全域 abort event，若有錯誤可中止所有 worker

    # 啟動多個 worker process，每個 process 處理分配到的資料
    for i in range(num_processes):
        event = manager.Event()  # 單一 worker 的完成事件
        queue = Manager().Queue(maxsize=1)  # 單一 worker 的 queue，maxsize=1 表示只存一個 batch
        # 分配資料給每個 worker（i::num_processes 代表分片）
        pr = context.Process(target=preprocess_fromfiles_save_to_queue_multimodal,
                     args=(
                         list_of_lists[i::num_processes],
                         list_of_segs_from_prev_stage_files[
                         i::num_processes] if list_of_segs_from_prev_stage_files is not None else None,
                         output_filenames_truncated[
                         i::num_processes] if output_filenames_truncated is not None else None,
                         plans_manager,
                         dataset_json,
                         configuration_manager,
                         queue,
                         event,
                         abort_event,
                         verbose,
                        clinical_data_dir
                     ), daemon=True)
        pr.start()
        target_queues.append(queue)
        done_events.append(event)
        processes.append(pr)

    # 主進程輪詢各 worker 的 queue，依序取得預處理結果
    worker_ctr = 0
    while (not done_events[worker_ctr].is_set()) or (not target_queues[worker_ctr].empty()):
        # 若 queue 有資料，則取出並 yield 給後續流程
        if not target_queues[worker_ctr].empty():
            item = target_queues[worker_ctr].get()
            worker_ctr = (worker_ctr + 1) % num_processes
        else:
            # 檢查所有 worker 是否正常運作，若有異常則報錯
            all_ok = all(
                [i.is_alive() or j.is_set() for i, j in zip(processes, done_events)]) and not abort_event.is_set()
            if not all_ok:
                raise RuntimeError('Background workers died. Look for the error message further up! If there is '
                                   'none then your RAM was full and the worker was killed by the OS. Use fewer '
                                   'workers or get more RAM in that case!')
            sleep(0.01)
            continue
        # 若需將資料放到 CUDA pinned memory（加速 GPU 傳輸），則執行 pin_memory
        if pin_memory:
            [i.pin_memory() for i in item.values() if isinstance(i, torch.Tensor)]
        yield item  # 回傳預處理後的 batch 給主流程

    # 等待所有 worker 結束
    [p.join() for p in processes]


