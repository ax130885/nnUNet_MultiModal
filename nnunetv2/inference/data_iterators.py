import multiprocessing
import queue
from torch.multiprocessing import Event, Queue, Manager

from time import sleep
from typing import Union, List

import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader

from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


def preprocess_fromfiles_save_to_queue(list_of_lists: List[List[str]],
                                       list_of_segs_from_prev_stage_files: Union[None, List[str]],
                                       output_filenames_truncated: Union[None, List[str]],
                                       plans_manager: PlansManager,
                                       dataset_json: dict,
                                       configuration_manager: ConfigurationManager,
                                       target_queue: Queue,
                                       done_event: Event,
                                       abort_event: Event,
                                       verbose: bool = False):
    try:
        # 取得標籤管理器（用於分割標籤 one-hot 編碼）
        label_manager = plans_manager.get_label_manager(dataset_json)
        # 取得預處理器實例（根據 configuration_manager 設定）
        preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
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

            # 組成一個 dict，包含預處理後的資料、屬性、輸出檔名
            item = {
                'data': data,
                'data_properties': data_properties,
                'ofile': output_filenames_truncated[idx] if output_filenames_truncated is not None else None
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


def preprocessing_iterator_fromfiles(list_of_lists: List[List[str]],
                                     list_of_segs_from_prev_stage_files: Union[None, List[str]],
                                     output_filenames_truncated: Union[None, List[str]],
                                     plans_manager: PlansManager,
                                     dataset_json: dict,
                                     configuration_manager: ConfigurationManager,
                                     num_processes: int,
                                     pin_memory: bool = False,
                                     verbose: bool = False):
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
        pr = context.Process(target=preprocess_fromfiles_save_to_queue,
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
                         verbose
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


class PreprocessAdapter(DataLoader):
    def __init__(self, list_of_lists: List[List[str]],
                 list_of_segs_from_prev_stage_files: Union[None, List[str]],
                 preprocessor: DefaultPreprocessor,
                 output_filenames_truncated: Union[None, List[str]],
                 plans_manager: PlansManager,
                 dataset_json: dict,
                 configuration_manager: ConfigurationManager,
                 num_threads_in_multithreaded: int = 1):
        # 初始化 PreprocessAdapter，負責將原始影像檔案、前階段分割、預處理器等資訊組合成 DataLoader 格式
        # list_of_lists: 每個案例的影像檔路徑清單
        # list_of_segs_from_prev_stage_files: 前階段分割檔案清單（可為 None）
        # preprocessor: 預處理器實例（如 DefaultPreprocessor）
        # output_filenames_truncated: 輸出檔案名稱清單（可為 None）
        # plans_manager, dataset_json, configuration_manager: nnUNet 的計劃、資料集、配置管理器
        # num_threads_in_multithreaded: 多線程數量（預設 1）

        # 儲存必要的物件供後續使用
        self.preprocessor, self.plans_manager, self.configuration_manager, self.dataset_json = \
            preprocessor, plans_manager, configuration_manager, dataset_json

        # 取得標籤管理器（用於 one-hot 編碼分割標籤）
        self.label_manager = plans_manager.get_label_manager(dataset_json)

        # 若前階段分割或輸出檔案名稱為 None，則補齊為同長度的 None 清單
        if list_of_segs_from_prev_stage_files is None:
            list_of_segs_from_prev_stage_files = [None] * len(list_of_lists)
        if output_filenames_truncated is None:
            output_filenames_truncated = [None] * len(list_of_lists)

        # 呼叫父類 DataLoader 初始化，將所有資料組合成 (影像, 分割, 輸出檔名) 的 tuple
        # batch_size 固定為 1，無 shuffle，無 infinite loop
        super().__init__(list(zip(list_of_lists, list_of_segs_from_prev_stage_files, output_filenames_truncated)),
                         1, num_threads_in_multithreaded,
                         seed_for_shuffle=1, return_incomplete=True,
                         shuffle=False, infinite=False, sampling_probabilities=None)

        # 建立索引清單，方便後續資料存取
        self.indices = list(range(len(list_of_lists)))

    def generate_train_batch(self):
        # 產生一個訓練批次（batch_size=1），回傳格式為 dict
        # 1. 取得目前要處理的資料索引
        # 如果有前階段的分割結果，必須與影像一起處理（如裁切），
        # 以確保分割標籤與影像空間對齊（必要時裁切）。
        # 否則分割只會被 resize 成預處理後的影像大小，可能導致標籤與影像不對齊。
        idx = self.get_indices()[0]
        files, seg_prev_stage, ofile = self._data[idx]

        # 2. 執行預處理：將影像與分割（如有）送入預處理器
        #    - files: 影像檔案路徑
        #    - seg_prev_stage: 前階段分割檔案（可為 None）
        #    - ofile: 輸出檔案名稱（可為 None）
        #    - data: 預處理後的影像 numpy array
        #    - seg: 預處理後的分割 numpy array
        #    - data_properties: 影像屬性（如 shape, spacing 等）
        data, seg, data_properties = self.preprocessor.run_case(files, seg_prev_stage, self.plans_manager,
                                                                self.configuration_manager,
                                                                self.dataset_json)

        # 3. 若有前階段分割，則將分割標籤 one-hot 編碼後與影像資料合併
        if seg_prev_stage is not None:
            seg_onehot = convert_labelmap_to_one_hot(seg[0], self.label_manager.foreground_labels, data.dtype)
            data = np.vstack((data, seg_onehot))

        # 4. 將 numpy array 轉換為 torch tensor，方便後續模型推論
        data = torch.from_numpy(data)

        # 5. 回傳一個 dict，包含：
        #    - 'data': 預處理後的影像 tensor
        #    - 'data_properties': 影像屬性
        #    - 'ofile': 輸出檔案名稱（可為 None）
        return {'data': data, 'data_properties': data_properties, 'ofile': ofile}


class PreprocessAdapterFromNpy(DataLoader):
    def __init__(self, list_of_images: List[np.ndarray],
                 list_of_segs_from_prev_stage: Union[List[np.ndarray], None],
                 list_of_image_properties: List[dict],
                 truncated_ofnames: Union[List[str], None],
                 plans_manager: PlansManager, dataset_json: dict, configuration_manager: ConfigurationManager,
                 num_threads_in_multithreaded: int = 1, verbose: bool = False):
        preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
        self.preprocessor, self.plans_manager, self.configuration_manager, self.dataset_json, self.truncated_ofnames = \
            preprocessor, plans_manager, configuration_manager, dataset_json, truncated_ofnames

        self.label_manager = plans_manager.get_label_manager(dataset_json)

        if list_of_segs_from_prev_stage is None:
            list_of_segs_from_prev_stage = [None] * len(list_of_images)
        if truncated_ofnames is None:
            truncated_ofnames = [None] * len(list_of_images)

        super().__init__(
            list(zip(list_of_images, list_of_segs_from_prev_stage, list_of_image_properties, truncated_ofnames)),
            1, num_threads_in_multithreaded,
            seed_for_shuffle=1, return_incomplete=True,
            shuffle=False, infinite=False, sampling_probabilities=None)

        self.indices = list(range(len(list_of_images)))

    def generate_train_batch(self):
        idx = self.get_indices()[0]
        image, seg_prev_stage, props, ofname = self._data[idx]
        # 如果有前階段的分割結果，必須與影像一起處理（如裁切），
        # 以確保分割標籤與影像空間對齊（必要時裁切）。
        # 否則分割只會被 resize 成預處理後的影像大小，可能導致標籤與影像不對齊。
        data, seg, props = self.preprocessor.run_case_npy(image, seg_prev_stage, props,
                                                   self.plans_manager,
                                                   self.configuration_manager,
                                                   self.dataset_json)
        if seg_prev_stage is not None:
            seg_onehot = convert_labelmap_to_one_hot(seg[0], self.label_manager.foreground_labels, data.dtype)
            data = np.vstack((data, seg_onehot))

        data = torch.from_numpy(data)

        return {'data': data, 'data_properties': props, 'ofile': ofname}


def preprocess_fromnpy_save_to_queue(list_of_images: List[np.ndarray],
                                     list_of_segs_from_prev_stage: Union[List[np.ndarray], None],
                                     list_of_image_properties: List[dict],
                                     truncated_ofnames: Union[List[str], None],
                                     plans_manager: PlansManager,
                                     dataset_json: dict,
                                     configuration_manager: ConfigurationManager,
                                     target_queue: Queue,
                                     done_event: Event,
                                     abort_event: Event,
                                     verbose: bool = False):
    try:
        label_manager = plans_manager.get_label_manager(dataset_json)
        preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
        for idx in range(len(list_of_images)):
            data, seg, props = preprocessor.run_case_npy(list_of_images[idx],
                                                  list_of_segs_from_prev_stage[
                                                      idx] if list_of_segs_from_prev_stage is not None else None,
                                                  list_of_image_properties[idx],
                                                  plans_manager,
                                                  configuration_manager,
                                                  dataset_json)
            list_of_image_properties[idx] = props
            if list_of_segs_from_prev_stage is not None and list_of_segs_from_prev_stage[idx] is not None:
                seg_onehot = convert_labelmap_to_one_hot(seg[0], label_manager.foreground_labels, data.dtype)
                data = np.vstack((data, seg_onehot))

            data = torch.from_numpy(data).to(dtype=torch.float32, memory_format=torch.contiguous_format)

            item = {'data': data, 'data_properties': list_of_image_properties[idx],
                    'ofile': truncated_ofnames[idx] if truncated_ofnames is not None else None}
            success = False
            while not success:
                try:
                    if abort_event.is_set():
                        return
                    target_queue.put(item, timeout=0.01)
                    success = True
                except queue.Full:
                    pass
        done_event.set()
    except Exception as e:
        abort_event.set()
        raise e


def preprocessing_iterator_fromnpy(list_of_images: List[np.ndarray],
                                   list_of_segs_from_prev_stage: Union[List[np.ndarray], None],
                                   list_of_image_properties: List[dict],
                                   truncated_ofnames: Union[List[str], None],
                                   plans_manager: PlansManager,
                                   dataset_json: dict,
                                   configuration_manager: ConfigurationManager,
                                   num_processes: int,
                                   pin_memory: bool = False,
                                   verbose: bool = False):
    context = multiprocessing.get_context('spawn')
    manager = Manager()
    num_processes = min(len(list_of_images), num_processes)
    assert num_processes >= 1
    target_queues = []
    processes = []
    done_events = []
    abort_event = manager.Event()
    for i in range(num_processes):
        event = manager.Event()
        queue = manager.Queue(maxsize=1)
        pr = context.Process(target=preprocess_fromnpy_save_to_queue,
                     args=(
                         list_of_images[i::num_processes],
                         list_of_segs_from_prev_stage[
                         i::num_processes] if list_of_segs_from_prev_stage is not None else None,
                         list_of_image_properties[i::num_processes],
                         truncated_ofnames[i::num_processes] if truncated_ofnames is not None else None,
                         plans_manager,
                         dataset_json,
                         configuration_manager,
                         queue,
                         event,
                         abort_event,
                         verbose
                     ), daemon=True)
        pr.start()
        done_events.append(event)
        processes.append(pr)
        target_queues.append(queue)

    worker_ctr = 0
    while (not done_events[worker_ctr].is_set()) or (not target_queues[worker_ctr].empty()):
        if not target_queues[worker_ctr].empty():
            item = target_queues[worker_ctr].get()
            worker_ctr = (worker_ctr + 1) % num_processes
        else:
            all_ok = all(
                [i.is_alive() or j.is_set() for i, j in zip(processes, done_events)]) and not abort_event.is_set()
            if not all_ok:
                raise RuntimeError('Background workers died. Look for the error message further up! If there is '
                                   'none then your RAM was full and the worker was killed by the OS. Use fewer '
                                   'workers or get more RAM in that case!')
            sleep(0.01)
            continue
        if pin_memory:
            [i.pin_memory() for i in item.values() if isinstance(i, torch.Tensor)]
        yield item
    [p.join() for p in processes]
