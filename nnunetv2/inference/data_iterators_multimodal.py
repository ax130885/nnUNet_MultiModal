# 為了讓 nnUNet 在「推論階段」也能讀取臨床資料 (prompt)
# 我們複製原有的 iterator 邏輯，並在裡面加上「讀取臨床 pkl」的步驟
import os
from typing import List, Union
import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_pickle, isfile, join
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

# 與 MyModel 內的 prompt_dim 保持一致，避免硬編碼錯誤
DEFAULT_PROMPT_DIM = 14


class PreprocessAdapterBareBonesMultimodal(DataLoader):
    # 不用繼承自訂dataloader，因為我們只需要簡單的預處理功能
    """
    簡化版的預處理器，專門為「多模態推論」設計。
    讀檔→預處理影像→讀取臨床資料→回傳 dict
    每個推論案例的 batch_size 固定為 1
    """
    def __init__(
        self,
        list_of_lists: List[List[str]],                       # 每個案例的影像檔路徑
        segs_from_prev_stage_files: Union[List[str], None],   # 上一階段的分割檔
        output_filenames_truncated: Union[List[str], None],   # 輸出檔前綴
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_threads_in_multithreaded: int = 1,
        perform_everything_on_gpu: bool = True,
        verbose: bool = False,
        clinical_data_folder: Union[str, None] = None         # 臨床資料資料夾
    ):
        # 初始化 DataLoader：list(zip(...)) 是為了讓每筆資料包含 (影像檔, seg_prev, ofile)
        super().__init__(
            list(zip(list_of_lists, segs_from_prev_stage_files or [None] * len(list_of_lists),
                     output_filenames_truncated or [None] * len(list_of_lists))),
            1,
            num_threads_in_multithreaded,
            seed_for_shuffle=1234,
            return_incomplete=True,
            shuffle=False,
            infinite=False
        )
        self.indices = list(range(len(list_of_lists)))

        # 把常用的東西存起來，避免重複建立
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.dataset_json = dataset_json
        self.verbose = verbose
        self.clinical_data_folder = clinical_data_folder

        # 取得預處理器與 input channels
        self.preprocessor = self.configuration_manager.preprocessor_class(verbose=verbose)
        self.num_input_channels = determine_num_input_channels(
            plans_manager, configuration_manager, dataset_json
        )

    def generate_train_batch(self):
        """
        產生一筆資料 (batch_size=1)
        回傳格式：
        {
            'data': {
                'data': 影像 tensor,
                'clinical_features': 臨床 prompt tensor,
                'has_clinical_data': bool tensor
            },
            'data_properties': props,
            'ofile': ofile # 輸出完整路徑與檔名 但沒有副檔名
        }
        """
        # DataLoader 的內建方法，會依序吐 index
        idx = self.get_indices()[0]
        image_fnames, seg_prev_fname, ofile = self._data[idx]

        # 1. 執行 nnUNet 標準預處理
        data, seg, props = self.preprocessor.run_case(
            image_fnames,
            seg_prev_fname,
            self.plans_manager,
            self.configuration_manager,
            self.dataset_json
        )

        # 2. 讀取臨床資料
        #   預設值：全部補 0，方便 model 統一處理
        prompt_np = np.zeros((1, DEFAULT_PROMPT_DIM), dtype=np.float32)
        has_clinical = False

        #   用 props 中的 case_identifier 去找對應 pkl
        case_id = props.get('case_identifier', None)
        if case_id and self.clinical_data_folder:
            pkl_path = join(self.clinical_data_folder, f"{case_id}_0000.pkl")
            if isfile(pkl_path):
                try:
                    tmp = load_pickle(pkl_path)
                    prompt_np = tmp['prompt_features'].astype(np.float32)[None]  # 加 batch 維
                    has_clinical = True
                except Exception as e:
                    if self.verbose:
                        print(f"[警告] 讀取 {pkl_path} 失敗: {e}")

        # 3. 打包回傳
        batch_dict = {
            'data': {
                'data': torch.from_numpy(data).float(),                   # 影像
                'clinical_features': torch.from_numpy(prompt_np).float(), # prompt
                'has_clinical_data': torch.tensor([has_clinical])         # flag
            },
            'data_properties': props,
            'ofile': ofile # 輸出完整路徑與檔名 但沒有副檔名
        }
        return batch_dict

# 此方法放在class外面
def preprocessing_iterator_fromfiles_multimodal(
    list_of_lists: List[List[str]],
    segs_from_prev_stage_files: Union[List[str], None],
    output_filenames_truncated: Union[List[str], None],
    plans_manager: PlansManager,
    dataset_json: dict,
    configuration_manager: ConfigurationManager,
    num_processes: int,
    pin_memory: bool = False,
    verbose: bool = False,
    clinical_data_folder: Union[str, None] = None
):
    """
    多程序/多線程版資料迭代器
    回傳 MultiThreadedAugmenter，可直接丟給 nnUNetPredictorMultimodal
    """
    # 建立 DataLoader
    dataloader = PreprocessAdapterBareBonesMultimodal(
        list_of_lists,
        segs_from_prev_stage_files,
        output_filenames_truncated,
        plans_manager,
        dataset_json,
        configuration_manager,
        num_threads_in_multithreaded=1,   # DataLoader 內部線程設 1，讓 MultiThreadedAugmenter 控制
        perform_everything_on_gpu=False,  # 預處理在 CPU
        verbose=verbose,
        clinical_data_folder=clinical_data_folder
    )

    # 用 MultiThreadedAugmenter 做 multiprocessing
    mt = MultiThreadedAugmenter(
        dataloader,
        None,               # 不做 augmentation
        num_processes,
        1,                  # num_cached_per_queue
        None,               # seeds
        pin_memory=pin_memory
    )
    return mt