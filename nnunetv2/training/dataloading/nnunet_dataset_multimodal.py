# nnunetv2/training/dataloading/nnunet_dataset_multimodal.py

import os
from typing import List, Union, Type, Tuple
import numpy as np
import blosc2
from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile
# 從 nnUNet 原有資料集介面和實作檔案中導入 nnUNetDatasetBlosc2
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2, nnUNetDatasetNumpy, file_ending_dataset_mapping
import inspect

class nnUNetDatasetMultimodal(nnUNetDatasetBlosc2):
    """
    擴展 nnUNetDatasetBlosc2，使其能夠加載臨床資料。
    這個類別會從預處理好的臨床資料資料夾中讀取額外的 .pkl 檔案。
    """
    def __init__(self, folder: str, identifiers: List[str] = None,
                 folder_with_segs_from_previous_stage: str = None,
                 clinical_data_folder: str = None):
        """
        初始化多模態資料集。
        Args:
            folder (str): 預處理後的 nnU-Net 資料集根資料夾 (例如 nnUNet_preprocessed/DatasetXXX_YYY/configurations/...).
            identifiers (List[str], optional): 要加載的病例識別符列表。如果為 None，則自動從資料夾中獲取。
            folder_with_segs_from_previous_stage (str, optional): 來自前一階段分割結果的資料夾。
            clinical_data_folder (str, optional): 包含預處理臨床資料 .pkl 檔案的資料夾路徑。
                                                如果為 None，則不會加載臨床資料。
        """
        super().__init__(folder, identifiers, folder_with_segs_from_previous_stage)
        self.clinical_data_folder = clinical_data_folder
        print(f"nnUNetDatasetMultimodal 初始化。臨床資料資料夾: {self.clinical_data_folder}")

    def load_case(self, identifier):
        """
        加載單個病例的影像、標註和臨床資料。
        覆寫父類的 load_case 方法以包含臨床資料。
        """
        # 調用父類的 load_case 方法加載影像和標註數據
        data, seg, seg_prev, properties = super().load_case(identifier)

        clinical_features_dict = {
            'prompt_features': None,
            'location_label': -1,
            't_stage_label': -1,
            'n_stage_label': -1,
            'm_stage_label': -1,
            'has_clinical_data': False # 預設為沒有臨床資料
        }

        # 如果指定了臨床資料資料夾，則嘗試加載臨床資料
        if self.clinical_data_folder:
            # 假設臨床資料檔案命名為 Case_Index_0000.pkl
            clinical_data_path = join(self.clinical_data_folder, f"{identifier}_0000.pkl") 
            if isfile(clinical_data_path):
                try:
                    loaded_clinical_data = load_pickle(clinical_data_path)
                    clinical_features_dict.update(loaded_clinical_data)
                    # 確保 prompt_features 為 float32 以匹配模型輸入
                    clinical_features_dict['prompt_features'] = loaded_clinical_data['prompt_features'].astype(np.float32)
                    # 'has_clinical_data' 在預處理時已決定，直接使用其值
                except Exception as e:
                    print(f"警告: 加載病例 {identifier} 的臨床資料時發生錯誤: {e}")
                    # 保持預設值 (即視為沒有臨床資料)
            else:
                print(f"警告: 未找到病例 {identifier} 的臨床資料檔案: {clinical_data_path}")
                # 保持預設值 (即視為沒有臨床資料)
        
        # 返回影像數據、分割標註、前一階段分割、屬性字典，以及新的臨床資料字典
        # 這裡將臨床資料作為一個獨立的字典返回，方便 Dataloader 和 Predictor 處理
        stack = inspect.stack()
        for frame in stack:
            # 如果是 determine_shapes 呼叫，只回傳 4 個值，因為它只需要影像的形狀信息
            if frame.function == "determine_shapes":
                return data, seg, seg_prev, properties
        
        # 對於其他所有情況 (包括 perform_actual_validation 和 DataLoader 的常規載入)，都回傳 5 個值
        return data, seg, seg_prev, properties, clinical_features_dict
    

# 更新 nnUNetDataset 中的 file_ending_dataset_mapping，以便能夠自動識別並使用我們的多模態資料集
# 我們需要一種方式來區分，例如，通過資料集名稱來決定是否使用多模態資料集。
# 或者，最簡單的方式是，如果存在 clinical_data_folder，就直接在 Trainer 中強制使用這個 Dataset。
# 為了靈活性，我們將在 Trainer 中判斷並選擇使用哪個 Dataset。
# 但為了 nnUNetDataset.infer_dataset_class 的通用性，我們可以讓它知道這個新的類別。
file_ending_dataset_mapping['b2nd_multimodal'] = nnUNetDatasetMultimodal # 假設一種新的文件結尾或標記

# 為了兼容性，保留原始的 infer_dataset_class，但我們會在 Trainer 中根據需求選擇
def infer_dataset_class_multimodal(folder: str, clinical_data_folder: str = None) -> Union[Type[nnUNetDatasetMultimodal], Type[nnUNetDatasetBlosc2], Type[nnUNetDatasetNumpy]]:
    """
    推斷資料集類別，如果存在臨床資料資料夾，則返回多模態資料集類別。
    """
    # 首先嘗試推斷常規 nnUNet 資料集類型 (Blosc2 或 Numpy)
    base_dataset_class = nnUNetDatasetBlosc2 # 預設使用 Blosc2
    
    # 檢查資料夾中是否存在 .b2nd 或 .npz 檔案來確定基礎類型
    # 這部分邏輯需要從 nnunet_dataset.py 中的 infer_dataset_class 拷貝過來
    # 由於我們沒有直接訪問該函數，假設直接從檔案結尾判斷
    has_b2nd = any(f.endswith(".b2nd") for f in os.listdir(folder))
    has_npz = any(f.endswith(".npz") for f in os.listdir(folder))

    if has_b2nd:
        base_dataset_class = nnUNetDatasetBlosc2
    elif has_npz:
        base_dataset_class = nnUNetDatasetNumpy
    else:
        raise RuntimeError(f"無法推斷資料集類型：{folder} 中沒有 .b2nd 或 .npz 檔案。")

    # 如果指定了臨床資料資料夾並且該資料夾存在，則使用多模態資料集類別
    if clinical_data_folder and os.path.isdir(clinical_data_folder):
        print(f"檢測到臨床資料資料夾 '{clinical_data_folder}'，將使用 nnUNetDatasetMultimodal。")
        return nnUNetDatasetMultimodal
    else:
        print(f"未檢測到臨床資料資料夾或路徑無效，將使用基礎 nnUNetDataset 類別 ({base_dataset_class.__name__})。")
        return base_dataset_class