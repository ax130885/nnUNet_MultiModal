# nnunetv2/training/dataloading/nnunet_dataset_multimodal.py

import os
from typing import List, Union, Type, Tuple
import numpy as np
import blosc2
from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile
# 從 nnUNet 原有資料集介面和實作檔案中導入 nnUNetDatasetBlosc2
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2, nnUNetDatasetNumpy
from nnunetv2.preprocessing.clinical_data_label_encoder import ClinicalDataLabelEncoder
import inspect
import torch

class nnUNetDatasetMultimodal(nnUNetDatasetBlosc2):
    """
    擴展 nnUNetDatasetBlosc2，使其能夠加載臨床資料。
    直接讀取原始的 csv 檔案。
    """
    def __init__(self, folder: str, identifiers: List[str] = None,
                 folder_with_segs_from_previous_stage: str = None,
                 clinical_data_dir: str = None):
        """
        初始化多模態資料集。
        Args:
            folder (str): 預處理後的 nnU-Net 資料集根資料夾 (例如 nnUNet_preprocessed/DatasetXXX_YYY/configurations/...).
            identifiers (List[str], optional): 要加載的病例識別符列表。如果為 None，則自動從資料夾中獲取。
            folder_with_segs_from_previous_stage (str, optional): 來自前一階段分割結果的資料夾。
            clinical_data_dir (str, optional): 預處理後 .csv 檔案的路徑。
                                                如果為 None，則不會加載臨床資料。
        可以使用 self.clinical_df 來訪問臨床資料的 DataFrame。
        """
        super().__init__(folder, identifiers, folder_with_segs_from_previous_stage)
        self.clinical_data_dir = clinical_data_dir
        # 讀取原始 CSV 檔案
        self.clinical_data_label_encoder = ClinicalDataLabelEncoder(self.clinical_data_dir)  # 初始化標籤編碼器
        # 進行預處理 (label encoding)
        self.clinical_full_df = self.clinical_data_label_encoder.forward()  # 讀取並編碼臨床資料
        # 如果有 Case_Index 欄位，則設為索引
        if 'Case_Index' in self.clinical_full_df.columns:
            self.clinical_full_df = self.clinical_full_df.set_index('Case_Index', drop=False)
        # 過濾只保留 identifiers 對應的 row
        if identifiers is not None:
            self.clinical_df = self.clinical_full_df.loc[identifiers]
        else:
            self.clinical_df = self.clinical_full_df


        # 每個特徵 類別的數量
        self.field_specs = {
            'location': self.clinical_data_label_encoder.num_location_classes,
            't_stage':  self.clinical_data_label_encoder.num_t_stage_classes,
            'n_stage':  self.clinical_data_label_encoder.num_n_stage_classes,
            'm_stage':  self.clinical_data_label_encoder.num_m_stage_classes
        }

        # 每個特徵的缺失標記 = 類別數量 - 1
        self.missing_flags = {
            'location': self.clinical_data_label_encoder.missing_flag_location,
            't_stage':  self.clinical_data_label_encoder.missing_flag_t_stage,
            'n_stage':  self.clinical_data_label_encoder.missing_flag_n_stage,
            'm_stage':  self.clinical_data_label_encoder.missing_flag_m_stage
        }


    def load_case(self, identifier):
        """
        加載單個病例的CT影像、Seg Mask 和臨床資料。
        覆寫父類的 load_case 方法以包含臨床資料。
        """
        # 調用父類的 load_case 方法加載影像和標註數據
        # data: 影像的 blosc2 檔案, seg: label的 blosc2 檔案, 
        # seg_prev: cascade模型專用 用不到, properties: 元資訊
        data, seg, seg_prev, properties = super().load_case(identifier)
        # print(f"正在加載病例: {identifier}")
        # print(f"影像數據形狀: {data.shape}, 標註數據形狀: {seg.shape}")

        # 特殊情況 原生nnunet模組使用的 別動!!
        # 不需要臨床資料 所以可以提早回傳
        stack = inspect.stack()
        for frame in stack:
            # 如果是 determine_shapes 呼叫，只回傳 4 個值，因為它只需要影像的形狀信息
            if frame.function == "determine_shapes":
                return data, seg, seg_prev, properties

        # 開始處理臨床影像
        row = self.clinical_df[self.clinical_df['Case_Index'] == identifier]
        if row.empty:
            raise KeyError(f"找不到 Case_Index={identifier} 的臨床資料")

        # 轉成索引 (原始CSV檔才有大寫開頭 後續程式全部小寫!)
        loc_idx = row['Location'].values[0]
        t_idx   = row['T_stage'].values[0]
        n_idx   = row['N_stage'].values[0]
        m_idx   = row['M_stage'].values[0]

        clinical_data_dict = {
            'location': loc_idx, # [1] 純量元素
            't_stage':  t_idx,
            'n_stage':  n_idx,
            'm_stage':  m_idx
        }

        # 4. 製作 mask（True = 有臨床資料）
        clinical_mask_bool = {
            'location': loc_idx != self.missing_flags['location'],
            't_stage':  t_idx   != self.missing_flags['t_stage'],
            'n_stage':  n_idx   != self.missing_flags['n_stage'],
            'm_stage':  m_idx   != self.missing_flags['m_stage']
        }

        return data, seg, seg_prev, properties, clinical_data_dict, clinical_mask_bool


# 為了兼容性，保留原始的 infer_dataset_class，但我們會在 Trainer 中根據需求選擇
def infer_dataset_class_multimodal(folder: str, clinical_data_dir: str = None) -> Union[Type[nnUNetDatasetMultimodal], Type[nnUNetDatasetBlosc2], Type[nnUNetDatasetNumpy]]:
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
    if clinical_data_dir and os.path.isdir(clinical_data_dir):
        print(f"檢測到臨床資料資料夾 '{clinical_data_dir}'，將使用 nnUNetDatasetMultimodal。")
        return nnUNetDatasetMultimodal
    else:
        print(f"未檢測到臨床資料資料夾或路徑無效，將使用基礎 nnUNetDataset 類別 ({base_dataset_class.__name__})。")
        return base_dataset_class
    

if __name__ == "__main__":
    # 測試 nnUNetDatasetMultimodal 是否能正確加載資料
    dataset = nnUNetDatasetMultimodal(
        folder='/mnt/data1/graduate/yuxin/Lab/model/UNet_base/nnunet_ins_data/data_test/nnUNet_preprocessed/Dataset101/nnUNetPlans_3d_fullres',
        clinical_data_dir='/mnt/data1/graduate/yuxin/Lab/model/UNet_base/nnunet_ins_data/data_test/nnUNet_raw/Dataset101'
    )
    data, seg, seg_prev, properties, clinical_data_dict, clinical_mask_bool = dataset.load_case("colon_001")

    print("影像數據形狀:", data.shape)
    print("標註數據形狀:", seg.shape)
    print("臨床資料:", clinical_data_dict)
    print("臨床資料 mask:", clinical_mask_bool)