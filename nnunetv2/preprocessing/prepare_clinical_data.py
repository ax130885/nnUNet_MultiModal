# nnunetv2/preprocessing/prepare_clinical_data.py
# 在訓練進行前 需要手動運行此腳本來處理臨床資料
# 訓練過程當中 不會使用到這個腳本

import pandas as pd
import numpy as np
import os
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles, save_pickle, isfile

def prepare_clinical_data_for_nnunet(csv_file: str, output_folder: str):
    """
    從 CSV 檔案中讀取臨床資料，進行編碼和預處理，並為每個病例儲存獨立的 .pkl 檔案。

    Args:
        csv_file (str): 包含臨床資料的 CSV 檔案路徑 (例如 crcCTlist.csv)。
        output_folder (str): 儲存預處理後臨床資料的資料夾路徑。
    """
    df = pd.read_csv(csv_file)

    # 確保輸出資料夾存在
    maybe_mkdir_p(output_folder)

    # 定義各屬性的映射表
    # Location (7 類別)
    location_mapping = {
        'ascending': 0, 'transverse': 1, 'descending': 2,
        'sigmoid': 3, 'rectal': 4, 'rectosigmoid': 5,
        'Missing': 6, 'NONE': 6 # 固定使用最後一個index當作缺失類別
    }
    num_location_classes = len(location_mapping)

    # T-Stage (6 類別)
    t_stage_mapping = {
        'T0': 0, 'T1': 1, 'T2': 2, 'T3': 3,
        'T4a': 4, 'T4b': 4, 'T4': 4, # 將 T4 合併到 T4b
        'Tx': 5, 'Missing': 5 # 將 Missing 視為 Tx
    }
    num_t_stage_classes = len(t_stage_mapping)

    # N-Stage (4 類別)
    n_stage_mapping = {
        'N0': 0,
        'N1': 1, 'N1a': 1, 'N1b': 1, 'N1c': 1,
        'N2': 2, 'N2a': 2, 'N2b': 2,
        'Nx': 3,
        'Missing': 3 # 專門的缺失類別
    }
    num_n_stage_classes = len(n_stage_mapping)

    # M-Stage (3 類別)
    m_stage_mapping = {
        'M0': 0,
        'M1': 1, 'M1a': 1, 'M1b': 1, 'M1c': 1,
        'Mx': 2,
        'Missing': 2 # 專門的缺失類別
    }
    num_m_stage_classes = len(m_stage_mapping)

    print("開始處理臨床資料...")

    # 取出每一個病歷資料
    for index, row in df.iterrows():
        case_id = row['Case_Index'] # 例如 colon_001_0000

        # 根據上面設定的 mapping 字典，使用get(key) 取得 value 
        # 其中 key 是表格中的原始字串，value 是 mapping 後的 index
        # 如果找不到該特徵 設index為缺失類別對應的index
        location_label = location_mapping.get(row['Location'], num_location_classes - 1) # get(key, default) 如果找不到key對應的value，則返回 default
        t_stage_label = t_stage_mapping.get(row['T_stage'], num_t_stage_classes - 1)
        n_stage_label = n_stage_mapping.get(row['N_stage'], num_n_stage_classes - 1) 
        m_stage_label = m_stage_mapping.get(row['M_stage'], num_m_stage_classes - 1)

        # 初始化最後保存的表格全為0
        # 表格維度為 17
        # location 進行 one-hot 編碼 (num_location_classes)(7)
        # TNM進行label編碼(3)，獨立的Location+TNM的缺失旗標(4)
        prompt_features = np.zeros(num_location_classes + 7, dtype=np.float32)

        # Location One-Hot
        location_one_hot = np.zeros(num_location_classes - 1, dtype=np.float32) # 初始化一個獨立的子表格
        location_one_hot[location_label] = 1.0                              # 將子表格的對應位置設為 1
        prompt_features[0:num_location_classes - 1] = location_one_hot          # 將子表格填入主表格當中

        # 1維 T-stage 數值
        prompt_features[num_location_classes] = t_stage_label

        # 1維 N-stage 數值
        prompt_features[num_location_classes + 1] = n_stage_label

        # 1維 M-stage 數值
        prompt_features[num_location_classes + 2] = m_stage_label

        # 4維 缺失標記
        prompt_features[num_location_classes + 3] = 1.0 if (location_label == num_location_classes - 1) else 0.0 # loc_missing_flag
        prompt_features[num_location_classes + 4] = 1.0 if (t_stage_label == num_t_stage_classes - 1) else 0.0 # t_stage_missing_flag
        prompt_features[num_location_classes + 5] = 1.0 if (n_stage_label == num_n_stage_classes - 1) else 0.0 # n_stage_missing_flag
        prompt_features[num_location_classes + 6] = 1.0 if (m_stage_label == num_m_stage_classes - 1) else 0.0 # m_stage_missing_flag

        # 判斷是否有臨床資料，用於 MyModel 的 has_clinical_data 標記
        # 如果所有特徵都缺失(找不到該欄位 或是欄位填入 Missing, NONE)，則認為沒有有效臨床資料
        # has_clinical_data 是一個 boolean 值，True才參與訓練，False則不參與訓練
        has_clinical_data = not (
            location_label == num_location_classes - 1 and
            t_stage_label == num_t_stage_classes - 1 and
            n_stage_label == num_n_stage_classes - 1 and
            m_stage_label == num_m_stage_classes - 1
        )

        # 儲存為 .pkl 檔案
        output_data = {
            'prompt_features': prompt_features,
            'location_label': location_label,
            't_stage_label': t_stage_label,
            'n_stage_label': n_stage_label,
            'm_stage_label': m_stage_label,
            'has_clinical_data': has_clinical_data
        }
        save_pickle(output_data, join(output_folder, f"{case_id}.pkl"))

    print(f"臨床資料處理完成，檔案儲存於：{output_folder}")

if __name__ == '__main__':
    # 這裡您需要根據您的實際路徑設定
    # 並且將在 nnUNet_preprocessed/DatasetXXX_ColonCancer/clinical_data 下儲存處理後的資料
    
    # 請根據您實際的 Dataset101 名稱進行調整
    # 例如，如果 Dataset101 的完整名稱是 Dataset101_CRCStage
    dataset_name_101 = "Dataset101" # 假設您的 Dataset101 完整名稱
    
    # 設定nnunet環境變數
    os.environ['nnUNet_raw'] = '/home/admin/yuxin/data/Lab/model/UNet_base/nnunet_ins_data/data_test/nnUNet_raw'
    os.environ['nnUNet_preprocessed'] = '/home/admin/yuxin/data/Lab/model/UNet_base/nnunet_ins_data/data_test/nnUNet_preprocessed'
        
    # 原始的 crcCTlist.csv 路徑
    crc_ct_list_csv = "/home/admin/yuxin/data/Lab/model/UNet_base/nnunet_ins_data/data_test/nnUNet_raw/Dataset101/crcCTlist.csv"

    # nnUNet 的預處理資料根目錄，從 nnunetv2.paths 導入
    from nnunetv2.paths import nnUNet_preprocessed
    
    # Dataset101 的預處理資料夾路徑
    dataset101_preprocessed_folder = join(nnUNet_preprocessed, dataset_name_101)
    
    # 臨床資料的輸出資料夾，我們在 Dataset101 的預處理資料夾下建立一個子資料夾
    output_clinical_data_folder = join(dataset101_preprocessed_folder, 'clinical_data')
    
    # 運行預處理
    prepare_clinical_data_for_nnunet(crc_ct_list_csv, output_clinical_data_folder)

    # 驗證部分資料
    print("\n驗證部分資料範例：")
    sample_case_id = "colon_001_0000"
    sample_data_path = join(output_clinical_data_folder, f"{sample_case_id}.pkl")
    if isfile(sample_data_path):
        from batchgenerators.utilities.file_and_folder_operations import load_pickle
        sample_data = load_pickle(sample_data_path)
        print(f"病例 {sample_case_id} 的臨床資料：")
        print(f"  prompt_features: {sample_data['prompt_features']}")
        print(f"  location_label: {sample_data['location_label']}")
        print(f"  t_stage_label: {sample_data['t_stage_label']}")
        print(f"  n_stage_label: {sample_data['n_stage_label']}")
        print(f"  m_stage_label: {sample_data['m_stage_label']}")
        print(f"  has_clinical_data: {sample_data['has_clinical_data']}")
    else:
        print(f"未找到病例 {sample_case_id} 的資料。")