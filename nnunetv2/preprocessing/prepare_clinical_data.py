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
        'Missing': 0, 'NONE': 0, # 'NONE' 可能在某些情況下出現，也視為缺失/未知
        'ascending': 1, 'transverse': 2, 'descending': 3,
        'sigmoid': 4, 'rectal': 5, 'rectosigmoid': 6
    }
    num_location_classes = 7

    # T-Stage (7 類別，合併 T4a, T4b)
    t_stage_mapping = {
        'T0': 0, 'T1': 1, 'T2': 2, 'T3': 3,
        'T4a': 4, 'T4b': 5, 'T4': 5, # 將 T4 合併到 T4b
        'Tx': 6, 'Missing': 6 # 將 Missing 視為 Tx
    }
    num_t_stage_classes = 7

    # N-Stage (5 類別，合併 N1a/b/c 和 N2a/b)
    n_stage_mapping = {
        'N0': 0,
        'N1': 1, 'N1a': 1, 'N1b': 1, 'N1c': 1,
        'N2': 2, 'N2a': 2, 'N2b': 2,
        'Nx': 3,
        'Missing': 4 # 專門的缺失類別
    }
    num_n_stage_classes = 5

    # M-Stage (4 類別，合併 M1a/b/c)
    m_stage_mapping = {
        'M0': 0,
        'M1': 1, 'M1a': 1, 'M1b': 1, 'M1c': 1,
        'Mx': 2,
        'Missing': 3 # 專門的缺失類別
    }
    num_m_stage_classes = 4

    print("開始處理臨床資料...")

    for index, row in df.iterrows():
        case_id = row['Case_Index'] # 例如 colon_001_0000

        # 初始化標籤和特徵
        location_label = location_mapping.get(row['Location'], 0) # 如果找不到，預設為 Missing
        t_stage_label = t_stage_mapping.get(row['T_stage'], 6) # 如果找不到，預設為 Tx
        n_stage_label = n_stage_mapping.get(row['N_stage'], 4) # 如果找不到，預設為 Missing (NONE)
        m_stage_label = m_stage_mapping.get(row['M_stage'], 3) # 如果找不到，預設為 Missing (NONE)

        # 構建 prompt_dim=17 的特徵向量
        prompt_features = np.zeros(17, dtype=np.float32)

        # 7維 Location One-Hot
        location_one_hot = np.zeros(num_location_classes, dtype=np.float32)
        location_one_hot[location_label] = 1.0
        prompt_features[0:7] = location_one_hot

        # 1維 T-stage 數值
        prompt_features[4] = t_stage_label

        # 1維 N-stage 數值
        prompt_features[5] = n_stage_label

        # 1維 M-stage 數值
        prompt_features[6] = m_stage_label

        # 3維 不確定性標記
        prompt_features[7] = 1.0 if row['T_stage'] == 'Tx' else 0.0 # T_uncertainty_flag
        prompt_features[8] = 1.0 if row['N_stage'] == 'Nx' else 0.0 # N_uncertainty_flag
        prompt_features[9] = 1.0 if row['M_stage'] == 'Mx' else 0.0 # M_uncertainty_flag

        # 4維 缺失標記 (從 CSV 中的 'Missing' 判斷)
        prompt_features[10] = 1.0 if row['Location'] == 'Missing' else 0.0 # loc_missing_flag
        prompt_features[11] = 1.0 if row['T_stage'] == 'Missing' else 0.0 # t_stage_missing_flag
        prompt_features[12] = 1.0 if row['N_stage'] == 'Missing' else 0.0 # n_stage_missing_flag
        prompt_features[13] = 1.0 if row['M_stage'] == 'Missing' else 0.0 # m_stage_missing_flag

        # 判斷是否有臨床資料，用於 MyModel 的 has_clinical_data 標記
        # 如果所有數值類別（T, N, M）的標籤都是其 '缺失' 類別 (T:Tx/Missing, N:NONE, M:NONE)
        # 且 Location 也是 Missing/NONE，則認為沒有有效臨床資料
        has_clinical_data = not (
            location_label == 0 and
            t_stage_label == 6 and # Tx/Missing
            n_stage_label == 4 and # Missing (NONE)
            m_stage_label == 3 and # Missing (NONE)
            np.all(prompt_features[10:] == 0) # 確保 uncertainty flags 和 missing flags 也都是 0
        )
        # 更簡單的判斷：只要 prompt_features 不全為 0，就認為有臨床資料
        if np.any(prompt_features != 0):
             has_clinical_data = True
        else:
             has_clinical_data = False


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