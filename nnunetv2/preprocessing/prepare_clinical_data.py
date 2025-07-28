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
        'ascending': 0, 
        'transverse': 1, 
        'descending': 2,
        'sigmoid': 3, 
        'rectal': 4, 
        'rectosigmoid': 5,
        'Missing': 6 # 固定使用最後一個index當作缺失類別 Tx NONE之類的要提早處理掉 這裡只能應付 Missing或根本沒有特徵
    }
    num_location_classes = len(location_mapping)

    # T-Stage (6 類別)
    t_stage_mapping = {
        'T0': 0, 
        'T1': 1, 
        'T2': 2, 
        'T3': 3,
        'T4': 4,
        'Missing': 5
    }
    num_t_stage_classes = len(t_stage_mapping)

    # N-Stage (4 類別)
    n_stage_mapping = {
        'N0': 0,
        'N1': 1, 
        'N2': 2,
        'Missing': 3
    }
    num_n_stage_classes = len(n_stage_mapping)

    # M-Stage (3 類別)
    m_stage_mapping = {
        'M0': 0,
        'M1': 1,
        'Missing': 2 # 專門的缺失類別
    }
    num_m_stage_classes = len(m_stage_mapping)

    print("開始處理臨床資料...")


    # 用於存放所有病例的處理結果
    all_rows = []

    # 取出每一個病歷資料

    # 準備csv輸出時的欄位名稱，one-hot部分用真實名稱
    location_names = list(location_mapping.keys())
    onehot_names = [f'Location_{name}' for name in location_names]
    feature_names = onehot_names + [
        'T_stage_label', 'N_stage_label', 'M_stage_label',
        'Location_missing_flag', 'T_stage_missing_flag', 'N_stage_missing_flag', 'M_stage_missing_flag'
    ]

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

        # 檢查維度是否正確
        if location_label >= num_location_classes:
            raise ValueError(f"location_label {location_label} 超出範圍，應小於 {num_location_classes}")

        # Location One-Hot
        location_one_hot = np.zeros(num_location_classes, dtype=np.float32) # 初始化一個獨立的子表格
        location_one_hot[location_label] = 1.0                              # 將子表格的對應位置設為 1
        prompt_features[0:num_location_classes] = location_one_hot          # 將子表格填入主表格當中

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
        # 如果所有特徵都缺失(找不到該欄位 或是欄位填入 Missing (不可填入NONE 等其他標示法 會引發邏輯錯誤))，則認為沒有有效臨床資料
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

        # 儲存為 .csv 檔案 (僅供檢查 後續不使用)
        row_dict = {
            'Case_Index': case_id,
        }
        # one-hot
        for i, name in enumerate(onehot_names):
            row_dict[name] = prompt_features[i]
        # 其他特徵
        row_dict['T_stage_label'] = prompt_features[num_location_classes]
        row_dict['N_stage_label'] = prompt_features[num_location_classes + 1]
        row_dict['M_stage_label'] = prompt_features[num_location_classes + 2]
        row_dict['Location_missing_flag'] = prompt_features[num_location_classes + 3]
        row_dict['T_stage_missing_flag'] = prompt_features[num_location_classes + 4]
        row_dict['N_stage_missing_flag'] = prompt_features[num_location_classes + 5]
        row_dict['M_stage_missing_flag'] = prompt_features[num_location_classes + 6]
        row_dict['has_clinical_data'] = has_clinical_data
        all_rows.append(row_dict)

    # 輸出所有處理後的資料到 csv
    out_csv_path = os.path.join(output_folder, 'all_clinical_data.csv')
    pd.DataFrame(all_rows).to_csv(out_csv_path, index=False)

    print(f"臨床資料處理完成，檔案儲存於：{output_folder}")
    print("###############################重要##########################")
    print(f"PROMPT_DIM = {num_location_classes - 1 + 7}，請填入trainer, predictor, model, dataloader, iterator 五個檔案中。")
    print(
        f"以下同樣需要填入 trainer, predictor, model 當中的參數"
        f"self.init_kwargs =\n"
        f"    'input_channels': input_channels,\n"
        f"    'num_classes': num_classes,\n"
        f"    'deep_supervision': deep_supervision,\n"
        f"    'prompt_dim': {num_location_classes - 1 + 7},\n"
        f"    'location_classes': {num_location_classes},\n"
        f"    't_stage_classes': {num_t_stage_classes},\n"
        f"    'n_stage_classes': {num_n_stage_classes},\n"
        f"    'm_stage_classes': {num_m_stage_classes},\n"
        f"    'missing_flags_dim': 4"
    )
    print("###############################重要##########################")

import argparse

def main():
    parser = argparse.ArgumentParser(
        description="nnUNetv2 臨床資料預處理：將原始臨床 csv 轉成 nnUNet 格式的 pkl 與檢查用 csv。\n\n範例：\n  nnunetv2_prepare_clinical_data --csv /path/to/crcCTlist.csv --output /path/to/output_folder",
        # nnunetv2_prepare_clinical_data --csv "${nnUNet_raw}/${DATASET_NAME}/crcCTlist.csv" --output "${nnUNet_preprocessed}/${DATASET_NAME}/clinical_data"
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--csv", required=True,
        help="原始臨床資料 csv 路徑（如 crcCTlist.csv）"
    )
    parser.add_argument(
        "--output", required=True,
        help="預處理後資料儲存資料夾（如 /path/to/clinical_data）"
    )
    args = parser.parse_args()
    # 自動建立 output 資料夾（如果不存在）
    maybe_mkdir_p(args.output)
    prepare_clinical_data_for_nnunet(args.csv, args.output)

if __name__ == '__main__':
    main()