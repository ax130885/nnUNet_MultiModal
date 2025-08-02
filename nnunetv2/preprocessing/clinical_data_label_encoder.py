# nnunetv2/preprocessing/clinical_data_label_encoder.py
import pandas as pd
import os
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles, save_pickle, isfile
# from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed

class ClinicalDataLabelEncoder:
    """
    將臨床資料的標籤轉換為數值編碼。
    包括 Location, T_stage, N_stage, M_stage 的編碼。
    """
    def __init__(self, csv_dir: str):
        """
        初始化標籤編碼器，並讀取原始 CSV 檔案。
        """
        # 設定檔名
        self.input_csv_path = join(csv_dir, 'crcCTlist.csv')
        self.output_csv_path = join(csv_dir, 'crcCTlist_encoded.csv')

        # 建立映射表
        # 固定使用最後一個值當作缺失類別 在 dataset 需要建立缺失遮罩
        self.location_mapping = {
            'ascending': 0, 
            'transverse': 1, 
            'descending': 2,
            'sigmoid': 3, 
            'rectal': 4, 
            'rectosigmoid': 5,
            'Missing': 6
        }
        self.t_stage_mapping = {
            'T0': 0, 
            'T1': 1, 
            'T2': 2, 
            'T3': 3,
            'T4': 4,
            'Missing': 5
        }
        self.n_stage_mapping = {
            'N0': 0,
            'N1': 1, 
            'N2': 2,
            'Missing': 3
        }
        self.m_stage_mapping = {
            'M0': 0,
            'M1': 1,
            'Missing': 2
        }

        # 計算類別數量
        self.num_location_classes = len(self.location_mapping)
        self.num_t_stage_classes = len(self.t_stage_mapping)
        self.num_n_stage_classes = len(self.n_stage_mapping)
        self.num_m_stage_classes = len(self.m_stage_mapping)

        # 設定缺失標記的索引 = 類別數量 - 1
        self.missing_flag_location = self.num_location_classes - 1
        self.missing_flag_t_stage = self.num_t_stage_classes - 1
        self.missing_flag_n_stage = self.num_n_stage_classes - 1
        self.missing_flag_m_stage = self.num_m_stage_classes - 1

        # 建立反向映射表
        self.reverse_location_mapping = {v: k for k, v in self.location_mapping.items()}
        self.reverse_t_stage_mapping = {v: k for k, v in self.t_stage_mapping.items()}
        self.reverse_n_stage_mapping = {v: k for k, v in self.n_stage_mapping.items()}
        self.reverse_m_stage_mapping = {v: k for k, v in self.m_stage_mapping.items()}

    # 進行標籤編碼
    def forward(self) -> pd.DataFrame:
        """
        將 DataFrame 中的臨床資料標籤轉換為數值編碼。
        """
        df = pd.read_csv(self.input_csv_path)

        # 確保每個欄位都是字串類型，填補空值為 'Missing'
        df['Location'] = df['Location'].astype(str).str.strip().replace('', 'Missing')
        df['T_stage'] = df['T_stage'].astype(str).str.strip().replace('', 'Missing')
        df['N_stage'] = df['N_stage'].astype(str).str.strip().replace('', 'Missing')
        df['M_stage'] = df['M_stage'].astype(str).str.strip().replace('', 'Missing')

        # 檢查是否有值不在映射表 key 中，若有則報錯
        for col, mapping, name in [
            ('Location', self.location_mapping, 'Location'),
            ('T_stage', self.t_stage_mapping, 'T_stage'),
            ('N_stage', self.n_stage_mapping, 'N_stage'),
            ('M_stage', self.m_stage_mapping, 'M_stage')
        ]:
            unknown_values = set(df[col].astype(str).str.strip().replace('', 'Missing')) - set(mapping.keys())
            if unknown_values:
                raise ValueError(f"{name} 特徵當中出現: {unknown_values}，不在映射表當中。請檢查原始資料。")

        # 根據 init 設定的對照表進行 label encoding 
        df['Location'] = df['Location'].map(self.location_mapping)
        df['T_stage'] = df['T_stage'].map(self.t_stage_mapping)
        df['N_stage'] = df['N_stage'].map(self.n_stage_mapping)
        df['M_stage'] = df['M_stage'].map(self.m_stage_mapping)

        df.to_csv(self.output_csv_path, index=False)
        return df
    
    # 反向標籤編碼
    def reverse(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        將 DataFrame 中的數值編碼轉換回原始標籤。
        """
        df['Location'] = df['Location'].map(self.reverse_location_mapping)
        df['T_stage'] = df['T_stage'].map(self.reverse_t_stage_mapping)
        df['N_stage'] = df['N_stage'].map(self.reverse_n_stage_mapping)
        df['M_stage'] = df['M_stage'].map(self.reverse_m_stage_mapping)
        return df



 
import argparse

def main(csv_dir: str = None):
    parser = argparse.ArgumentParser(
        description="nnUNetv2 臨床資料預處理：僅做Label Encoding",
        # nnunetv2_prepare_clinical_data --csv "${nnUNet_raw}/${DATASET_NAME}/crcCTlist.csv" --output "${nnUNet_preprocessed}/${DATASET_NAME}/clinical_data"
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--csv_dir", required=False,
        help="原始 csv 資料夾（需要存在 crcCTlist.csv，輸出為 crcCTlist_encoded.csv）"
    )

    args = parser.parse_args()
    if args.csv_dir is not None:
        print("使用指定的 CSV 資料夾:", args.csv_dir)
        csv_dir = args.csv_dir

    if args.csv_dir is None and csv_dir is None:
        raise ValueError("請提供 --csv_dir 參數或設定 csv_dir 變數。")
    

    encoder = ClinicalDataLabelEncoder(csv_dir)
    encoder.forward()
    print(f"編碼成功 保存在: {encoder.output_csv_path}")



if __name__ == '__main__':
    csv_dir = '/home/admin/yuxin/data/Lab/model/UNet_base/nnunet_ins_data/data_test/nnUNet_raw/Dataset101'
    main(csv_dir)
    # main()