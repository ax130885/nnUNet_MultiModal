# nnunetv2/training/dataloading/nnunet_data_loader_multimodal.py

import numpy as np
import torch
from typing import Union, Tuple, List
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from threadpoolctl import threadpool_limits

# 從 nnUNet 原有資料載入器檔案中導入 nnUNetDataLoader
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
# 導入我們剛剛創建的多模態 Dataset
from nnunetv2.training.dataloading.nnunet_dataset_multimodal import nnUNetDatasetMultimodal

class nnUNetDataLoaderMultimodal(nnUNetDataLoader):
    """
    擴展 nnUNetDataLoader，使其能夠為多模態模型生成批次數據。
    這個 DataLoader 會將影像數據和臨床資料組織成 MyModel 所需的字典格式。
    """
    def __init__(self,
                 data: Union[nnUNetDatasetMultimodal, any], # 可以是多模態或原始 Dataset
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...]] = None,
                 probabilistic_oversampling: bool = False,
                 transforms=None,
                 clinical_drop_probability_global: float = 0.35,
                 clinical_drop_probability_column: float = 0.3
                 ):
        """
        初始化多模態資料載入器。
        Args:
            data: 資料集實例 (可以是 nnUNetDatasetMultimodal 或其他 nnUNetDataset)。
            其他參數同 nnUNetDataLoader。
        """
        # 設定 drop rate
        self.clinical_drop_probability_global = clinical_drop_probability_global
        self.clinical_drop_probability_column = clinical_drop_probability_column

        super().__init__(data, batch_size, patch_size, final_patch_size, label_manager,
                         oversample_foreground_percent, sampling_probabilities,
                         pad_sides, probabilistic_oversampling, transforms)
        
        # 確保傳入的 data 是我們自定義的多模態 Dataset
        if not isinstance(data, nnUNetDatasetMultimodal):
             raise TypeError("nnUNetDataLoaderMultimodal 需要 nnUNetDatasetMultimodal 類型的數據集。")
        
        # 儲存 label encoder 和 missing flag，供資料增強時使用
        self.clinical_data_label_encoder = data.clinical_data_label_encoder
        self.missing_flags = data.missing_flags # 這是你提供的字典
        print(f"nnUNetDataLoaderMultimodal 初始化完成。有效類別數量=缺失標記: {self.missing_flags}")

    def generate_text_description(self, clinical_data_dict, clinical_mask_dict):
        """
        根據可用的臨床特徵生成文字描述
        
        Args:
            clinical_data_dict: 包含各特徵值的字典
            clinical_mask_dict: 包含各特徵是否有效的字典
        
        Returns:
            str: 生成的文字描述
        """
        # 獲取反向映射表中的類別名稱
        location_mapping = self.clinical_data_label_encoder.reverse_location_mapping
        t_stage_mapping = self.clinical_data_label_encoder.reverse_t_stage_mapping
        n_stage_mapping = self.clinical_data_label_encoder.reverse_n_stage_mapping
        m_stage_mapping = self.clinical_data_label_encoder.reverse_m_stage_mapping
        
        # 基礎描述
        base_text = "A computerized tomography scan reveals a colorectal cancer"
        
        # 收集有效的特徵描述
        feature_descriptions = []
        
        # Location 描述
        if clinical_mask_dict['location']:
            loc_idx = clinical_data_dict['location']
            location_name = location_mapping[loc_idx]
            if location_name != 'Missing':
                feature_descriptions.append(f"located in the {location_name} region")
        
        # T Stage 描述
        if clinical_mask_dict['t_stage']:
            t_idx = clinical_data_dict['t_stage']
            t_stage_name = t_stage_mapping[t_idx]
            if t_stage_name != 'T0' or t_stage_name != 'Missing':
                feature_descriptions.append(f"with T stage {t_stage_name}")
        
        # N Stage 描述
        if clinical_mask_dict['n_stage']:
            n_idx = clinical_data_dict['n_stage']
            n_stage_name = n_stage_mapping[n_idx]
            if n_stage_name != 'N0' or n_stage_name != 'Missing':
                feature_descriptions.append(f"N stage {n_stage_name}")
        
        # M Stage 描述
        if clinical_mask_dict['m_stage']:
            m_idx = clinical_data_dict['m_stage']
            m_stage_name = m_stage_mapping[m_idx]
            if  m_stage_name != 'M0' or m_stage_name != 'Missing':
                metastasis_desc = "with distant metastasis" if m_stage_name == "M1" else "without distant metastasis"
                feature_descriptions.append(metastasis_desc)
        
        # 組合描述
        if feature_descriptions:
            full_text = base_text + " " + ", ".join(feature_descriptions) + "."
        else:
            full_text = base_text + "."
        
        return full_text

    def generate_train_batch(self):
        """
        生成一個訓練批次，包括影像數據、標註和臨床資料。
        覆寫父類的 generate_train_batch 方法。
        """
        # 取得目前batch的索引列表 例如batch_size=2時，可能是 ['colon_001', 'colon_002']
        selected_keys = self.get_indices()

        # 初始化影像和label
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)

        # 初始化臨床資料（原始 label）
        clinical_data_label = {'location': [], 't_stage': [], 'n_stage': [], 'm_stage': []}
        # 初始化臨床資料 mask（是否有 label）
        clinical_mask = {'location': [], 't_stage': [], 'n_stage': [], 'm_stage': []}

        # j: 第幾次迴圈
        # i: 當前選中的病例識別符 (索引)
        for j, i in enumerate(selected_keys):
            force_fg = self.get_do_oversample(j)
            
            # 從 nnUNetDatasetMultimodal 取得影像、標註、臨床資料和 mask
            data, seg, seg_prev, properties, clinical_data_dict, clinical_mask_bool = self._data.load_case(i)

            # 影像裁剪與填充邏輯 (與父類相同)
            shape = data.shape[1:]
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

            data_all[j] = crop_and_pad_nd(data, bbox, 0)
            seg_cropped = crop_and_pad_nd(seg, bbox, -1)

            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)[None]))
            seg_all[j] = seg_cropped
            
            # 儲存原始臨床資料作為 label（強制轉為 Python int）
            for k in ['location', 't_stage', 'n_stage', 'm_stage']:
                clinical_data_label[k].append(int(clinical_data_dict[k]))
                clinical_mask[k].append(bool(clinical_mask_bool[k]))

        # 2D 特例
        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]

        # 套用影像的 Augmentation Transforms (與父類相同)
        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images

        # --- 新增：對臨床資料進行資料增強 (隨機 Drop) ---
        # 注意：這裡只對「輸入」做 drop，label 保持不變
        clinical_data_aug = {k: v[:] for k, v in clinical_data_label.items()}  # 複製一份
        clinical_mask_input = {k: v[:] for k, v in clinical_mask.items()}        # 複製一份

        # 1. 轉為 Tensor
        clinical_data_tensor = {}
        clinical_mask_tensor = {}
        for key in clinical_data_aug.keys():
            clinical_data_tensor[key] = torch.tensor(clinical_data_aug[key], dtype=torch.long)
            clinical_mask_tensor[key] = torch.tensor(clinical_mask_input[key], dtype=torch.bool)

        batch_size = next(iter(clinical_data_tensor.values())).size(0)

        # 2. 執行隨機 Drop（只影響 input）
        global_random_numbers = torch.rand(batch_size)
        global_drop_mask = (global_random_numbers < self.clinical_drop_probability_global)

        for key in clinical_data_tensor.keys():
            column_random_numbers = torch.rand(batch_size)
            column_drop_mask = (column_random_numbers < self.clinical_drop_probability_column) & clinical_mask_tensor[key]
            drop_mask = (global_drop_mask | column_drop_mask) & clinical_mask_tensor[key]

            clinical_data_tensor[key] = torch.where(
                drop_mask,
                torch.tensor(self.missing_flags[key], dtype=clinical_data_tensor[key].dtype),
                clinical_data_tensor[key]
            )
            # 注意：不修改 clinical_mask_tensor，因為 mask 是原始 label 是否有效的標記

        # 3. 轉回 list 格式
        clinical_data_aug = {k: v.tolist() for k, v in clinical_data_tensor.items()}
        # mask 保持不變，使用原始值
        clinical_mask_final = {k: v for k, v in clinical_mask.items()}

        # --- 結束新增 ---

        # --- 根據 drop 後的特徵生成文字描述 ---
        text_descriptions = []
        for b in range(batch_size):
            # 為每個樣本生成對應的特徵字典和 mask
            sample_clinical_data = {k: clinical_data_aug[k][b] for k in clinical_data_aug.keys()}
            sample_clinical_mask = {}
            
            # 檢查每個特徵是否被 drop（即是否等於缺失標記）
            for k in clinical_data_aug.keys():
                # 如果原本有效且沒有被 drop 為缺失標記，則認為有效
                is_not_dropped = clinical_data_aug[k][b] != self.missing_flags[k]
                sample_clinical_mask[k] = clinical_mask_final[k][b] and is_not_dropped
            
            # 生成文字描述
            text_desc = self.generate_text_description(sample_clinical_data, sample_clinical_mask)
            text_descriptions.append(text_desc)

        # 回傳結構
        return {
            'data': data_all,                    # [B, C, D, H, W] tensor
            'target': seg_all,                   # [B, C, D, H, W] tensor
            'clinical_data_aug': clinical_data_aug,      # 增強後的臨床資料 (模型輸入)
            'clinical_data_label': clinical_data_label,  # 原始臨床資料 (計算loss用)
            'clinical_mask': clinical_mask_final,        # 臨床資料 mask
            'keys': selected_keys,                        # [B] list of identifiers
            'text_descriptions': text_descriptions        # 文字描述列表
        }

if __name__ == "__main__":
    # 測試 nnUNetDataLoaderMultimodal 是否能正確生成批次數據
    dataset = nnUNetDatasetMultimodal(
        folder='/mnt/data1/graduate/yuxin/Lab/model/UNet_base/nnunet_ins_data/data_test/nnUNet_preprocessed/Dataset101/nnUNetPlans_3d_fullres',
        clinical_data_dir='/mnt/data1/graduate/yuxin/Lab/model/UNet_base/nnunet_ins_data/data_test/nnUNet_raw/Dataset101'
    )
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    plans_manager = PlansManager("/mnt/data1/graduate/yuxin/Lab/model/UNet_base/nnunet_ins_data/data_test/nnUNet_preprocessed/Dataset101/nnUNetPlans.json")
    dataset_json = load_json("/mnt/data1/graduate/yuxin/Lab/model/UNet_base/nnunet_ins_data/data_test/nnUNet_preprocessed/Dataset101/dataset.json")
    label_manager = plans_manager.get_label_manager(dataset_json)

    data_loader = nnUNetDataLoaderMultimodal(
        data=dataset,
        batch_size=2,
        patch_size=(128, 128, 128),
        final_patch_size=(128, 128, 128),
        label_manager=label_manager,
        oversample_foreground_percent=0.5,
        transforms=None
    )
    batch = data_loader.generate_train_batch()
    print("------------------------------------------------------")
    batch = data_loader.generate_train_batch()
    print("Batch generated successfully:", batch.keys())
    print("Label (original):", batch['clinical_data_label'])
    print("Input (dropped):", batch['clinical_data_aug'])
    print("Mask (unchanged):", batch['clinical_mask'])
    print("Keys:", batch['keys'])
    print("Text descriptions:", batch['text_descriptions'])