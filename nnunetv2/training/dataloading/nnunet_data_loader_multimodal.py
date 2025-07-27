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
                 label_manager, # 從 nnunetv2.utilities.label_handling.label_handling 導入
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...]] = None,
                 probabilistic_oversampling: bool = False,
                 transforms=None):
        """
        初始化多模態資料載入器。
        Args:
            data: 資料集實例 (可以是 nnUNetDatasetMultimodal 或其他 nnUNetDataset)。
            其他參數同 nnUNetDataLoader。
        """
        super().__init__(data, batch_size, patch_size, final_patch_size, label_manager,
                         oversample_foreground_percent, sampling_probabilities,
                         pad_sides, probabilistic_oversampling, transforms)
        print("nnUNetDataLoaderMultimodal 初始化完成。")

    def generate_train_batch(self):
        """
        生成一個訓練批次，包括影像數據、標註和臨床資料。
        覆寫父類的 generate_train_batch 方法。
        """
        selected_keys = self.get_indices()

        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)

        # 初始化臨床資料相關的儲存空間
        # 假設 prompt_features 是 17 維，且其他標籤是單一整數
        # 使用列表來收集每個病例的臨床數據
        all_prompt_features = []
        all_location_labels = []
        all_t_stage_labels = []
        all_n_stage_labels = []
        all_m_stage_labels = []
        all_has_clinical_data_flags = []

        for j, i in enumerate(selected_keys):
            force_fg = self.get_do_oversample(j)
            
            # 這裡的 self._data.load_case 會根據實際 Dataset 類型返回不同的結果
            # 如果是 nnUNetDatasetMultimodal，會返回 clinical_features_dict
            # 否則只返回 data, seg, seg_prev, properties
            load_result = self._data.load_case(i)
            
            # 判斷是否返回了臨床資料
            if len(load_result) == 5:
                data, seg, seg_prev, properties, clinical_features_dict = load_result
                # 從字典中提取數據
                prompt_features = clinical_features_dict['prompt_features']
                location_label = clinical_features_dict['location_label']
                t_stage_label = clinical_features_dict['t_stage_label']
                n_stage_label = clinical_features_dict['n_stage_label']
                m_stage_label = clinical_features_dict['m_stage_label']
                has_clinical_data = clinical_features_dict['has_clinical_data']
            else: # 如果不是多模態 Dataset，則沒有臨床資料
                data, seg, seg_prev, properties = load_result
                # 對於沒有臨床資料的病例，設置預設值
                prompt_features = np.zeros(17, dtype=np.float32) # 與 MyModel 的 prompt_dim 匹配
                location_label = -1
                t_stage_label = -1
                n_stage_label = -1
                m_stage_label = -1
                has_clinical_data = False

            # 將臨床數據添加到列表中
            all_prompt_features.append(prompt_features)
            all_location_labels.append(location_label)
            all_t_stage_labels.append(t_stage_label)
            all_n_stage_labels.append(n_stage_label)
            all_m_stage_labels.append(m_stage_label)
            all_has_clinical_data_flags.append(has_clinical_data)

            # 影像裁剪與填充邏輯 (與父類相同)
            shape = data.shape[1:]
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

            data_all[j] = crop_and_pad_nd(data, bbox, 0)
            seg_cropped = crop_and_pad_nd(seg, bbox, -1)

            if seg_prev is not None:
                # 確保 seg_prev 是正確的維度，通常是 (1, D, H, W) 或 (D, H, W)
                # crop_and_pad_nd 返回的形狀會是 (C, D, H, W)
                # 如果 seg_prev 已經是 (C, D, H, W) 而不是 (D, H, W)，則不需要 [None]
                seg_prev_cropped = crop_and_pad_nd(seg_prev, bbox, -1)
                if seg_prev_cropped.ndim == 3: # 假設 seg_prev 也是 (D, H, W)
                    seg_cropped = np.vstack((seg_cropped, seg_prev_cropped[None]))
                else: # 假設 seg_prev 已經是 (C, D, H, W)
                    seg_cropped = np.vstack((seg_cropped, seg_prev_cropped))
                    
            seg_all[j] = seg_cropped

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]

        # 應用數據增強 Transforms (與父類相同)
        if self.transforms is not None:
            with torch.no_grad():
                # from threadpoolctl import threadpool_limits # already imported in nnUNetDataLoader
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

        # 將臨床資料轉換為 PyTorch 張量
        prompt_features_tensor = torch.from_numpy(np.array(all_prompt_features)).float()
        location_labels_tensor = torch.from_numpy(np.array(all_location_labels)).long()
        t_stage_labels_tensor = torch.from_numpy(np.array(all_t_stage_labels)).long()
        n_stage_labels_tensor = torch.from_numpy(np.array(all_n_stage_labels)).long()
        m_stage_labels_tensor = torch.from_numpy(np.array(all_m_stage_labels)).long()
        has_clinical_data_flags_tensor = torch.from_numpy(np.array(all_has_clinical_data_flags)).bool()
        
        # 提取真正的 missing_flags 標籤 (即 prompt_features 中的最後四維)
        # 這些標籤應該是 0 或 1
        missing_flags_labels_tensor = prompt_features_tensor[:, 13:17].long()


        # 構建 MyModel forward 方法所需的輸入字典
        model_input = {
            'data': data_all,
            'clinical_features': prompt_features_tensor,
            'has_clinical_data': has_clinical_data_flags_tensor
        }

        # 返回批次數據，包括影像、標註和臨床資料標籤
        return {
            'data': model_input, # 傳遞給模型的是這個字典
            'target': seg_all, # 標註還是獨立作為 target
            'clinical_labels': { # 臨床資料的真實標籤，用於計算損失和指標
                'location': location_labels_tensor,
                't_stage': t_stage_labels_tensor,
                'n_stage': n_stage_labels_tensor,
                'm_stage': m_stage_labels_tensor,
                'missing_flags': missing_flags_labels_tensor
            },
            'keys': selected_keys
        }