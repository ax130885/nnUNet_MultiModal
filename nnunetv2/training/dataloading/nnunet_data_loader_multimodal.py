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
                 clinical_drop_probability: float = 0.1):
        """
        初始化多模態資料載入器。
        Args:
            data: 資料集實例 (可以是 nnUNetDatasetMultimodal 或其他 nnUNetDataset)。
            其他參數同 nnUNetDataLoader。
        """
        # 設定 drop rate
        self.clinical_drop_probability = clinical_drop_probability

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

        # 初始化臨床資料
        clinical_data = {'location': [], 't_stage': [], 'n_stage': [], 'm_stage': []}
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
            
            
            for k in ['location', 't_stage', 'n_stage', 'm_stage']:
                clinical_data[k].append(clinical_data_dict[k])
                clinical_mask[k].append(clinical_mask_bool[k])


                    
        # 2D 特例
        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]


        # 套用影像的 Augmentation Transforms (與父類相同)
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


        # --- 新增：對臨床資料進行資料增強 (隨機 Drop) ---
        # 此時 data_all 和 seg_all 已經是 PyTorch Tensor
        # clinical_data 和 clinical_mask 還是 Python list (原始格式)

        # 1. 將臨床資料 list 轉換為 PyTorch Tensor (用於高效增強)
        clinical_data_tensor = {}
        clinical_mask_tensor = {}
        for key in clinical_data.keys():
            # 確保資料是整數類型，適合索引和比較
            clinical_data_tensor[key] = torch.tensor(clinical_data[key], dtype=torch.long)
            clinical_mask_tensor[key] = torch.tensor(clinical_mask[key], dtype=torch.bool)

        # 2. 執行隨機 Drop (使用 Tensor 操作)
        #    只對原始有效的特徵 (mask=True) 進行 Drop
        for key in clinical_data_tensor.keys():
            batch_size = clinical_data_tensor[key].size(0)
            # 生成隨機數 [B]
            random_numbers = torch.rand(batch_size)
            # 生成 Drop mask: 當 random_number < drop_probability 且 原始 mask 為 True 時，Drop
            drop_mask = (random_numbers < self.clinical_drop_probability) & clinical_mask_tensor[key]
            # 應用 Drop: 將被選中的特徵值替換為對應的缺失標記
            clinical_data_tensor[key] = torch.where(
                drop_mask,
                torch.tensor(self.missing_flags[key], dtype=clinical_data_tensor[key].dtype),
                clinical_data_tensor[key]
            )
            # 更新 mask: 被 Drop 的特徵現在變為無效 (False)
            clinical_mask_tensor[key] = clinical_mask_tensor[key] & (~drop_mask)

        # 3. 將處理後的 Tensor 轉換回 Trainer 期望的格式 (dict of lists)
        #    這是與原始程式碼的主要區別：轉換回來
        clinical_data_final = {k: v.tolist() for k, v in clinical_data_tensor.items()}
        clinical_mask_final = {k: v.tolist() for k, v in clinical_mask_tensor.items()}
        # --- 結束新增 ---

        # 回傳結構
        return {
            'data':          data_all, # [B, C, D, H, W] tensor
            'target':        seg_all, # [B, C, D, H, W] tensor
            'clinical_data': clinical_data_final, # {[B]} dict of lists, {'location': [B], 't_stage': [B], ...}
            'clinical_mask': clinical_mask_final, # {[B]} dict of lists, {'location': [B], 't_stage': [B], ...}
            'keys':          selected_keys # [B] list of identifiers
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
        label_manager=label_manager,  # 假設有一個 label manager
        oversample_foreground_percent=0.5,
        transforms=None  # 假設沒有使用任何 transforms
    )
    # init的時候 會偷偷多加載一筆資料 所以跑完總共會載入 1 + 2 + 2 = 5 筆資料
    # 可以在 dataset 的 Load case 加入 print 檢查 (一次只返回一筆資料)
    batch = data_loader.generate_train_batch()
    print("------------------------------------------------------")
    batch = data_loader.generate_train_batch()
    print("Batch generated successfully:", batch.keys())