
# nnunetv2/training/nnUNetTrainer/nnunet_trainer_multimodal.py

import os
import inspect
import multiprocessing
import numpy as np
import torch
from typing import Tuple, Union, List
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p

from torch import nn
from torch import distributed as dist
from torch.cuda import device_count

# 導入 nnUNet 原有 Trainer 和其他輔助模組
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader # 原始 DataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class as infer_dataset_class_original # 原始 Dataset 推斷
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss # 分割損失
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans # 原始網路構建
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.inference.sliding_window_prediction import compute_gaussian

# 導入我們自訂的多模態模組
from nnunetv2.training.dataloading.nnunet_dataset_multimodal import nnUNetDatasetMultimodal, infer_dataset_class_multimodal # 多模態 Dataset
from nnunetv2.training.dataloading.nnunet_data_loader_multimodal import nnUNetDataLoaderMultimodal # 多模態 DataLoader
from nnunetv2.training.logging.nnunet_logger_multimodal import nnUNetLoggerMultimodal # 多模態 Logger
from nnunetv2.training.loss.focal_loss import FocalLoss # 我們自定義的 Focal Loss

# 導入您的 MyModel
from nnunetv2.training.nnUNetTrainer.multitask_segmentation_model import MyModel

# 計算指標用
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from torch.nn.parallel import DistributedDataParallel as DDP
from torch._dynamo import OptimizedModule


import warnings
from time import sleep








class nnUNetTrainerMultimodal(nnUNetTrainer):

    
    
    """
    擴展 nnUNetTrainer，支援多模態數據（影像 + 臨床資料）訓練，
    並處理多任務損失、評估以及兩階段訓練的邏輯。
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """
        初始化多模態 Trainer。
        Args:
            plans (dict): nnU-Net 的 plans.json 內容。
            configuration (str): 使用的配置名稱 (例如 '3d_fullres')。
            fold (int): 交叉驗證的折數。
            dataset_json (dict): dataset.json 的內容。
            device (torch.device): 訓練設備 (例如 'cuda')。
        """
        super().__init__(plans, configuration, fold, dataset_json, device)

        print("nnUNetTrainerMultimodal 初始化開始...")

        # 定義臨床資料資料夾路徑
        # 假設 Dataset101 的臨床資料儲存在其預處理資料夾下的 'clinical_data' 子資料夾
        # 這裡需要根據 dataset_name_or_id 判斷是 Dataset100 還是 Dataset101
        # 我們在 run_training_entry 會傳入 dataset_name_or_id
        # 為了簡化，在 Trainer 內部，我們將根據 dataset_json 中的 'name' 來判斷
        self.is_stage2_dataset = (self.plans_manager.dataset_name == "Dataset101") # 請替換為您 Dataset101 的完整名稱
        
        self.clinical_data_folder_path = None
        if self.is_stage2_dataset:
            # 確保 nnUNet_preprocessed 變數是從 nnunetv2.paths 正確導入的
            from nnunetv2.paths import nnUNet_preprocessed
            self.clinical_data_folder_path = join(nnUNet_preprocessed, self.plans_manager.dataset_name, 'clinical_data')
            print(f"Dataset101 模式啟用，臨床資料路徑: {self.clinical_data_folder_path}")
        else:
            print("Dataset100 模式啟用，無臨床資料。")

        # 重新初始化 Logger 為多模態版本
        self.logger = nnUNetLoggerMultimodal()

        # 初始化臨床資料相關的損失和評估指標
        self.focal_loss = FocalLoss(gamma=2.0, reduction='mean') # 可以調整 gamma 和 alpha
        self.bce_loss_logits = nn.BCEWithLogitsLoss() # 用於 missing_flags 的二元分類損失
        
        # 記錄臨床指標的歷史
        self.history_clinical_metrics = {
            'loc_acc': [], 't_acc': [], 'n_acc': [], 'm_acc': [], 'missing_flags_acc': []
        }
        self.history_clinical_losses = {
            'loc_loss': [], 't_loss': [], 'n_loss': [], 'm_loss': [], 'missing_flags_loss': []
        }

        # 設置臨床損失的權重，可能需要根據實驗調整
        self.clinical_loss_weights = {
            'location': 0.0,  # 位置分類損失權重
            't_stage': 0.0,
            'n_stage': 0.0,
            'm_stage': 0.0,
            'missing_flags': 0.00 # 權重較低
        }

        # 設定目標權重 (根據你的需求)
        self.target_clinical_loss_weights = {
            'location': 1.0,      # 位置分類損失權重最高
            't_stage': 0.5,       # T 分期權重次之
            'n_stage': 0.2,       # N 分期權重較低
            'm_stage': 0.2,       # M 分期權重較低
            'missing_flags': 0.1  # 缺失標記權重極低
        }

        self.clinical_loss_start_epoch = 500  # 從第 500 個 epoch 開始增加權重
        self.clinical_loss_end_epoch = self.num_epochs   # 到最後一個 epoch 達到目標權重

        print("nnUNetTrainerMultimodal 初始化完成。")

    def initialize(self):
        """
        初始化模型、優化器、學習率排程器、損失函數和資料集類別。
        覆寫父類的 initialize 方法。
        """
        if not self.was_initialized:
            # 調用父類的 _set_batch_size_and_oversample 來處理 DDP 批次大小
            self._set_batch_size_and_oversample()

            # determine_num_input_channels 會自動檢查是否是級聯模型並增加通道數
            # 不需要特殊處理，MyModel 會處理單通道或多通道輸入
            from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
            self.num_input_channels_img = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)
            
            # 使用您的 MyModel 架構
            # MyModel 的 __init__ 參數
            my_model_init_kwargs = {
                'input_channels': self.num_input_channels_img,
                'num_classes': self.label_manager.num_segmentation_heads,
                'deep_supervision': self.enable_deep_supervision,
                'prompt_dim': 14,
                'location_classes': 7,
                't_stage_classes': 6,
                'n_stage_classes': 4,
                'm_stage_classes': 3,
                'missing_flags_dim': 4
            }
            self.network = self.build_network_architecture(MyModel, my_model_init_kwargs).to(self.device)

            # 編譯網路 (如果支援且啟用)
            if self._do_i_compile():
                self.print_to_log_file('使用 torch.compile...')
                self.network = torch.compile(self.network)

            # 配置優化器和學習率排程器
            self.optimizer, self.lr_scheduler = self.configure_optimizers()

            # 如果使用 DDP，則包裝網路
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = nn.parallel.DistributedDataParallel(self.network, device_ids=[self.local_rank])

            # 構建損失函數
            self.loss = self._build_loss()

            # 設定資料集類別
            # 根據是否是 Dataset101 決定使用哪個 Dataset 類別
            if self.is_stage2_dataset and isfile(join(self.clinical_data_folder_path, 'colon_001_0000.pkl')): # 簡單檢查臨床資料是否存在
                self.dataset_class = nnUNetDatasetMultimodal
                print(f" Trainer 將使用 {self.dataset_class.__name__} 資料集類別。")
            else:
                self.dataset_class = infer_dataset_class_original(self.preprocessed_dataset_folder)
                print(f" Trainer 將使用 {self.dataset_class.__name__} 資料集類別。")
            
            self.was_initialized = True
        else:
            raise RuntimeError("Trainer 已經初始化，不應重複調用 initialize。")

    @staticmethod
    def build_network_architecture(*args, **kwargs):
        """
        傳入參數 回傳模型
        取代原生trainer 透過get_network_from_plans產生模型
        我們直接給定固定的模型

        model_class = MyModel
        my_model_init_kwargs = {
                'input_channels': self.num_input_channels_img,
                'num_classes': self.label_manager.num_segmentation_heads,
                'deep_supervision': self.enable_deep_supervision,
                'prompt_dim': 14,
                'location_classes': 7,
                't_stage_classes': 6,
                'n_stage_classes': 4,
                'm_stage_classes': 3,
                'missing_flags_dim': 4
            }
        """
        if len(args) == 2:
            model_class = args[0]
            model_init_kwargs = args[1]
            return model_class(**model_init_kwargs)
        else:
            raise ValueError("build_network_architecture 參數數量不正確")

    def _build_loss(self):
        """
        構建多任務損失函數：分割損失 + 臨床分類損失。
        覆寫父類的 _build_loss 方法。
        """
        # 分割損失 (與父類相同)
        if self.label_manager.has_regions:
            seg_loss = DC_and_BCE_loss({},
                                     {'batch_dice': self.configuration_manager.batch_dice,
                                      'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                     use_ignore_label=self.label_manager.ignore_label is not None)
        else:
            seg_loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                      'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                      ignore_label=self.label_manager.ignore_label)
        
        # 如果啟用深度監督，則包裝分割損失
        if self.enable_deep_supervision:
            from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6 # DDP 和 compile 的特殊處理
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            seg_loss = DeepSupervisionWrapper(seg_loss, weights)
        
        # 這裡我們將返回一個新的複合損失函數
        # 這個複合損失函數在 forward 裡會同時接收分割輸出和臨床輸出，並計算總損失
        # 由於 nnUNetTrainer 的 self.loss 期望是一個直接可調用的損失模組，
        # 我們將定義一個內部類或函數來實現複合損失的邏輯。
        
        # 為了簡化，我們在 train_step 和 validation_step 中手動計算和彙總損失
        # self.loss 變數將只用於影像分割部分，臨床損失則單獨計算。
        return seg_loss

    def get_tr_and_val_datasets(self):
        """
        獲取訓練和驗證資料集。
        覆寫父類的 get_tr_and_val_datasets 方法。
        """
        tr_keys, val_keys = self.do_split()

        # 根據是否是 Stage 2 訓練來傳遞 clinical_data_folder
        if self.is_stage2_dataset:
            dataset_tr = self.dataset_class(self.preprocessed_dataset_folder, tr_keys,
                                             folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                             clinical_data_folder=self.clinical_data_folder_path)
            dataset_val = self.dataset_class(self.preprocessed_dataset_folder, val_keys,
                                              folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                              clinical_data_folder=self.clinical_data_folder_path)
        else:
            # Dataset100 不傳遞 clinical_data_folder
            # 使用 infer_dataset_class_original 確保使用正確的基礎 Dataset 類
            # 因為 self.dataset_class 在 initialize 中可能被設置為 nnUNetDatasetMultimodal
            # 但 Dataset100 實際上沒有 clinical_data_folder，所以 load_case 不會加載
            # 這裡確保實例化的是正確的Dataset，但如果已經是 nnUNetDatasetMultimodal 也沒關係，因為 clinical_data_folder 為 None
            original_dataset_class = infer_dataset_class_original(self.preprocessed_dataset_folder)
            dataset_tr = original_dataset_class(self.preprocessed_dataset_folder, tr_keys,
                                             folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            dataset_val = original_dataset_class(self.preprocessed_dataset_folder, val_keys,
                                              folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
        
        print(f"訓練資料集載入器使用 Dataset 類別: {type(dataset_tr).__name__}")
        print(f"驗證資料集載入器使用 Dataset 類別: {type(dataset_val).__name__}")
        return dataset_tr, dataset_val

    def get_dataloaders(self):
        """
        獲取訓練和驗證資料載入器。
        覆寫父類的 get_dataloaders 方法。
        """
        # 確保 self.dataset_class 已經在 initialize 中被正確設定
        if self.dataset_class is None:
            self.initialize() # 確保 dataset_class 被設定

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        val_transforms = self.get_validation_transforms(deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if
            self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        
        # 根據是否是 Stage 2 訓練來選擇 DataLoader
        DataLoader_Class = nnUNetDataLoaderMultimodal if self.is_stage2_dataset else nnUNetDataLoader

        dl_tr = DataLoader_Class(dataset_tr, self.batch_size,
                                 initial_patch_size,
                                 self.configuration_manager.patch_size,
                                 self.label_manager,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                 probabilistic_oversampling=self.probabilistic_oversampling)

        dl_val = DataLoader_Class(dataset_val, self.batch_size,
                                  self.configuration_manager.patch_size,
                                  self.configuration_manager.patch_size,
                                  self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                  probabilistic_oversampling=self.probabilistic_oversampling)

        allowed_num_processes = get_allowed_n_proc_DA()
        
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            allowed_num_processes = get_allowed_n_proc_DA()
            
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)

        return mt_gen_train, mt_gen_val

    def train_step(self, batch: dict) -> dict:
        """
        執行一個訓練步驟，包括前向傳播、損失計算和反向傳播。
        覆寫父類的 train_step 方法。
        """
        # 從 DataLoader 獲取批次數據
        # batch['data'] 是一個字典，包含 'data' (影像), 'clinical_features', 'has_clinical_data'
        # batch['target'] 是分割標註
        # batch['clinical_labels'] 是一個字典，包含臨床分類的真實標籤
        model_input = batch['data']
        seg_target = batch['target']
        clinical_labels = batch['clinical_labels']

        # 將影像數據和分割標註移動到指定設備
        image_data = model_input['data'].to(self.device, non_blocking=True)
        if isinstance(seg_target, list):
            seg_target = [i.to(self.device, non_blocking=True) for i in seg_target]
        else:
            seg_target = seg_target.to(self.device, non_blocking=True)

        # 將臨床特徵和標籤移動到指定設備
        clinical_features = model_input['clinical_features'].to(self.device, non_blocking=True)
        has_clinical_data = model_input['has_clinical_data'].to(self.device, non_blocking=True)
        loc_label = clinical_labels['location'].to(self.device, non_blocking=True)
        t_label = clinical_labels['t_stage'].to(self.device, non_blocking=True)
        n_label = clinical_labels['n_stage'].to(self.device, non_blocking=True)
        m_label = clinical_labels['m_stage'].to(self.device, non_blocking=True)
        missing_flags_label = clinical_labels['missing_flags'].to(self.device, non_blocking=True)


        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else self.dummy_context():
            # 模型前向傳播：輸入影像和臨床特徵
            # MyModel 返回 seg_outputs, attributes
            seg_outputs, attributes = self.network(image_data, clinical_features)

            # --- 計算分割損失 ---
            seg_loss_val = self.loss(seg_outputs, seg_target)

            # --- 計算臨床分類損失 ---
            # 只有當 has_clinical_data 為 True 的樣本才計算臨床損失
            # 找到有臨床資料的樣本索引
            valid_indices = has_clinical_data
            
            # 初始化臨床損失，如果沒有有效樣本則為 0
            loc_loss_val = torch.tensor(0.0, device=self.device)
            t_loss_val = torch.tensor(0.0, device=self.device)
            n_loss_val = torch.tensor(0.0, device=self.device)
            m_loss_val = torch.tensor(0.0, device=self.device)
            missing_flags_loss_val = torch.tensor(0.0, device=self.device)

            if torch.any(valid_indices):
                # 對於每個臨床屬性，計算 Focal Loss
                # attributes: 預測結果, loc_label: 真實標籤
                loc_loss_val = self.focal_loss(attributes['location'][valid_indices], loc_label[valid_indices])
                t_loss_val = self.focal_loss(attributes['t_stage'][valid_indices], t_label[valid_indices])
                n_loss_val = self.focal_loss(attributes['n_stage'][valid_indices], n_label[valid_indices])
                m_loss_val = self.focal_loss(attributes['m_stage'][valid_indices], m_label[valid_indices])
                
                # 對於 missing_flags，使用 BCEWithLogitsLoss (多標籤二元分類)
                # missing_flags_label 是 (batch_size, 4)
                # attributes['missing_flags'] 也是 (batch_size, 4)
                if attributes['missing_flags'] is not None:
                     # 確保標籤和預測匹配維度且都是 float
                    missing_flags_loss_val = self.bce_loss_logits(
                        attributes['missing_flags'][valid_indices].float(),
                        missing_flags_label[valid_indices].float()
                    )


            # --- 總損失 ---
            total_loss = (
                seg_loss_val +
                self.clinical_loss_weights['location'] * loc_loss_val +
                self.clinical_loss_weights['t_stage'] * t_loss_val +
                self.clinical_loss_weights['n_stage'] * n_loss_val +
                self.clinical_loss_weights['m_stage'] * m_loss_val +
                self.clinical_loss_weights['missing_flags'] * missing_flags_loss_val
            )

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        # 返回所有損失值 (轉移到 CPU 並轉換為 numpy)
        return {
            'loss': total_loss.detach().cpu().numpy(),
            'seg_loss': seg_loss_val.detach().cpu().numpy(),
            'loc_loss': loc_loss_val.detach().cpu().numpy(),
            't_loss': t_loss_val.detach().cpu().numpy(),
            'n_loss': n_loss_val.detach().cpu().numpy(),
            'm_loss': m_loss_val.detach().cpu().numpy(),
            'missing_flags_loss': missing_flags_loss_val.detach().cpu().numpy(),
            'has_clinical_data_count': valid_indices.sum().detach().cpu().numpy() # 統計有效臨床資料樣本數
        }

    def on_train_epoch_end(self, train_outputs: List[dict]):
        """
        在每個訓練 Epoch 結束時彙總結果並記錄。
        覆寫父類的 on_train_epoch_end 方法。
        """
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            # 彙總所有 worker 的損失
            losses_tr_total = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr_total, outputs['loss'])
            loss_here = np.vstack(losses_tr_total).mean()

            # 彙總分割損失
            seg_losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(seg_losses_tr, outputs['seg_loss'])
            seg_loss_here = np.vstack(seg_losses_tr).mean()

            # 彙總其他臨床損失
            loc_losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(loc_losses_tr, outputs['loc_loss'])
            loc_loss_here = np.vstack(loc_losses_tr).mean()
            # 對 T, N, M, missing_flags 損失進行同樣的彙總
            t_losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(t_losses_tr, outputs['t_loss'])
            t_loss_here = np.vstack(t_losses_tr).mean()

            n_losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(n_losses_tr, outputs['n_loss'])
            n_loss_here = np.vstack(n_losses_tr).mean()

            m_losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(m_losses_tr, outputs['m_loss'])
            m_loss_here = np.vstack(m_losses_tr).mean()
            
            missing_flags_losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(missing_flags_losses_tr, outputs['missing_flags_loss'])
            missing_flags_loss_here = np.vstack(missing_flags_losses_tr).mean()

        else:
            loss_here = np.mean(outputs['loss'])
            seg_loss_here = np.mean(outputs['seg_loss'])
            loc_loss_here = np.mean(outputs['loc_loss'])
            t_loss_here = np.mean(outputs['t_loss'])
            n_loss_here = np.mean(outputs['n_loss'])
            m_loss_here = np.mean(outputs['m_loss'])
            missing_flags_loss_here = np.mean(outputs['missing_flags_loss'])


        # 記錄到 logger（全部使用 train_xxx_losses）
        self.logger.log('train_total_losses', loss_here, self.current_epoch)
        self.logger.log('train_seg_losses', seg_loss_here, self.current_epoch)
        self.logger.log('train_loc_losses', loc_loss_here, self.current_epoch)
        self.logger.log('train_t_losses', t_loss_here, self.current_epoch)
        self.logger.log('train_n_losses', n_loss_here, self.current_epoch)
        self.logger.log('train_m_losses', m_loss_here, self.current_epoch)
        self.logger.log('train_missing_flags_losses', missing_flags_loss_here, self.current_epoch)


    def validation_step(self, batch: dict) -> dict:
        """
        執行一個驗證步驟，包括前向傳播、損失計算和指標計算。
        覆寫父類的 validation_step 方法。
        """
        model_input = batch['data']
        seg_target = batch['target']
        clinical_labels = batch['clinical_labels']

        image_data = model_input['data'].to(self.device, non_blocking=True)
        if isinstance(seg_target, list):
            seg_target = [i.to(self.device, non_blocking=True) for i in seg_target]
        else:
            seg_target = seg_target.to(self.device, non_blocking=True)

        clinical_features = model_input['clinical_features'].to(self.device, non_blocking=True)
        has_clinical_data = model_input['has_clinical_data'].to(self.device, non_blocking=True)
        loc_label = clinical_labels['location'].to(self.device, non_blocking=True)
        t_label = clinical_labels['t_stage'].to(self.device, non_blocking=True)
        n_label = clinical_labels['n_stage'].to(self.device, non_blocking=True)
        m_label = clinical_labels['m_stage'].to(self.device, non_blocking=True)
        missing_flags_label = clinical_labels['missing_flags'].to(self.device, non_blocking=True)

        # Autocast 只在 CUDA 上啟用，CPU/MPS 不啟用以避免效能低落或報錯
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else self.dummy_context():
            seg_outputs, attributes = self.network(image_data, clinical_features)
            # 釋放 image_data 記憶體，減少顯存佔用
            del image_data

            # --- 計算分割損失和 Dice ---
            seg_loss_val = self.loss(seg_outputs, seg_target) # 分割損失

            # 如果啟用深度監督，會有多個解析度的輸出（list），只用最高解析度計算 Dice
            if self.enable_deep_supervision:
                output_seg = seg_outputs[0]
                seg_target_for_dice = seg_target[0]
            else:
                output_seg = seg_outputs
                seg_target_for_dice = seg_target

            # 計算偽 Dice（用於日誌），與原始 nnU-Net 邏輯相同
            from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
            axes = [0] + list(range(2, output_seg.ndim))

            if self.label_manager.has_regions:
                predicted_segmentation_onehot = (torch.sigmoid(output_seg) > 0.5).long()
            else:
                # softmax 不需要，直接 argmax
                output_seg_argmax = output_seg.argmax(1)[:, None]
                predicted_segmentation_onehot = torch.zeros(output_seg.shape, device=output_seg.device, dtype=torch.float32)
                predicted_segmentation_onehot.scatter_(1, output_seg_argmax, 1)
                # 釋放 output_seg_argmax 記憶體
                del output_seg_argmax

            # 處理 ignore_label，注意 target 被修改後不要再用原本的 target
            mask = None
            if self.label_manager.has_ignore_label:
                if not self.label_manager.has_regions:
                    mask = (seg_target_for_dice != self.label_manager.ignore_label).float()
                    seg_target_for_dice[seg_target_for_dice == self.label_manager.ignore_label] = 0
                else:
                    if seg_target_for_dice.dtype == torch.bool:
                        mask = ~seg_target_for_dice[:, -1:]
                    else:
                        mask = 1 - seg_target_for_dice[:, -1:]
                    seg_target_for_dice = seg_target_for_dice[:, :-1] # 移除 ignore_label 的 channel

            tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, seg_target_for_dice, axes=axes, mask=mask)
            tp_hard = tp.detach().cpu().numpy()
            fp_hard = fp.detach().cpu().numpy()
            fn_hard = fn.detach().cpu().numpy()

            if not self.label_manager.has_regions:
                # 如果不是 regions 訓練，去掉背景，只保留前景
                tp_hard = tp_hard[1:]
                fp_hard = fp_hard[1:]
                fn_hard = fn_hard[1:]

            # --- 計算臨床分類損失（同 train_step）---
            valid_indices = has_clinical_data

            loc_loss_val = torch.tensor(0.0, device=self.device)
            t_loss_val = torch.tensor(0.0, device=self.device)
            n_loss_val = torch.tensor(0.0, device=self.device)
            m_loss_val = torch.tensor(0.0, device=self.device)
            missing_flags_loss_val = torch.tensor(0.0, device=self.device)

            # --- 計算臨床分類準確率 ---
            loc_acc_val = torch.tensor(0.0, device=self.device)
            t_acc_val = torch.tensor(0.0, device=self.device)
            n_acc_val = torch.tensor(0.0, device=self.device)
            m_acc_val = torch.tensor(0.0, device=self.device)
            missing_flags_acc_val = torch.tensor(0.0, device=self.device) # 平均準確率

            if torch.any(valid_indices):
                # 計算損失
                loc_loss_val = self.focal_loss(attributes['location'][valid_indices], loc_label[valid_indices])
                t_loss_val = self.focal_loss(attributes['t_stage'][valid_indices], t_label[valid_indices])
                n_loss_val = self.focal_loss(attributes['n_stage'][valid_indices], n_label[valid_indices])
                m_loss_val = self.focal_loss(attributes['m_stage'][valid_indices], m_label[valid_indices])

                if attributes['missing_flags'] is not None:
                    missing_flags_loss_val = self.bce_loss_logits(
                        attributes['missing_flags'][valid_indices].float(),
                        missing_flags_label[valid_indices].float()
                    )

                # 計算準確率
                # location, T, N, M 是多分類，使用 argmax
                loc_preds = attributes['location'][valid_indices].argmax(dim=1)
                loc_acc_val = (loc_preds == loc_label[valid_indices]).float().mean()

                t_preds = attributes['t_stage'][valid_indices].argmax(dim=1)
                t_acc_val = (t_preds == t_label[valid_indices]).float().mean()

                n_preds = attributes['n_stage'][valid_indices].argmax(dim=1)
                n_acc_val = (n_preds == n_label[valid_indices]).float().mean()

                m_preds = attributes['m_stage'][valid_indices].argmax(dim=1)
                m_acc_val = (m_preds == m_label[valid_indices]).float().mean()

                # missing_flags 是多標籤二元分類，預測每個 flag 是否存在
                # logits 轉機率再閾值化
                if attributes['missing_flags'] is not None:
                    missing_flags_preds = (torch.sigmoid(attributes['missing_flags'][valid_indices]) > 0.5).long()
                    missing_flags_acc_val = (missing_flags_preds == missing_flags_label[valid_indices]).float().mean() # 對所有標籤求平均準確率

            total_loss = (
                seg_loss_val +
                self.clinical_loss_weights['location'] * loc_loss_val +
                self.clinical_loss_weights['t_stage'] * t_loss_val +
                self.clinical_loss_weights['n_stage'] * n_loss_val +
                self.clinical_loss_weights['m_stage'] * m_loss_val +
                self.clinical_loss_weights['missing_flags'] * missing_flags_loss_val
            )

        # 回傳所有損失和指標（轉到 CPU 並轉 numpy）
        return {
            'loss': total_loss.detach().cpu().numpy(), # 總損失
            'seg_loss': seg_loss_val.detach().cpu().numpy(),
            'loc_loss': loc_loss_val.detach().cpu().numpy(),
            't_loss': t_loss_val.detach().cpu().numpy(),
            'n_loss': n_loss_val.detach().cpu().numpy(),
            'm_loss': m_loss_val.detach().cpu().numpy(),
            'missing_flags_loss': missing_flags_loss_val.detach().cpu().numpy(),
            'loc_acc': loc_acc_val.detach().cpu().numpy(),
            't_acc': t_acc_val.detach().cpu().numpy(),
            'n_acc': n_acc_val.detach().cpu().numpy(),
            'm_acc': m_acc_val.detach().cpu().numpy(),
            'missing_flags_acc': missing_flags_acc_val.detach().cpu().numpy(),
            'tp_hard': tp_hard, # 分割 Dice 統計
            'fp_hard': fp_hard,
            'fn_hard': fn_hard,
            'has_clinical_data_count': valid_indices.sum().detach().cpu().numpy() # 有效臨床資料樣本數
        }

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        """
        在每個驗證 Epoch 結束時彙總結果並記錄。
        覆寫父類的 on_validation_epoch_end 方法。
        """
        outputs_collated = collate_outputs(val_outputs)

        # --- 彙總分割 Dice 指標 ---
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            # 彙總所有 worker 的 tp, fp, fn
            world_size = dist.get_world_size()
            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)
            
            # 彙總所有 worker 的總損失
            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()

            # 彙總分割損失
            seg_losses_val = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(seg_losses_val, outputs_collated['seg_loss'])
            seg_loss_here = np.vstack(seg_losses_val).mean()

            # 彙總所有 worker 的臨床損失
            loc_losses_val = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(loc_losses_val, outputs_collated['loc_loss'])
            loc_loss_here = np.vstack(loc_losses_val).mean()
            # 對 T, N, M, missing_flags 損失進行同樣的彙總
            t_losses_val = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(t_losses_val, outputs_collated['t_loss'])
            t_loss_here = np.vstack(t_losses_val).mean()

            n_losses_val = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(n_losses_val, outputs_collated['n_loss'])
            n_loss_here = np.vstack(n_losses_val).mean()

            m_losses_val = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(m_losses_val, outputs_collated['m_loss'])
            m_loss_here = np.vstack(m_losses_val).mean()
            
            missing_flags_losses_val = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(missing_flags_losses_val, outputs_collated['missing_flags_loss'])
            missing_flags_loss_here = np.vstack(missing_flags_losses_val).mean()

            # 彙總所有 worker 的臨床準確度 (需要加權平均或只取有數據的樣本)
            # 為了簡化，我們將收集所有樣本的準確度，然後平均
            loc_accs_val = [None for _ in range(world_size)]
            dist.all_gather_object(loc_accs_val, outputs_collated['loc_acc'])
            loc_acc_here = np.vstack(loc_accs_val).mean()
            # 對 T, N, M, missing_flags 準確度進行同樣的彙總
            t_accs_val = [None for _ in range(world_size)]
            dist.all_gather_object(t_accs_val, outputs_collated['t_acc'])
            t_acc_here = np.vstack(t_accs_val).mean()

            n_accs_val = [None for _ in range(world_size)]
            dist.all_gather_object(n_accs_val, outputs_collated['n_acc'])
            n_acc_here = np.vstack(n_accs_val).mean()

            m_accs_val = [None for _ in range(world_size)]
            dist.all_gather_object(m_accs_val, outputs_collated['m_acc'])
            m_acc_here = np.vstack(m_accs_val).mean()
            
            missing_flags_accs_val = [None for _ in range(world_size)]
            dist.all_gather_object(missing_flags_accs_val, outputs_collated['missing_flags_acc'])
            missing_flags_acc_here = np.vstack(missing_flags_accs_val).mean()


        else:
            loss_here = np.mean(outputs_collated['loss'])
            seg_loss_here = np.mean(outputs_collated['seg_loss'])
            loc_loss_here = np.mean(outputs_collated['loc_loss'])
            t_loss_here = np.mean(outputs_collated['t_loss'])
            n_loss_here = np.mean(outputs_collated['n_loss'])
            m_loss_here = np.mean(outputs_collated['m_loss'])
            missing_flags_loss_here = np.mean(outputs_collated['missing_flags_loss'])

            loc_acc_here = np.mean(outputs_collated['loc_acc'])
            t_acc_here = np.mean(outputs_collated['t_acc'])
            n_acc_here = np.mean(outputs_collated['n_acc'])
            m_acc_here = np.mean(outputs_collated['m_acc'])
            missing_flags_acc_here = np.mean(outputs_collated['missing_flags_acc'])


        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)

        # 記錄到 logger（全部使用 val_xxx_losses 與 val_xxx_accs）
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)
        self.logger.log('val_seg_losses', seg_loss_here, self.current_epoch)
        self.logger.log('val_loc_losses', loc_loss_here, self.current_epoch)
        self.logger.log('val_t_losses', t_loss_here, self.current_epoch)
        self.logger.log('val_n_losses', n_loss_here, self.current_epoch)
        self.logger.log('val_m_losses', m_loss_here, self.current_epoch)
        self.logger.log('val_missing_flags_losses', missing_flags_loss_here, self.current_epoch)

        self.logger.log('val_loc_accs', loc_acc_here, self.current_epoch)
        self.logger.log('val_t_accs', t_acc_here, self.current_epoch)
        self.logger.log('val_n_accs', n_acc_here, self.current_epoch)
        self.logger.log('val_m_accs', m_acc_here, self.current_epoch)
        self.logger.log('val_missing_flags_accs', missing_flags_acc_here, self.current_epoch)


        # 原本打印指標是在on epoch end，但是因為會延遲變成在下個epoch才印出來
        # 所以提前打印
        if self.local_rank == 0:
            # 額外打印臨床相關指標
            loc_acc = self.logger.my_fantastic_logging['val_loc_accs'][-1] if len(self.logger.my_fantastic_logging['val_loc_accs']) > 0 else "N/A"
            t_acc = self.logger.my_fantastic_logging['val_t_accs'][-1] if len(self.logger.my_fantastic_logging['val_t_accs']) > 0 else "N/A"
            n_acc = self.logger.my_fantastic_logging['val_n_accs'][-1] if len(self.logger.my_fantastic_logging['val_n_accs']) > 0 else "N/A"
            m_acc = self.logger.my_fantastic_logging['val_m_accs'][-1] if len(self.logger.my_fantastic_logging['val_m_accs']) > 0 else "N/A"
            missing_flags_acc = self.logger.my_fantastic_logging['val_missing_flags_accs'][-1] if len(self.logger.my_fantastic_logging['val_missing_flags_accs']) > 0 else "N/A"

            self.print_to_log_file(f"Val Location Acc: {np.round(loc_acc, decimals=4)}", add_timestamp=True)
            self.print_to_log_file(f"Val T Stage Acc: {np.round(t_acc, decimals=4)}", add_timestamp=True)
            self.print_to_log_file(f"Val N Stage Acc: {np.round(n_acc, decimals=4)}", add_timestamp=True)
            self.print_to_log_file(f"Val M Stage Acc: {np.round(m_acc, decimals=4)}", add_timestamp=True)
            self.print_to_log_file(f"Val Missing Flags Acc: {np.round(missing_flags_acc, decimals=4)}", add_timestamp=True)

            # 更新最佳 EMA pseudo Dice 的邏輯已經在父類 on_epoch_end 中處理
            # 如果需要根據臨床指標決定 best checkpoint，需要修改父類的邏輯或在這裡添加額外判斷



    def on_epoch_end(self):
        """
        在每個 Epoch 結束時記錄時間戳，並處理檢查點儲存。
        覆寫父類的 on_epoch_end 方法。
        """
        # 更新臨床損失權重
        self.update_clinical_loss_weights()

        super().on_epoch_end() # 調用父類方法處理時間戳、Dice 記錄和檢查點儲存


    def update_clinical_loss_weights(self):
        """
        根據當前 epoch 動態更新臨床損失權重。
        在 start_epoch 之前權重為 0。
        在 start_epoch 到 end_epoch 之間線性增長至目標權重。
        在 end_epoch 之後保持目標權重。
        """
        current_epoch = self.current_epoch
        start_epoch = self.clinical_loss_start_epoch
        end_epoch = self.clinical_loss_end_epoch

        if current_epoch < start_epoch:
            # Epoch 小於起始 epoch，權重保持為 0
            # self.clinical_loss_weights 已經初始化為 0，無需更改
            pass
        elif current_epoch >= start_epoch and current_epoch <= end_epoch:
            # 在線性增長區間內
            # 計算增長比例
            progress_ratio = (current_epoch - start_epoch) / (end_epoch - start_epoch)
            # 對每個權重項進行線性插值
            for key in self.clinical_loss_weights.keys():
                target_weight = self.target_clinical_loss_weights.get(key, 0.0)
                # 使用線性插值：current_weight = start_weight + ratio * (target_weight - start_weight)
                # start_weight 是 0
                self.clinical_loss_weights[key] = 0.0 + progress_ratio * (target_weight - 0.0)
        else:
            # Epoch 大於結束 epoch，權重設為目標權重
            self.clinical_loss_weights = self.target_clinical_loss_weights.copy()

        # 可選：打印當前權重（僅在 rank 0 進程，避免重複打印）
        if self.local_rank == 0:
            # 每隔一定 epoch 打印一次，或只在權重開始變化時打印
            if current_epoch <= start_epoch or current_epoch % 20 == 0 or current_epoch == end_epoch + 1:
                self.print_to_log_file(f"Epoch {current_epoch}: 更新臨床損失權重為 {self.clinical_loss_weights}")


    def set_deep_supervision_enabled(self, enabled: bool):
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if hasattr(mod, 'set_deep_supervision'):
            mod.set_deep_supervision(enabled)
        else:
            mod.decoder.deep_supervision = enabled

    def perform_actual_validation(self, save_probabilities: bool = False):
        """
        執行實際的驗證 (validation)，包括影像分割和臨床屬性預測的評估。
        此方法會載入驗證案例、使用 Predictor 進行預測，並計算所有相關指標。
        
        Args:
            save_probabilities (bool): 是否儲存預測的機率圖。
        """
        print("開始perform_actual_validation階段...")
        from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
        from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
        from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
        from nnunetv2.paths import nnUNet_preprocessed
        from nnunetv2.inference.nnunet_predictor_multimodal import nnUNetPredictorMultimodal 
        from nnunetv2.training.dataloading.nnunet_dataset_multimodal import infer_dataset_class_multimodal 
        from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
        from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot
        
        # 停用深度監督，因為驗證時通常只需要最終輸出
        self.set_deep_supervision_enabled(False)

        # 將網路設定為評估模式
        self.network.eval()

        # 處理一個已知的 torch.compile 問題的警告 (保持原樣)
        if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
            self.print_to_log_file("警告！訓練時批次大小為 1 且啟用了 torch.compile。如果你 "
                                    "在驗證時遇到崩潰，那是因為 torch.compile 忘記了 "
                                    "觸發重新編譯以停用深度監督。"
                                    "這會導致 torch.flip 抱怨收到元組作為輸入。只需重新執行 "
                                    "驗證並加上 --val (與之前完全相同) 然後它就會運作。"
                                    "為什麼？因為 --val 觸發 nnU-Net 只執行驗證，表示第一次 "
                                    "前向傳播 (觸發編譯的地方) 已經停用了深度監督。"
                                    "這正是我們在 perform_actual_validation 中需要的")

        # 建立 nnUNetPredictorMultimodal 物件用於滑動視窗預測
        predictor = nnUNetPredictorMultimodal(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False
        )
        print("nnUNetPredictorMultimodal 已初始化")

        # 手動初始化 predictor，傳入訓練好的網路和相關設定
        # 這裡的 self.network 已經是 MyModel，會被正確傳遞
        predictor.manual_initialization(
            self.network, 
            self.plans_manager,
            self.configuration_manager,
            None, # parameters: 在這裡不需要多折集成，因為我們只評估當前訓練的模型
            self.dataset_json,
            self.__class__.__name__,
            self.inference_allowed_mirroring_axes
        )

        # 建立多行程池用於匯出分割結果
        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            # 獲取驗證集鍵列表
            _, val_keys = self.do_split()

            # 如果是 DDP (分散式數據並行) 模式，則將驗證鍵分配給不同的進程
            if self.is_ddp:
                last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1
                val_keys = val_keys[self.local_rank:: dist.get_world_size()]

            # 建立驗證資料集物件 (確保使用正確的 Dataset 類別和 clinical_data_folder)
            if self.is_stage2_dataset:
                # Stage 2 訓練：使用多模態 Dataset 並傳入臨床資料路徑
                dataset_val = self.dataset_class(
                    self.preprocessed_dataset_folder,
                    val_keys,
                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                    clinical_data_folder=self.clinical_data_folder_path
                )
            else:
                # Stage 1 訓練：使用原始 Dataset 類別 (不傳遞臨床資料路徑)
                dataset_val = self.dataset_class( 
                    self.preprocessed_dataset_folder,
                    val_keys,
                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
                )

            next_stages = self.configuration_manager.next_stage_names
            if next_stages is not None:
                # 為每個下一個階段建立預測結果的資料夾
                _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

            results = [] # 用於儲存異步操作的結果物件

            # --- 初始化臨床指標累計變數 ---
            # 這些列表將儲存所有有效案例的真實標籤和預測標籤，以便在所有案例處理完畢後進行整體計算
            all_true_loc_labels = []
            all_pred_loc_labels = []
            all_true_t_labels = []
            all_pred_t_labels = []
            all_true_n_labels = []
            all_pred_n_labels = []
            all_true_m_labels = []
            all_pred_m_labels = []
            
            # 對於 missing_flags，處理多標籤問題，儲存 NumPy 陣列
            all_true_missing_flags = []
            all_pred_missing_flags = []

            total_valid_clinical_cases_processed = 0 # 統計實際處理的、有臨床資料的案例數

            # 迭代驗證集中的每個案例
            for i, k in enumerate(dataset_val.identifiers):
                # 檢查 worker 是否忙碌，以避免佇列過長
                proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                          allowed_num_queued=2)
                while not proceed:
                    sleep(0.1) # 等待一段時間
                    proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                              allowed_num_queued=2)
                
                self.print_to_log_file(f"正在預測案例: {k}")

                # 載入案例資料：現在 `dataset_val.load_case(k)` 會返回 5 個值，如果 Dataset 是 Multimodal
                load_result = dataset_val.load_case(k)

                image_data_np = None
                seg_prev_np = None
                properties = None
                clinical_features_dict = None

                # 判斷返回結果的長度，以區分是否為多模態 Dataset
                if len(load_result) == 5:
                    image_data_np, _, seg_prev_np, properties, clinical_features_dict = load_result
                else: 
                    # 如果不是多模態 Dataset (例如 Dataset100)，則沒有臨床資料
                    image_data_np, _, seg_prev_np, properties = load_result
                    # 為臨床資料字典設定預設值，以確保程式碼能繼續執行
                    clinical_features_dict = {
                        'prompt_features': np.zeros(14, dtype=np.float32), # 匹配 MyModel 的 prompt_dim
                        'location_label': -1,
                        't_stage_label': -1,
                        'n_stage_label': -1,
                        'm_stage_label': -1,
                        'has_clinical_data': False
                    }

                # 將 Blosc2 數據 (如果適用) 轉換為 NumPy 陣列
                image_data_np = image_data_np[:]

                # 如果是級聯訓練，則處理前一階段的分割結果
                if self.is_cascaded:
                    seg_prev_np = seg_prev_np[:]
                    # 將前一階段結果轉換為 one-hot 編碼並與輸入資料串接
                    image_data_np = np.vstack((image_data_np, convert_labelmap_to_one_hot(
                        seg_prev_np, self.label_manager.foreground_labels, output_dtype=image_data_np.dtype
                    )))
                
                # 忽略 'The given NumPy array is not writable' 警告，並將 NumPy 陣列轉換為 PyTorch 張量
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    image_data_tensor = torch.from_numpy(image_data_np)
                
                # 處理臨床特徵為 PyTorch 張量
                clinical_features_tensor = torch.from_numpy(clinical_features_dict['prompt_features']).float()
                # 確保 clinical_features_tensor 具有批次維度 (即使是單個案例，批次大小也是 1)
                if clinical_features_tensor.ndim == 1:
                    clinical_features_tensor = clinical_features_tensor[None] # 添加批次維度 (1, prompt_dim)

                self.print_to_log_file(f'案例 {k}, 影像形狀 {image_data_tensor.shape}, 當前 rank {self.local_rank}')
                
                output_filename_truncated = join(validation_output_folder, k)

                # **使用 nnUNetPredictorMultimodal 執行滑動視窗預測**
                # 它會同時返回分割的 logits 和臨床屬性預測的 logits
                prediction_seg_logits, prediction_attr_logits = predictor.predict_sliding_window_return_logits(
                    image_data_tensor, clinical_features_tensor
                )
                prediction_seg_logits = prediction_seg_logits.cpu() # 將分割結果移動到 CPU

                # --- 處理臨床屬性預測結果並累計 ---
                # 只有當案例實際包含臨床資料時才進行處理
                if clinical_features_dict['has_clinical_data']:
                    total_valid_clinical_cases_processed += 1
                    
                    # 獲取真實標籤
                    true_loc = clinical_features_dict['location_label']
                    true_t = clinical_features_dict['t_stage_label']
                    true_n = clinical_features_dict['n_stage_label']
                    true_m = clinical_features_dict['m_stage_label']
                    # # 最終驗證不用 missing_flags 輔助，不需要預測
                    # true_missing_flags = clinical_features_dict['missing_flags']

                    # 將臨床屬性 logits 轉換為預測標籤 (argmax for multi-class, sigmoid+threshold for binary)
                    pred_loc = prediction_attr_logits['location'].argmax(dim=-1).item()
                    pred_t = prediction_attr_logits['t_stage'].argmax(dim=-1).item()
                    pred_n = prediction_attr_logits['n_stage'].argmax(dim=-1).item()
                    pred_m = prediction_attr_logits['m_stage'].argmax(dim=-1).item()

                    # 累計真實和預測標籤到列表中，以便後面計算整體指標
                    all_true_loc_labels.append(true_loc)
                    all_pred_loc_labels.append(pred_loc)
                    all_true_t_labels.append(true_t)
                    all_pred_t_labels.append(pred_t)
                    all_true_n_labels.append(true_n)
                    all_pred_n_labels.append(pred_n)
                    all_true_m_labels.append(true_m)
                    all_pred_m_labels.append(pred_m)
                    # # 最終驗證不用 missing_flags 輔助，不需要預測
                    # all_true_missing_flags.append(true_missing_flags) 
                    # all_pred_missing_flags.append(pred_missing_flags) 

                # 將分割預測結果的匯出工作放入背景行程池中執行 (非同步)
                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_logits, # 導出函數
                        ((prediction_seg_logits, properties, self.configuration_manager, self.plans_manager,
                          self.dataset_json, output_filename_truncated, save_probabilities),)
                    )
                )

                # 如果有下一個階段 (級聯模型)，則匯出 softmax 預測供下一階段使用
                if next_stages is not None:
                    for n in next_stages:
                        next_stage_config_manager = self.plans_manager.get_configuration(n)
                        expected_preprocessed_folder = join(nnUNet_preprocessed, self.plans_manager.dataset_name,
                                                            next_stage_config_manager.data_identifier)
                        
                        # 推斷下一個階段的資料集類別 (使用 multimodal_infer 確保正確性)
                        dataset_class_next_stage = infer_dataset_class_multimodal(
                            expected_preprocessed_folder, 
                            clinical_data_folder=self.clinical_data_folder_path if self.is_stage2_dataset else None
                        )

                        try:
                            # 載入下一階段的原始影像數據以獲取目標形狀
                            tmp_dataset = dataset_class_next_stage(
                                expected_preprocessed_folder,
                                [k], # 只需要載入這一個案例
                                clinical_data_folder=self.clinical_data_folder_path if self.is_stage2_dataset else None 
                            )
                            d_next_stage_info = tmp_dataset.load_case(k) # 載入數據，可能返回 4 或 5 個元素
                            # 取影像數據部分來獲取形狀
                            if isinstance(d_next_stage_info, dict): # 如果是多模態數據的預處理結果
                                target_shape = d_next_stage_info['data'].shape[1:]
                            else: # 如果是單模態的預處理結果
                                target_shape = d_next_stage_info.shape[1:]
                            
                        except FileNotFoundError:
                            self.print_to_log_file(
                                f"預測下一個階段 {n} 對案例 {k} 失敗，因為找不到預處理檔案！"
                                f"請先執行此設定的預處理！")
                            continue

                        output_folder = join(self.output_folder_base, 'predicted_next_stage', n)
                        output_file_truncated = join(output_folder, k)

                        # 將重新取樣和儲存的工作放入背景行程池中執行 (非同步)
                        results.append(segmentation_export_pool.starmap_async(
                            resample_and_save, ( # 執行函數
                                (prediction_seg_logits, target_shape, output_file_truncated, self.plans_manager,
                                 self.configuration_manager,
                                 properties,
                                 self.dataset_json,
                                 default_num_processes,
                                 dataset_class_next_stage), # 函數參數
                            )
                        ))

                # 如果是 DDP 且不是最後一個屏障索引，且每 20 個案例設置一次屏障 (避免 NCCL 超時)
                if self.is_ddp and i < last_barrier_at_idx and (i + 1) % 20 == 0:
                    dist.barrier()

            # 等待所有非同步匯出工作完成
            _ = [r.get() for r in results]

            # 如果是 DDP，則在所有 rank 間設置最終屏障
            if self.is_ddp:
                dist.barrier()
            
            # **只有 Rank 0 執行後續的指標計算和列印**
            if self.local_rank == 0:
                # --- 彙總所有 worker 的臨床資料 (DDP 模式下) ---
                if self.is_ddp:
                    # 使用 dist.all_gather_object 收集所有 GPU 的真實標籤和預測標籤
                    # 注意：all_gather_object 需要物件是可序列化的，list 沒問題，numpy array 需要 tolist()
                    all_true_loc_labels_gathered = [None for _ in range(dist.get_world_size())]
                    dist.all_gather_object(all_true_loc_labels_gathered, all_true_loc_labels)
                    all_true_loc_labels = [item for sublist in all_true_loc_labels_gathered for item in sublist]

                    all_pred_loc_labels_gathered = [None for _ in range(dist.get_world_size())]
                    dist.all_gather_object(all_pred_loc_labels_gathered, all_pred_loc_labels)
                    all_pred_loc_labels = [item for sublist in all_pred_loc_labels_gathered for item in sublist]

                    # 對其他屬性也進行同樣的收集
                    all_true_t_labels_gathered = [None for _ in range(dist.get_world_size())]
                    dist.all_gather_object(all_true_t_labels_gathered, all_true_t_labels)
                    all_true_t_labels = [item for sublist in all_true_t_labels_gathered for item in sublist]

                    all_pred_t_labels_gathered = [None for _ in range(dist.get_world_size())]
                    dist.all_gather_object(all_pred_t_labels_gathered, all_pred_t_labels)
                    all_pred_t_labels = [item for sublist in all_pred_t_labels_gathered for item in sublist]

                    all_true_n_labels_gathered = [None for _ in range(dist.get_world_size())]
                    dist.all_gather_object(all_true_n_labels_gathered, all_true_n_labels)
                    all_true_n_labels = [item for sublist in all_true_n_labels_gathered for item in sublist]

                    all_pred_n_labels_gathered = [None for _ in range(dist.get_world_size())]
                    dist.all_gather_object(all_pred_n_labels_gathered, all_pred_n_labels)
                    all_pred_n_labels = [item for sublist in all_pred_n_labels_gathered for item in sublist]

                    all_true_m_labels_gathered = [None for _ in range(dist.get_world_size())]
                    dist.all_gather_object(all_true_m_labels_gathered, all_true_m_labels)
                    all_true_m_labels = [item for sublist in all_true_m_labels_gathered for item in sublist]

                    all_pred_m_labels_gathered = [None for _ in range(dist.get_world_size())]
                    dist.all_gather_object(all_pred_m_labels_gathered, all_pred_m_labels)
                    all_pred_m_labels = [item for sublist in all_pred_m_labels_gathered for item in sublist]
                    
                    # # 對 missing_flags 進行收集 (需要將 numpy array 轉換為 list，再轉回 numpy)
                    # all_true_missing_flags_gathered = [None for _ in range(dist.get_world_size())]
                    # dist.all_gather_object(all_true_missing_flags_gathered, [arr.tolist() for arr in all_true_missing_flags])
                    # all_true_missing_flags = [np.array(item) for sublist in all_true_missing_flags_gathered for item in sublist]

                    # all_pred_missing_flags_gathered = [None for _ in range(dist.get_world_size())]
                    # dist.all_gather_object(all_pred_missing_flags_gathered, [arr.tolist() for arr in all_pred_missing_flags])
                    # all_pred_missing_flags = [np.array(item) for sublist in all_pred_missing_flags_gathered for item in sublist]

                    total_valid_clinical_cases_processed_gathered = [None for _ in range(dist.get_world_size())]
                    dist.all_gather_object(total_valid_clinical_cases_processed_gathered, total_valid_clinical_cases_processed)
                    total_valid_clinical_cases_processed = sum(total_valid_clinical_cases_processed_gathered)
                
                # --- 計算臨床指標並記錄到 Logger ---
                if total_valid_clinical_cases_processed > 0:
                    self.print_to_log_file("\n--- 臨床屬性驗證結果 ---")
                    
                    # 從模型中獲取各屬性的類別數量 (如果模型已編譯，需要訪問 _orig_mod)
                    if isinstance(self.network, (DDP, OptimizedModule)):
                        model_base = self.network.module if isinstance(self.network, DDP) else self.network._orig_mod
                        model_init_kwargs = model_base.init_kwargs
                    else:
                        model_init_kwargs = self.network.init_kwargs # 假設 MyModel 有 init_kwargs 屬性

                    # 定義要評估的臨床屬性及其對應的真實標籤、預測標籤和總類別數
                    attrs_to_evaluate = {
                        'location': (all_true_loc_labels, all_pred_loc_labels, model_init_kwargs.get('location_classes', 7)),
                        't_stage': (all_true_t_labels, all_pred_t_labels, model_init_kwargs.get('t_stage_classes', 6)),
                        'n_stage': (all_true_n_labels, all_pred_n_labels, model_init_kwargs.get('n_stage_classes', 4)),
                        'm_stage': (all_true_m_labels, all_pred_m_labels, model_init_kwargs.get('m_stage_classes', 3)),
                    }

                    for attr_type, (true_labels, pred_labels, num_classes) in attrs_to_evaluate.items():
                        if len(true_labels) > 0: # 確保該屬性有有效的案例數據
                            acc = accuracy_score(true_labels, pred_labels)
                            # F1-score, 精確率 (Precision), 召回率 (Recall) 使用 'weighted' 平均，處理類別不平衡
                            f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
                            precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
                            recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
                            
                            # 計算混淆矩陣
                            cm = confusion_matrix(true_labels, pred_labels, labels=list(range(num_classes)))

                            # 計算每個類別的準確率
                            per_class_metrics_dict = {}
                            for c in range(num_classes):
                                true_idx_c = np.where(np.array(true_labels) == c)
                                if len(true_idx_c) > 0:
                                    correct_pred_c = np.sum(np.array(pred_labels)[true_idx_c] == c)
                                    class_acc = correct_pred_c / len(true_idx_c)
                                else:
                                    class_acc = 0.0 
                                per_class_metrics_dict[str(c)] = {'accuracy': class_acc} # 只記錄 accuracy

                            # 記錄到 logger
                            self.logger.log(f'{attr_type}_accs', acc, self.current_epoch)
                            self.logger.log('clinical_metrics', {
                                attr_type: {
                                    'precision': precision,
                                    'recall': recall,
                                    'f1': f1,
                                    'per_class_metrics': per_class_metrics_dict 
                                }
                            }, self.current_epoch)
                            self.logger.log('confusion_matrices', {f'{attr_type}_confusion_matrix': cm.tolist()}, self.current_epoch)

                            # 記錄類別分佈 (真實 vs. 預測)
                            unique_true, counts_true = np.unique(true_labels, return_counts=True)
                            true_dist_dict = dict(zip(unique_true, counts_true))
                            true_counts_all_classes = [true_dist_dict.get(c, 0) for c in range(num_classes)]

                            unique_pred, counts_pred = np.unique(pred_labels, return_counts=True)
                            pred_dist_dict = dict(zip(unique_pred, counts_pred))
                            pred_counts_all_classes = [pred_dist_dict.get(c, 0) for c in range(num_classes)]

                            self.logger.log('class_distributions', {f'true_{attr_type}_counts': true_counts_all_classes}, self.current_epoch)
                            self.logger.log('class_distributions', {f'pred_{attr_type}_counts': pred_counts_all_classes}, self.current_epoch)
                            
                            self.print_to_log_file(f"- {attr_type} 準確率 (Accuracy): {np.round(acc, decimals=4)}", add_timestamp=False)
                            self.print_to_log_file(f"- {attr_type} F1 分數 (F1 Score): {np.round(f1, decimals=4)}", add_timestamp=False)

                    # 處理 Missing Flags (多標籤二元分類)
                    if len(all_true_missing_flags) > 0:
                        all_true_missing_flags_flat = np.vstack(all_true_missing_flags) # 堆疊成 (N, 4)
                        all_pred_missing_flags_flat = np.vstack(all_pred_missing_flags) # 堆疊成 (N, 4)

                        # 計算總體準確率 (所有標籤的平均)
                        missing_acc_overall = np.mean(all_true_missing_flags_flat == all_pred_missing_flags_flat)
                        
                        self.logger.log('missing_flags_acc', missing_acc_overall, self.current_epoch)
                        self.logger.log('clinical_metrics', {
                            'missing_flags': {
                                'accuracy_overall': missing_acc_overall 
                            }
                        }, self.current_epoch)
                        self.print_to_log_file(f"- 缺失標記總體準確率 (Missing Flags Overall Accuracy): {np.round(missing_acc_overall, decimals=4)}", add_timestamp=False)
                    
                    # 記錄臨床資料的有效案例比例
                    self.logger.log('epoch_clinical_valid_ratio', total_valid_clinical_cases_processed / len(val_keys), self.current_epoch)
                    self.print_to_log_file(f"- 臨床資料有效比例 (Clinical Valid Ratio): {np.round(total_valid_clinical_cases_processed / len(val_keys), decimals=4)}", add_timestamp=False)
                else:
                    self.print_to_log_file("\n無有效臨床資料案例進行驗證。跳過臨床指標計算。", add_timestamp=False)

                # 計算分割指標 (與父類 nnUNetTrainer 的邏輯相同)
                metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                    validation_output_folder,
                                                    join(validation_output_folder, 'summary.json'),
                                                    self.plans_manager.image_reader_writer_class(),
                                                    self.dataset_json["file_ending"],
                                                    self.label_manager.foreground_regions if self.label_manager.has_regions else
                                                    self.label_manager.foreground_labels,
                                                    self.label_manager.ignore_label, chill=True,
                                                    num_processes=default_num_processes * dist.get_world_size() if
                                                    self.is_ddp else default_num_processes)
                self.print_to_log_file("分割驗證完成", also_print_to_console=True)
                self.print_to_log_file("平均驗證 Dice: ", (metrics['foreground_mean']["Dice"]),
                                        also_print_to_console=True)

            self.set_deep_supervision_enabled(True) # 重新啟用深度監督
            compute_gaussian.cache_clear() # 清理高斯核快取