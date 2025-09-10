# nnunetv2/training/nnUNetTrainer/nnunet_trainer_multimodal.py

from collections import Counter
import os
import inspect
import multiprocessing
import numpy as np
import pandas as pd
import torch
from typing import Tuple, Union, List
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from nnunetv2.utilities.crossval_split import generate_crossval_split

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
from nnunetv2.preprocessing.clinical_data_label_encoder import ClinicalDataLabelEncoder

# 導入您的 MyModel
from nnunetv2.training.nnUNetTrainer.multitask_segmentation_model import MyModel
from nnunetv2.training.nnUNetTrainer.multitask_model import MyMultiModel


# 計算指標用
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from torch.nn.parallel import DistributedDataParallel as DDP
from torch._dynamo import OptimizedModule

import csv
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
        # 假設 Dataset101 的臨床資料儲存在 nnUNet_raw/Dataset101/crcCTlist.csv
        # 這裡需要根據 dataset_name_or_id 判斷是 Dataset100 還是 Dataset101
        # 我們在 run_training_entry 會傳入 dataset_name_or_id
        # 為了簡化，在 Trainer 內部，我們將根據 dataset_json 中的 'name' 來判斷
        self.is_stage2_dataset = (self.plans_manager.dataset_name == "Dataset101")
        
        self.clinical_data_dir = None
        if self.is_stage2_dataset:
            # 確保 nnUNet_preprocessed 變數是從 nnunetv2.paths 正確導入的
            from nnunetv2.paths import nnUNet_preprocessed
            from nnunetv2.paths import nnUNet_raw
            self.clinical_data_dir = join(nnUNet_raw, self.plans_manager.dataset_name)
            if self.local_rank == 0:
                print(f"Dataset101 模式啟用，臨床資料路徑: {self.clinical_data_dir}")
        else:
            if self.local_rank == 0:
                print("Dataset100 模式啟用，無臨床資料。")

        # 重新初始化 Logger 為多模態版本
        self.logger = nnUNetLoggerMultimodal()
        if self.local_rank == 0:
            print(f"logger種類為: {type(self.logger)}")
            print("nnUNetTrainerMultimodal 初始化完成。")



    def initialize(self):
        """
        初始化模型、優化器、學習率排程器、損失函數和資料集類別。
        覆寫父類的 initialize 方法。
        """
        if not self.was_initialized:
            # 調用父類的 _set_batch_size_and_oversample 來處理 DDP 批次大小
            self._set_batch_size_and_oversample()

            # 設置臨床損失權重的控制點
            # 格式：{ 任務名稱: [(epoch數, 權重值), ...], ... }
            # 控制點按 epoch 升序排列，每個任務至少需要一個控制點
            self.clinical_loss_weight_schedule = {
                'location': [
                    (0, 0.1),
                    (5, 0.2),
                    (int(self.num_epochs * 0.3), 0.8),
                    (self.num_epochs - 1, 2.0)
                ],
                't_stage': [
                    (0, 0.075),
                    (5, 0.15),
                    (int(self.num_epochs * 0.3), 0.6),
                    (self.num_epochs - 1, 1.5)
                ],
                'n_stage': [
                    (0, 0.02),
                    (5, 0.04),
                    (int(self.num_epochs * 0.3), 0.2),
                    (self.num_epochs - 1, 0.6)
                ],
                'm_stage': [
                    (0, 0.02),
                    (5, 0.04),
                    (int(self.num_epochs * 0.3), 0.2),
                    (self.num_epochs - 1, 0.6)
                ]
            }
            
            # 當前使用的權重 (初始化為初始值)
            self.clinical_loss_manual_weights = {
                task: schedule[0][1] for task, schedule in self.clinical_loss_weight_schedule.items()
            }
            
            # 初始化 grad norm EMA
            self.ema_grad_norm = {
                'seg': 0.0,
                'location': 0.0,
                't_stage': 0.0,
                'n_stage': 0.0,
                'm_stage': 0.0
            }

            # 初始化 grad norm 最終權重
            self.grad_norm_factors = {
                'seg': 1.0, 
                'location': 1.0, 
                't_stage': 1.0, 
                'n_stage': 1.0, 
                'm_stage': 1.0
            }

            # grad norm 設定
            self.ema_alpha = 0.95  # EMA 平滑係數，較高的值表示更平滑的移動平均
            self.last_grad_norm_update_epoch = -1  # 上次更新梯度範數的 epoch
            
            # determine_num_input_channels 會自動檢查是否是級聯模型並增加通道數
            # 不需要特殊處理，反正不會用到cascade模型 只是為了避免報錯
            from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
            self.num_input_channels_img = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)
            
            # 使用您的 MyMultiModel 架構
            # MyMultiModel 的 __init__ 參數
            my_model_init_kwargs = {
                'input_channels': self.num_input_channels_img,
                'num_classes': self.label_manager.num_segmentation_heads,
                'deep_supervision': self.enable_deep_supervision,
                'clinical_csv_dir': self.clinical_data_dir,  # 臨床資料資料夾
            }
            self.network = self.build_network_architecture(MyMultiModel, my_model_init_kwargs).to(self.device)

            # 編譯網路 (如果支援且啟用)
            if self._do_i_compile():
                if self.local_rank == 0:
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
            if self.is_stage2_dataset:
                self.dataset_class = nnUNetDatasetMultimodal
                if self.local_rank == 0:
                    print(f" Trainer 將使用 {self.dataset_class.__name__} 資料集類別。")
            else:
                self.dataset_class = infer_dataset_class_original(self.preprocessed_dataset_folder)
                if self.local_rank == 0:
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
        # 每層的權重會是 [1, 0.5, 0.25 ...] 等等
        # 最深層的權重 需要是 0 否則會有問題
        if self.enable_deep_supervision:
            from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            # 如果是 ddp 並且沒有 compile，則最後一個權重設為 1e-6
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

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.json file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.json file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class_original(self.preprocessed_dataset_folder)

        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            case_identifiers = self.dataset_class.get_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            if self.is_stage2_dataset:
                # 如果是 Stage 2 的資料集，則使用 nnUNetDatasetMultimodal
                dataset = nnUNetDatasetMultimodal(self.preprocessed_dataset_folder,
                                                  identifiers=None,
                                                  folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                                  clinical_data_dir=self.clinical_data_dir)
            else:
                # 如果是 Stage 1 的資料集，則使用原始的 nnUNetDataset
                dataset = self.dataset_class(self.preprocessed_dataset_folder,
                                            identifiers=None,
                                            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                all_keys_sorted = list(np.sort(list(dataset.identifiers)))
                splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
                save_json(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file(f"The split file contains {len(splits)} splits.")

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(dataset.identifiers))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            if any([i in val_keys for i in tr_keys]):
                self.print_to_log_file('WARNING: Some validation cases are also in the training set. Please check the '
                                       'splits.json or ignore if this is intentional.')
        return tr_keys, val_keys


    def get_tr_and_val_datasets(self):
        """
        獲取訓練和驗證資料集。
        覆寫父類的 get_tr_and_val_datasets 方法。
        """
        
        tr_keys, val_keys = self.do_split()

        # 根據是否是 Stage 2 訓練來傳遞 clinical_data
        if self.is_stage2_dataset:
            dataset_tr = self.dataset_class(self.preprocessed_dataset_folder, tr_keys,
                                             folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                             clinical_data_dir=self.clinical_data_dir)
            dataset_val = self.dataset_class(self.preprocessed_dataset_folder, val_keys,
                                              folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                              clinical_data_dir=self.clinical_data_dir)
            
            # 統計臨床資料的類別數量 同時計算focal loss的權重
            clinical_df = dataset_tr.clinical_df
            clinical_df_tr = clinical_df.loc[tr_keys]
            clinical_valid_class_counts = {}
            clinical_class_weights = {}
            # 定義每個欄位的 missing flag（通常是最大類別編號）
            missing_flags = {
                "Location": dataset_tr.clinical_data_label_encoder.missing_flag_location,
                "T_stage": dataset_tr.clinical_data_label_encoder.missing_flag_t_stage,
                "N_stage": dataset_tr.clinical_data_label_encoder.missing_flag_n_stage,
                "M_stage": dataset_tr.clinical_data_label_encoder.missing_flag_m_stage,
            }
            # 定義每個欄位的有效類別數（不含 missing flag）
            num_classes_dict = missing_flags

            for col in ["Location", "T_stage", "N_stage", "M_stage"]:
                value_counts = clinical_df_tr[col].value_counts().sort_index()
                missing_flag = missing_flags[col]
                value_counts_no_missing = {k: v for k, v in value_counts.items() if k != missing_flag}
                clinical_valid_class_counts[col] = value_counts_no_missing
                if self.local_rank == 0:
                    print(f"[FocalLoss統計] {col} 各類別數量(不含missing): {value_counts_no_missing}")
                weights = {k: 1.0/(v+1e-6) for k, v in value_counts_no_missing.items()}
                max_w = max(weights.values()) if weights else 1.0
                weights_norm = {k: w/max_w for k, w in weights.items()}
                if self.local_rank == 0:
                    print(f"[FocalLoss權重] {col} (已正規化): {weights_norm}")
                num_classes = num_classes_dict[col]
                # 權重 list，順序為 0,1,...,num_classes-1，missing_flag 欄位補 0.0
                alpha = []
                for i in range(num_classes):
                    if i == missing_flag:
                        alpha.append(0.0)
                    else:
                        alpha.append(weights_norm.get(i, 0.0))
                clinical_class_weights[col] = alpha
                if self.local_rank == 0:
                    print(f"[FocalLoss權重list] {col}: {alpha}")
            
            
            # print("訓練集 T_stage 分布:", clinical_df_tr['T_stage'].value_counts().sort_index())
            self.clinical_class_weights = clinical_class_weights

            # focal loss 臨床特徵用
            self.focal_loss_loc = FocalLoss(alpha=self.clinical_class_weights["Location"], gamma=2.0, reduction='masked_mean')
            self.focal_loss_t = FocalLoss(alpha=self.clinical_class_weights["T_stage"], gamma=2.0, reduction='masked_mean')
            self.focal_loss_n = FocalLoss(alpha=self.clinical_class_weights["N_stage"], gamma=2.0, reduction='masked_mean')
            self.focal_loss_m = FocalLoss(alpha=self.clinical_class_weights["M_stage"], gamma=2.0, reduction='masked_mean')

            # ce loss 臨床特徵用
            self.loc_weight = torch.tensor(self.clinical_class_weights["Location"], device=self.device)
            self.t_weight = torch.tensor(self.clinical_class_weights["T_stage"], device=self.device)
            self.n_weight = torch.tensor(self.clinical_class_weights["N_stage"], device=self.device)
            self.m_weight = torch.tensor(self.clinical_class_weights["M_stage"], device=self.device)
            if self.local_rank == 0:
                print(f"Location 權重張量: {self.loc_weight}")
                print(f"T_stage 權重張量: {self.t_weight}")
                print(f"N_stage 權重張量: {self.n_weight}")
                print(f"M_stage 權重張量: {self.m_weight}")

            # 設置交叉熵損失 (CE Loss)
            self.ce_loss_loc = nn.CrossEntropyLoss(weight=self.loc_weight, reduction='mean')
            self.ce_loss_t = nn.CrossEntropyLoss(weight=self.t_weight, reduction='mean')
            self.ce_loss_n = nn.CrossEntropyLoss(weight=self.n_weight, reduction='mean')
            self.ce_loss_m = nn.CrossEntropyLoss(weight=self.m_weight, reduction='mean')


            # 將 focal loss 套用深度監督包裝器 (直接計算所有解析度輸出的loss)
            if self.enable_deep_supervision:
                from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
                deep_supervision_scales = self._get_deep_supervision_scales()
                weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
                if self.is_ddp and not self._do_i_compile():
                    weights[-1] = 1e-6 # DDP 和 compile 的特殊處理
                else:
                    weights[-1] = 0
                weights = weights / weights.sum()
                self.focal_loss_loc = DeepSupervisionWrapper(self.focal_loss_loc, weights)
                self.focal_loss_t = DeepSupervisionWrapper(self.focal_loss_t, weights)
                self.focal_loss_n = DeepSupervisionWrapper(self.focal_loss_n, weights)
                self.focal_loss_m = DeepSupervisionWrapper(self.focal_loss_m, weights)


            # 將 ce loss 套用深度監督包裝器 (直接計算所有解析度輸出的loss)
            if self.enable_deep_supervision:
                from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
                deep_supervision_scales = self._get_deep_supervision_scales()
                weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
                if self.is_ddp and not self._do_i_compile():
                    weights[-1] = 1e-6 # DDP 和 compile 的特殊處理
                else:
                    weights[-1] = 0
                weights = weights / weights.sum()
                self.ce_loss_loc = DeepSupervisionWrapper(self.ce_loss_loc, weights)
                self.ce_loss_t = DeepSupervisionWrapper(self.ce_loss_t, weights)
                self.ce_loss_n = DeepSupervisionWrapper(self.ce_loss_n, weights)
                self.ce_loss_m = DeepSupervisionWrapper(self.ce_loss_m, weights)

        else:
            # Dataset100 不傳遞 clinical_data_dir
            # 使用 infer_dataset_class_original 確保使用正確的基礎 Dataset 類
            # 因為 self.dataset_class 在 initialize 中可能被設置為 nnUNetDatasetMultimodal
            # 但 Dataset100 實際上沒有 clinical_data_dir，所以 load_case 不會加載
            # 這裡確保實例化的是正確的Dataset，但如果已經是 nnUNetDatasetMultimodal 也沒關係，因為 clinical_data_dir 為 None
            original_dataset_class = infer_dataset_class_original(self.preprocessed_dataset_folder)
            dataset_tr = original_dataset_class(self.preprocessed_dataset_folder, tr_keys,
                                             folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            dataset_val = original_dataset_class(self.preprocessed_dataset_folder, val_keys,
                                              folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
        if self.local_rank == 0:
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

        # 設定 patch_size 和深度監督
        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # 設定資料增強
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

        # 抓CPU有幾核心
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


    def on_train_epoch_start(self):
        self.network.train()
        # self.lr_scheduler.step()  # 在每個 epoch 開始時更新學習率
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)



    def train_step(self, batch: dict) -> dict:
        """
        執行一個訓練步驟，包括前向傳播、損失計算和反向傳播。
        覆寫父類的 train_step 方法。
        """
        # 從 DataLoader 獲取批次數據
        # return {
        #     'data': data_all,                    # [B, C, D, H, W] tensor
        #     'target': seg_all,                   # [B, C, D, H, W] tensor
        #     'clinical_data_aug': clinical_data_aug,      # 增強後的臨床資料 (模型輸入)
        #     'clinical_data_label': clinical_data_label,  # 原始臨床資料 (計算loss用)
        #     'clinical_mask': clinical_mask,              # 臨床資料 mask
        #     'keys': selected_keys                        # [B] list of identifiers
        # }

        img_data = batch['data']
        seg_target = batch['target']
        clinical_data_aug = batch['clinical_data_aug']      # 增強後的資料 (用於模型輸入)
        clinical_data_label = batch['clinical_data_label']  # 原始資料 (用於計算loss)
        clinical_mask = batch['clinical_mask']              # mask
        keys = batch['keys']

        # 將影像數據和seg label移動到指定設備
        image_data = img_data.to(self.device, non_blocking=True)
        if isinstance(seg_target, list): # 如果有多種分割目標器官
            seg_target = [i.to(self.device, non_blocking=True) for i in seg_target]
        else: # 如果只有一種分割目標
            seg_target = seg_target.to(self.device, non_blocking=True)

        # 將臨床特徵(模型輸入)移動到指定設備
        loc_input = torch.tensor(clinical_data_aug['location']).to(self.device, non_blocking=True)
        t_input = torch.tensor(clinical_data_aug['t_stage']).to(self.device, non_blocking=True)
        n_input = torch.tensor(clinical_data_aug['n_stage']).to(self.device, non_blocking=True)
        m_input = torch.tensor(clinical_data_aug['m_stage']).to(self.device, non_blocking=True)

        # 將臨床標籤(用於計算loss)移動到指定設備
        loc_label = torch.tensor(clinical_data_label['location']).to(self.device, non_blocking=True)
        t_label = torch.tensor(clinical_data_label['t_stage']).to(self.device, non_blocking=True)
        n_label = torch.tensor(clinical_data_label['n_stage']).to(self.device, non_blocking=True)
        m_label = torch.tensor(clinical_data_label['m_stage']).to(self.device, non_blocking=True)

        # Missing特徵的遮罩 用於計算損失時忽略
        loc_mask = torch.tensor(clinical_mask['location']).to(self.device, non_blocking=True)
        t_mask = torch.tensor(clinical_mask['t_stage']).to(self.device, non_blocking=True)
        n_mask = torch.tensor(clinical_mask['n_stage']).to(self.device, non_blocking=True)
        m_mask = torch.tensor(clinical_mask['m_stage']).to(self.device, non_blocking=True)


        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else self.dummy_context():
            # 模型前向傳播：輸入影像和臨床特徵
            # MyMultiModel 返回 seg_out, cli_out
            seg_out, cli_out = self.network(image_data, loc_input, t_input, n_input, m_input)

            # --- 計算分割損失 ---
            seg_loss_tr = self.loss(seg_out, seg_target)

             # 對於每個臨床屬性，計算 Focal Loss
            # cli_out: 預測結果, loc_label: 真實標籤 (使用原始未增強的標籤計算loss)
            if self.enable_deep_supervision:
                # # focal loss
                # loc_loss_tr = self.focal_loss_loc(
                #     cli_out['location'],           # list of tensor，每個分支一個 tensor
                #     [loc_label] * len(cli_out['location']),  # list，每個分支都用同一個 label tensor
                #     [loc_mask] * len(cli_out['location'])    # list，每個分支都用同一個 mask tensor
                # )
                # t_loss_tr = self.focal_loss_t(
                #     cli_out['t_stage'],
                #     [t_label] * len(cli_out['t_stage']),
                #     [t_mask] * len(cli_out['t_stage'])
                # )
                # n_loss_tr = self.focal_loss_n(
                #     cli_out['n_stage'],
                #     [n_label] * len(cli_out['n_stage']),
                #     [n_mask] * len(cli_out['n_stage'])
                # )
                # m_loss_tr = self.focal_loss_m(
                #     cli_out['m_stage'],
                #     [m_label] * len(cli_out['m_stage']),
                #     [m_mask] * len(cli_out['m_stage'])
                # )

                # ce loss
                loc_loss_tr = self.ce_loss_loc(
                    cli_out['location'],
                    [loc_label] * len(cli_out['location']),
                )
                t_loss_tr = self.ce_loss_t(
                    cli_out['t_stage'],
                    [t_label] * len(cli_out['t_stage']),
                )
                n_loss_tr = self.ce_loss_n(
                    cli_out['n_stage'],
                    [n_label] * len(cli_out['n_stage']),
                )
                m_loss_tr = self.ce_loss_m(
                    cli_out['m_stage'],
                    [m_label] * len(cli_out['m_stage']),
                )


            else:
                # # focal loss
                # loc_loss_tr = self.focal_loss_loc(cli_out['location'], loc_label, loc_mask)
                # t_loss_tr = self.focal_loss_t(cli_out['t_stage'], t_label, t_mask)
                # n_loss_tr = self.focal_loss_n(cli_out['n_stage'], n_label, n_mask)
                # m_loss_tr = self.focal_loss_m(cli_out['m_stage'], m_label, m_mask)

                # ce loss
                loc_loss_tr = self.ce_loss_loc(cli_out['location'], loc_label)
                t_loss_tr = self.ce_loss_t(cli_out['t_stage'], t_label)
                n_loss_tr = self.ce_loss_n(cli_out['n_stage'], n_label)
                m_loss_tr = self.ce_loss_m(cli_out['m_stage'], m_label)

            # 只在每個 epoch 開始時計算梯度範數以減少計算負擔
            if self.current_epoch != self.last_grad_norm_update_epoch:
                # 計算每個任務的 grad norm
                seg_grad_norm = self._calculate_grad_norm(seg_loss_tr, "seg")
                loc_grad_norm = self._calculate_grad_norm(loc_loss_tr, "location")
                t_grad_norm = self._calculate_grad_norm(t_loss_tr, "t_stage")
                n_grad_norm = self._calculate_grad_norm(n_loss_tr, "n_stage")
                m_grad_norm = self._calculate_grad_norm(m_loss_tr, "m_stage")
                
                # 更新 grad norm 的 EMA
                self._update_ema_grad_norm('seg', seg_grad_norm) # 更新 self.ema_grad_norm[key]
                self._update_ema_grad_norm('location', loc_grad_norm)
                self._update_ema_grad_norm('t_stage', t_grad_norm)
                self._update_ema_grad_norm('n_stage', n_grad_norm)
                self._update_ema_grad_norm('m_stage', m_grad_norm)
                
                # 防止除以零
                eps = 1e-8
                
                # 計算權重 (將 grad norm 的 ema 進行倒數)
                if self.ema_grad_norm['seg'] >= eps:
                    # 分割任務是主要任務，使用它的梯度範數作為基準
                    seg_norm = max(self.ema_grad_norm['seg'], eps)
                    
                    # 根據梯度範數比例計算調整因子
                    for key in ['location', 't_stage', 'n_stage', 'm_stage']:
                        task_norm = max(self.ema_grad_norm[key], eps)
                        # 倒數
                        norm_ratio = seg_norm / task_norm
                        # 限制權重上限
                        self.grad_norm_factors[key] = min(norm_ratio, 3.0)



                # 更新最後計算梯度範數的 epoch
                self.last_grad_norm_update_epoch = self.current_epoch

                # 輸出梯度範數和調整因子 (方便調試)
                if self.local_rank == 0:
                    self.print_to_log_file(f"Epoch {self.current_epoch} - Grad Norms: seg={seg_grad_norm:.4f}, "
                                          f"loc={loc_grad_norm:.4f}, t={t_grad_norm:.4f}, "
                                          f"n={n_grad_norm:.4f}, m={m_grad_norm:.4f}")
                    self.print_to_log_file(f"Epoch {self.current_epoch} - EMA Norms: seg={self.ema_grad_norm['seg']:.4f}, "
                                          f"loc={self.ema_grad_norm['location']:.4f}, "
                                          f"t={self.ema_grad_norm['t_stage']:.4f}, "
                                          f"n={self.ema_grad_norm['n_stage']:.4f}, "
                                          f"m={self.ema_grad_norm['m_stage']:.4f}")
                    self.print_to_log_file(f"Epoch {self.current_epoch} - GradNorm Factors: "
                                        f"seg={self.grad_norm_factors['seg']:.4f}, "
                                        f"loc={self.grad_norm_factors['location']:.4f}, "
                                        f"t={self.grad_norm_factors['t_stage']:.4f}, "
                                        f"n={self.grad_norm_factors['n_stage']:.4f}, "
                                        f"m={self.grad_norm_factors['m_stage']:.4f}")
                    self.print_to_log_file(f"Epoch {self.current_epoch} - Base Weights: "
                                        f"seg={self.clinical_loss_manual_weights.get('seg', 1.0):.4f}, "
                                        f"loc={self.clinical_loss_manual_weights['location']:.4f}, "
                                        f"t={self.clinical_loss_manual_weights['t_stage']:.4f}, "
                                        f"n={self.clinical_loss_manual_weights['n_stage']:.4f}, "
                                        f"m={self.clinical_loss_manual_weights['m_stage']:.4f}")
                    self.print_to_log_file(f"Epoch {self.current_epoch} - Final Weights: "
                                        f"seg={self.clinical_loss_manual_weights.get('seg', 1.0) * self.grad_norm_factors['seg']:.4f}, "
                                        f"loc={self.clinical_loss_manual_weights['location'] * self.grad_norm_factors['location']:.4f}, "
                                        f"t={self.clinical_loss_manual_weights['t_stage'] * self.grad_norm_factors['t_stage']:.4f}, "
                                        f"n={self.clinical_loss_manual_weights['n_stage'] * self.grad_norm_factors['n_stage']:.4f}, "
                                        f"m={self.clinical_loss_manual_weights['m_stage'] * self.grad_norm_factors['m_stage']:.4f}")
                    self.print_to_log_file("") # 空行
            

            # 計算各類最終損失
            # 確保 'seg' 鍵存在於 clinical_loss_manual_weights 中，否則使用預設值 1.0
            seg_weight = self.clinical_loss_manual_weights.get('seg', 1.0)
            final_loss_seg = seg_weight * self.grad_norm_factors['seg'] * seg_loss_tr
            final_loss_loc = self.clinical_loss_manual_weights['location'] * self.grad_norm_factors['location'] * loc_loss_tr
            final_loss_t = self.clinical_loss_manual_weights['t_stage'] * self.grad_norm_factors['t_stage'] * t_loss_tr
            final_loss_n = self.clinical_loss_manual_weights['n_stage'] * self.grad_norm_factors['n_stage'] * n_loss_tr
            final_loss_m = self.clinical_loss_manual_weights['m_stage'] * self.grad_norm_factors['m_stage'] * m_loss_tr

            # --- 最終總損失 ---
            total_loss = (
                final_loss_seg +
                # 權重 * 梯度調整因子 * loss
                final_loss_loc +
                final_loss_t +
                final_loss_n +
                final_loss_m
            )

        # 梯度計算 反向更新
        # 如果有啟用混合精度(有偵測到GPU就會啟用), grad_scaler 混合精度 (AMP, fp16)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward() # 縮放損失以適應 fp16 接著反向傳播
            self.grad_scaler.unscale_(self.optimizer)     # 取消縮放以獲取梯度
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12) # 梯度裁減 防止梯度爆炸
            self.grad_scaler.step(self.optimizer)         # 優化器步驟
            self.grad_scaler.update()                     # 更新縮放因子
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()


        # 返回所有損失值 (轉移到 CPU 並轉換為 numpy)
        return {
            # 原始損失
            'seg_loss': seg_loss_tr.detach().cpu().numpy(),
            'loc_loss': loc_loss_tr.detach().cpu().numpy(),
            't_loss': t_loss_tr.detach().cpu().numpy(),
            'n_loss': n_loss_tr.detach().cpu().numpy(),
            'm_loss': m_loss_tr.detach().cpu().numpy(),
            # 最終權重 (基本權重 * 調整因子)
            'final_weight_seg': np.array(self.clinical_loss_manual_weights.get('seg', 1.0) * self.grad_norm_factors['seg']),
            'final_weight_loc': np.array(self.clinical_loss_manual_weights['location'] * self.grad_norm_factors['location']),
            'final_weight_t': np.array(self.clinical_loss_manual_weights['t_stage'] * self.grad_norm_factors['t_stage']),
            'final_weight_n': np.array(self.clinical_loss_manual_weights['n_stage'] * self.grad_norm_factors['n_stage']),
            'final_weight_m': np.array(self.clinical_loss_manual_weights['m_stage'] * self.grad_norm_factors['m_stage']),
            # 最終損失
            'loss': total_loss.detach().cpu().numpy(), # 總損失
            'final_loss_seg': final_loss_seg.detach().cpu().numpy(),
            'final_loss_loc': final_loss_loc.detach().cpu().numpy(),
            'final_loss_t': final_loss_t.detach().cpu().numpy(),
            'final_loss_n': final_loss_n.detach().cpu().numpy(),
            'final_loss_m': final_loss_m.detach().cpu().numpy(),
        }

    def _calculate_grad_norm(self, loss, key):
        """
        計算梯度大小 (norm)
        
        當輸入的 loss 不是標量時，會先對其進行平均操作，確保它是標量
        然後通過反向傳播計算梯度，並計算所有參數梯度的 L2 範數
        """
        # '''法一: 傳統的 loss.backward() 計算梯度並獲取梯度範數'''
        # # 確保損失是標量
        # if loss.dim() > 0:  # 如果損失不是標量
        #     loss = loss.mean()  # 取平均值變成標量
            
        # # 清零梯度
        # self.optimizer.zero_grad(set_to_none=True)
        
        # # # 梯度計算 反向更新
        # # # 如果有啟用混合精度(有偵測到GPU就會啟用), grad_scaler 混合精度 (AMP, fp16)
        # # if self.grad_scaler is not None:
        # #     self.grad_scaler.scale(loss).backward(retain_graph=True) # 縮放損失以適應 fp16 接著反向傳播
        # #     self.grad_scaler.unscale_(self.optimizer)     # 取消縮放以獲取梯度
        # # else:
        # #     loss.backward(retain_graph=True)
        # loss.backward(retain_graph=True)

        # # 計算梯度大小
        # grad_norm_1 = 0.0
        # for param in [p for p in self.network.parameters() if p.requires_grad]:
        #     if param.grad is not None:
        #         grad_norm_1 += torch.sum(param.grad ** 2).item()
        # grad_norm_1 = np.sqrt(grad_norm_1)
        
        # # 清零梯度以準備下一次計算
        # self.optimizer.zero_grad(set_to_none=True)

        '''法二: 使用 torch.autograd.grad 計算梯度並獲取梯度範數'''
        # 獲取需要梯度的參數並計算梯度
        params = [p for p in self.network.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        
        # 計算梯度範數
        grad_norm = torch.sqrt(sum(torch.norm(grad) ** 2 for grad in grads if grad is not None)).item()
    
        # '''檢驗兩種方法是否一致'''
        # assert np.isclose(grad_norm, grad_norm_1), self.print_to_log_file(f"Gradient norm 不一致: {grad_norm} vs {grad_norm_1}")


        if grad_norm > 20:
            if self.current_epoch != 0:
                self.print_to_log_file(f"Warning: {key} 的原始 Gradient norm 爆炸 {grad_norm} on epoch {self.current_epoch}")
            grad_norm = 20.0  # 限制最大值，防止極端值影響穩定性


        return grad_norm
        
    def _update_ema_grad_norm(self, key, grad_norm):
        """
        更新指定任務的 EMA 梯度範數
        """
        # # ema = alpha * 舊的 ema + (1 - alpha) * 新的值
        self.ema_grad_norm[key] = self.ema_alpha * self.ema_grad_norm[key] + (1 - self.ema_alpha) * grad_norm
        return self.ema_grad_norm[key]
    
    def on_train_epoch_end(self, train_outputs: List[dict]):
        """
        在每個訓練 Epoch 結束時彙總結果並記錄。
        覆寫父類的 on_train_epoch_end 方法。
        """
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            # # 彙總所有 worker 的損失
            # 最終總損失
            losses_tr_total = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr_total, outputs['loss'])
            loss_here = np.vstack(losses_tr_total).mean()

            # 原始損失
            seg_losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(seg_losses_tr, outputs['seg_loss'])
            seg_loss_here = np.vstack(seg_losses_tr).mean()

            loc_losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(loc_losses_tr, outputs['loc_loss'])
            loc_loss_here = np.vstack(loc_losses_tr).mean()

            t_losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(t_losses_tr, outputs['t_loss'])
            t_loss_here = np.vstack(t_losses_tr).mean()

            n_losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(n_losses_tr, outputs['n_loss'])
            n_loss_here = np.vstack(n_losses_tr).mean()

            m_losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(m_losses_tr, outputs['m_loss'])
            m_loss_here = np.vstack(m_losses_tr).mean()

            # 最終權重
            final_weight_seg = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(final_weight_seg, outputs['final_weight_seg'])
            final_weight_seg = np.vstack(final_weight_seg).mean()

            final_weight_loc = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(final_weight_loc, outputs['final_weight_loc'])
            final_weight_loc = np.vstack(final_weight_loc).mean()

            final_weight_t = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(final_weight_t, outputs['final_weight_t'])
            final_weight_t = np.vstack(final_weight_t).mean()

            final_weight_n = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(final_weight_n, outputs['final_weight_n'])
            final_weight_n = np.vstack(final_weight_n).mean()

            final_weight_m = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(final_weight_m, outputs['final_weight_m'])
            final_weight_m = np.vstack(final_weight_m).mean()

            # 最終損失
            final_loss_seg = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(final_loss_seg, outputs['final_loss_seg'])
            final_loss_seg = np.vstack(final_loss_seg).mean()

            final_loss_loc = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(final_loss_loc, outputs['final_loss_loc'])
            final_loss_loc = np.vstack(final_loss_loc).mean()

            final_loss_t = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(final_loss_t, outputs['final_loss_t'])
            final_loss_t = np.vstack(final_loss_t).mean()

            final_loss_n = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(final_loss_n, outputs['final_loss_n'])
            final_loss_n = np.vstack(final_loss_n).mean()

            final_loss_m = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(final_loss_m, outputs['final_loss_m'])
            final_loss_m = np.vstack(final_loss_m).mean()

        else:
            # 最終總損失
            loss_here = np.mean(outputs['loss'])
            # 原始損失
            seg_loss_here = np.mean(outputs['seg_loss'])
            loc_loss_here = np.mean(outputs['loc_loss'])
            t_loss_here = np.mean(outputs['t_loss'])
            n_loss_here = np.mean(outputs['n_loss'])
            m_loss_here = np.mean(outputs['m_loss'])
            # 最終權重
            final_weight_seg = np.mean(outputs['final_weight_seg'])
            final_weight_loc = np.mean(outputs['final_weight_loc'])
            final_weight_t = np.mean(outputs['final_weight_t'])
            final_weight_n = np.mean(outputs['final_weight_n'])
            final_weight_m = np.mean(outputs['final_weight_m'])
            # 最終損失
            final_loss_seg = np.mean(outputs['final_loss_seg'])
            final_loss_loc = np.mean(outputs['final_loss_loc'])
            final_loss_t = np.mean(outputs['final_loss_t'])
            final_loss_n = np.mean(outputs['final_loss_n'])
            final_loss_m = np.mean(outputs['final_loss_m'])

        # 記錄到 logger（全部使用 train_xxx_losses）
        self.logger.log('train_losses', seg_loss_here, self.current_epoch) # 給原生 trainer logger 用 避免報錯
        # 最終總損失
        self.logger.log('train_total_losses', loss_here, self.current_epoch)
        # 原始損失
        self.logger.log('train_seg_losses', seg_loss_here, self.current_epoch)
        self.logger.log('train_loc_losses', loc_loss_here, self.current_epoch)
        self.logger.log('train_t_losses', t_loss_here, self.current_epoch)
        self.logger.log('train_n_losses', n_loss_here, self.current_epoch)
        self.logger.log('train_m_losses', m_loss_here, self.current_epoch)
        # 最終權重
        self.logger.log('tr_loss_weights_seg', final_weight_seg, self.current_epoch)
        self.logger.log('tr_loss_weights_loc', final_weight_loc, self.current_epoch)
        self.logger.log('tr_loss_weights_t', final_weight_t, self.current_epoch)
        self.logger.log('tr_loss_weights_n', final_weight_n, self.current_epoch)
        self.logger.log('tr_loss_weights_m', final_weight_m, self.current_epoch)
        # 最終損失
        self.logger.log('tr_final_loss_seg', final_loss_seg, self.current_epoch)
        self.logger.log('tr_final_loss_loc', final_loss_loc, self.current_epoch)
        self.logger.log('tr_final_loss_t', final_loss_t, self.current_epoch)
        self.logger.log('tr_final_loss_n', final_loss_n, self.current_epoch)
        self.logger.log('tr_final_loss_m', final_loss_m, self.current_epoch)

    def validation_step(self, batch: dict) -> dict:
        """
        執行一個驗證步驟，包括前向傳播、損失計算和指標計算。
        覆寫父類的 validation_step 方法。
        """
        img_data = batch['data']
        seg_target = batch['target']
        clinical_data_aug = batch['clinical_data_aug']       # 增強後的資料 (用於模型輸入)
        clinical_data_label = batch['clinical_data_label']   # 原始資料 (用於計算loss)
        clinical_mask = batch['clinical_mask']               # mask
        keys = batch['keys']

        # 將影像數據和seg label移動到指定設備
        image_data = img_data.to(self.device, non_blocking=True)
        if isinstance(seg_target, list): # 如果有多種分割目標器官
            seg_target = [i.to(self.device, non_blocking=True) for i in seg_target]
        else:
            seg_target = seg_target.to(self.device, non_blocking=True)

        # 將臨床特徵(模型輸入)移動到指定設備
        loc_input = torch.tensor(clinical_data_aug['location']).to(self.device, non_blocking=True)
        t_input = torch.tensor(clinical_data_aug['t_stage']).to(self.device, non_blocking=True)
        n_input = torch.tensor(clinical_data_aug['n_stage']).to(self.device, non_blocking=True)
        m_input = torch.tensor(clinical_data_aug['m_stage']).to(self.device, non_blocking=True)

        # 將臨床標籤(用於計算loss)移動到指定設備
        loc_label = torch.tensor(clinical_data_label['location']).to(self.device, non_blocking=True)
        t_label = torch.tensor(clinical_data_label['t_stage']).to(self.device, non_blocking=True)
        n_label = torch.tensor(clinical_data_label['n_stage']).to(self.device, non_blocking=True)
        m_label = torch.tensor(clinical_data_label['m_stage']).to(self.device, non_blocking=True)

        # Missing特徵的遮罩 用於計算損失時忽略
        loc_mask = torch.tensor(clinical_mask['location']).to(self.device, non_blocking=True)
        t_mask = torch.tensor(clinical_mask['t_stage']).to(self.device, non_blocking=True)
        n_mask = torch.tensor(clinical_mask['n_stage']).to(self.device, non_blocking=True)
        m_mask = torch.tensor(clinical_mask['m_stage']).to(self.device, non_blocking=True)

        # 推論結果
        # Autocast 只在 CUDA 上啟用，CPU/MPS 不啟用以避免效能低落或報錯
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else self.dummy_context():
            seg_out, cli_out = self.network(image_data, loc_input, t_input, n_input, m_input)
            # 釋放 image_data 記憶體，減少顯存佔用 (臨床特徵 還要計算指標用 別刪)
            del image_data

            # --- 計算分割損失和 Dice ---
            seg_loss_val = self.loss(seg_out, seg_target) # 分割損失

            # 如果啟用深度監督，會有多個解析度的輸出（list），只用最高解析度計算 Dice
            if self.enable_deep_supervision:
                output_seg = seg_out[0]
                seg_target_for_dice = seg_target[0]
            else:
                output_seg = seg_out
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


            # 初始化臨床損失
            loc_loss_val = torch.tensor(0.0, device=self.device)
            t_loss_val = torch.tensor(0.0, device=self.device)
            n_loss_val = torch.tensor(0.0, device=self.device)
            m_loss_val = torch.tensor(0.0, device=self.device)

            # --- 計算臨床分類準確率 ---
            loc_acc_val = torch.tensor(0.0, device=self.device)
            t_acc_val = torch.tensor(0.0, device=self.device)
            n_acc_val = torch.tensor(0.0, device=self.device)
            m_acc_val = torch.tensor(0.0, device=self.device)

            # 計算損失
            if self.enable_deep_supervision:
                loc_loss_val = self.focal_loss_loc(
                    cli_out['location'],           # list of tensor，每個分支一個 tensor
                    [loc_label] * len(cli_out['location']),  # list，每個分支都用同一個 label tensor
                    [loc_mask] * len(cli_out['location'])    # list，每個分支都用同一個 mask tensor
                )
                t_loss_val = self.focal_loss_t(
                    cli_out['t_stage'],
                    [t_label] * len(cli_out['t_stage']),
                    [t_mask] * len(cli_out['t_stage'])
                )
                n_loss_val = self.focal_loss_n(
                    cli_out['n_stage'],
                    [n_label] * len(cli_out['n_stage']),
                    [n_mask] * len(cli_out['n_stage'])
                )
                m_loss_val = self.focal_loss_m(
                    cli_out['m_stage'],
                    [m_label] * len(cli_out['m_stage']),
                    [m_mask] * len(cli_out['m_stage'])
                )

            else:
                loc_loss_val = self.focal_loss_loc(cli_out['location'], loc_label, loc_mask)
                t_loss_val = self.focal_loss_t(cli_out['t_stage'], t_label, t_mask)
                n_loss_val = self.focal_loss_n(cli_out['n_stage'], n_label, n_mask)
                m_loss_val = self.focal_loss_m(cli_out['m_stage'], m_label, m_mask)

            # 計算準確率
            if self.enable_deep_supervision:
                # 只取最高解析度分支 (通常是 list[0])
                loc_logits = cli_out['location'][0]
                t_logits   = cli_out['t_stage'][0]
                n_logits   = cli_out['n_stage'][0]
                m_logits   = cli_out['m_stage'][0]
            else:
                loc_logits = cli_out['location']
                t_logits   = cli_out['t_stage']
                n_logits   = cli_out['n_stage']
                m_logits   = cli_out['m_stage']


            # location
            loc_preds = loc_logits.argmax(dim=1) # 取得預測logits 並且透過 argmax 取得預測類別
            loc_correct = (loc_preds == loc_label)        # 是否預測正確 (True/False)
            loc_valid = loc_mask.bool()                   # 有效樣本遮罩 (True/False)
            loc_correct_valid = loc_correct & loc_valid   # 只保留有效且正確的樣本
            # 有效樣本的準確率（分母加clamp避免除以0）
            loc_acc_val = loc_correct_valid.float().sum() / loc_valid.sum().clamp(min=1)

            # t_stage
            t_preds = t_logits.argmax(dim=1)
            t_correct = (t_preds == t_label)
            t_valid = t_mask.bool()
            t_correct_valid = t_correct & t_valid
            t_acc_val = t_correct_valid.float().sum() / t_valid.sum().clamp(min=1)

            # n_stage
            n_preds = n_logits.argmax(dim=1)
            n_correct = (n_preds == n_label)
            n_valid = n_mask.bool()
            n_correct_valid = n_correct & n_valid
            n_acc_val = n_correct_valid.float().sum() / n_valid.sum().clamp(min=1)

            # m_stage
            m_preds = m_logits.argmax(dim=1)
            m_correct = (m_preds == m_label)
            m_valid = m_mask.bool()
            m_correct_valid = m_correct & m_valid
            m_acc_val = m_correct_valid.float().sum() / m_valid.sum().clamp(min=1)

            
            # 計算各類最終損失 (加上 grad_norm_factors 調整)
            seg_weight = self.clinical_loss_manual_weights.get('seg', 1.0)
            final_loss_seg = seg_weight * self.grad_norm_factors['seg'] * seg_loss_val
            final_loss_loc = self.clinical_loss_manual_weights['location'] * self.grad_norm_factors['location'] * loc_loss_val
            final_loss_t = self.clinical_loss_manual_weights['t_stage'] * self.grad_norm_factors['t_stage'] * t_loss_val
            final_loss_n = self.clinical_loss_manual_weights['n_stage'] * self.grad_norm_factors['n_stage'] * n_loss_val
            final_loss_m = self.clinical_loss_manual_weights['m_stage'] * self.grad_norm_factors['m_stage'] * m_loss_val

            # 最終總損失
            total_loss = (
                final_loss_seg +
                final_loss_loc +
                final_loss_t +
                final_loss_n +
                final_loss_m
            )

        # 回傳所有損失和指標（轉到 CPU 並轉 numpy）
        return {
            'loss': total_loss.detach().cpu().numpy(), # 總損失
            'seg_loss': seg_loss_val.detach().cpu().numpy(),
            'loc_loss': loc_loss_val.detach().cpu().numpy(),
            't_loss': t_loss_val.detach().cpu().numpy(),
            'n_loss': n_loss_val.detach().cpu().numpy(),
            'm_loss': m_loss_val.detach().cpu().numpy(),
            'final_loss_seg': final_loss_seg.detach().cpu().numpy(),
            'final_loss_loc': final_loss_loc.detach().cpu().numpy(),
            'final_loss_t': final_loss_t.detach().cpu().numpy(),
            'final_loss_n': final_loss_n.detach().cpu().numpy(),
            'final_loss_m': final_loss_m.detach().cpu().numpy(),
            'loc_acc': loc_acc_val.detach().cpu().numpy(),
            't_acc': t_acc_val.detach().cpu().numpy(),
            'n_acc': n_acc_val.detach().cpu().numpy(),
            'm_acc': m_acc_val.detach().cpu().numpy(),
            'tp_hard': tp_hard, # 分割 Dice 統計
            'fp_hard': fp_hard,
            'fn_hard': fn_hard
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
            # 對 T, N, M 損失進行同樣的彙總
            t_losses_val = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(t_losses_val, outputs_collated['t_loss'])
            t_loss_here = np.vstack(t_losses_val).mean()

            n_losses_val = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(n_losses_val, outputs_collated['n_loss'])
            n_loss_here = np.vstack(n_losses_val).mean()

            m_losses_val = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(m_losses_val, outputs_collated['m_loss'])
            m_loss_here = np.vstack(m_losses_val).mean()
            
            # 彙總所有 worker 的調整後損失
            final_seg_losses_val = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(final_seg_losses_val, outputs_collated['final_loss_seg'])
            final_seg_loss_here = np.vstack(final_seg_losses_val).mean()
            
            final_loc_losses_val = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(final_loc_losses_val, outputs_collated['final_loss_loc'])
            final_loc_loss_here = np.vstack(final_loc_losses_val).mean()
            
            final_t_losses_val = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(final_t_losses_val, outputs_collated['final_loss_t'])
            final_t_loss_here = np.vstack(final_t_losses_val).mean()
            
            final_n_losses_val = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(final_n_losses_val, outputs_collated['final_loss_n'])
            final_n_loss_here = np.vstack(final_n_losses_val).mean()
            
            final_m_losses_val = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(final_m_losses_val, outputs_collated['final_loss_m'])
            final_m_loss_here = np.vstack(final_m_losses_val).mean()

            # 彙總所有 worker 的臨床準確度 (需要加權平均或只取有數據的樣本)
            # 為了簡化，我們將收集所有樣本的準確度，然後平均
            loc_accs_val = [None for _ in range(world_size)]
            dist.all_gather_object(loc_accs_val, outputs_collated['loc_acc'])
            loc_acc_here = np.vstack(loc_accs_val).mean()
            # 對 T, N, M 準確度進行同樣的彙總
            t_accs_val = [None for _ in range(world_size)]
            dist.all_gather_object(t_accs_val, outputs_collated['t_acc'])
            t_acc_here = np.vstack(t_accs_val).mean()

            n_accs_val = [None for _ in range(world_size)]
            dist.all_gather_object(n_accs_val, outputs_collated['n_acc'])
            n_acc_here = np.vstack(n_accs_val).mean()

            m_accs_val = [None for _ in range(world_size)]
            dist.all_gather_object(m_accs_val, outputs_collated['m_acc'])
            m_acc_here = np.vstack(m_accs_val).mean()

        else:
            loss_here = np.mean(outputs_collated['loss'])
            seg_loss_here = np.mean(outputs_collated['seg_loss'])
            loc_loss_here = np.mean(outputs_collated['loc_loss'])
            t_loss_here = np.mean(outputs_collated['t_loss'])
            n_loss_here = np.mean(outputs_collated['n_loss'])
            m_loss_here = np.mean(outputs_collated['m_loss'])
            
            # 調整後的損失
            final_seg_loss_here = np.mean(outputs_collated['final_loss_seg'])
            final_loc_loss_here = np.mean(outputs_collated['final_loss_loc'])
            final_t_loss_here = np.mean(outputs_collated['final_loss_t'])
            final_n_loss_here = np.mean(outputs_collated['final_loss_n'])
            final_m_loss_here = np.mean(outputs_collated['final_loss_m'])

            loc_acc_here = np.mean(outputs_collated['loc_acc'])
            t_acc_here = np.mean(outputs_collated['t_acc'])
            n_acc_here = np.mean(outputs_collated['n_acc'])
            m_acc_here = np.mean(outputs_collated['m_acc'])


        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)

        # 記錄到 logger（全部使用 val_xxx_losses 與 val_xxx_accs）
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', seg_loss_here, self.current_epoch) # 給原生 trainer logger 用 避免報錯
        self.logger.log('val_total_losses', loss_here, self.current_epoch)
        self.logger.log('val_seg_losses', seg_loss_here, self.current_epoch)
        self.logger.log('val_loc_losses', loc_loss_here, self.current_epoch)
        self.logger.log('val_t_losses', t_loss_here, self.current_epoch)
        self.logger.log('val_n_losses', n_loss_here, self.current_epoch)
        self.logger.log('val_m_losses', m_loss_here, self.current_epoch)
        
        # 記錄調整後的損失
        self.logger.log('val_final_loss_seg', final_seg_loss_here, self.current_epoch)
        self.logger.log('val_final_loss_loc', final_loc_loss_here, self.current_epoch)
        self.logger.log('val_final_loss_t', final_t_loss_here, self.current_epoch)
        self.logger.log('val_final_loss_n', final_n_loss_here, self.current_epoch)
        self.logger.log('val_final_loss_m', final_m_loss_here, self.current_epoch)

        self.logger.log('val_loc_accs', loc_acc_here, self.current_epoch)
        self.logger.log('val_t_accs', t_acc_here, self.current_epoch)
        self.logger.log('val_n_accs', n_acc_here, self.current_epoch)
        self.logger.log('val_m_accs', m_acc_here, self.current_epoch)


        # 原本打印指標是在on epoch end，但是因為會延遲變成在下個epoch才印出來
        # 所以提前打印
        if self.local_rank == 0:
            # 額外打印臨床相關指標
            loc_acc = self.logger.my_fantastic_logging['val_loc_accs'][-1] if len(self.logger.my_fantastic_logging['val_loc_accs']) > 0 else "N/A"
            t_acc = self.logger.my_fantastic_logging['val_t_accs'][-1] if len(self.logger.my_fantastic_logging['val_t_accs']) > 0 else "N/A"
            n_acc = self.logger.my_fantastic_logging['val_n_accs'][-1] if len(self.logger.my_fantastic_logging['val_n_accs']) > 0 else "N/A"
            m_acc = self.logger.my_fantastic_logging['val_m_accs'][-1] if len(self.logger.my_fantastic_logging['val_m_accs']) > 0 else "N/A"
            
            self.print_to_log_file(f"Val location Acc: {np.round(loc_acc, decimals=4)}", add_timestamp=True)
            self.print_to_log_file(f"Val T Stage Acc: {np.round(t_acc, decimals=4)}", add_timestamp=True)
            self.print_to_log_file(f"Val N Stage Acc: {np.round(n_acc, decimals=4)}", add_timestamp=True)
            self.print_to_log_file(f"Val M Stage Acc: {np.round(m_acc, decimals=4)}", add_timestamp=True)
            
            # # 輸出調整後的損失
            # self.print_to_log_file(f"Val Final Losses - Seg: {np.round(final_seg_loss_here, decimals=4)}, "
            #                       f"Loc: {np.round(final_loc_loss_here, decimals=4)}, "
            #                       f"T: {np.round(final_t_loss_here, decimals=4)}, "
            #                       f"N: {np.round(final_n_loss_here, decimals=4)}, "
            #                       f"M: {np.round(final_m_loss_here, decimals=4)}", add_timestamp=True)
            
            # 更新最佳 EMA pseudo Dice 的邏輯已經在父類 on_epoch_end 中處理
            # 如果需要根據臨床指標決定 best checkpoint，需要修改父類的邏輯或在這裡添加額外判斷



    def on_epoch_end(self):
        """
        在每個 Epoch 結束時記錄時間戳，並處理檢查點儲存。
        覆寫父類的 on_epoch_end 方法。
        """
        # 更新臨床損失權重
        self.update_clinical_loss_manual_weights()

        super().on_epoch_end() # 調用父類方法處理時間戳、Dice 記錄和檢查點儲存

        self.lr_scheduler.step() # 

    
    def save_checkpoint(self, filename: str) -> None:
        """
        儲存訓練檢查點，包括模型權重、優化器狀態、學習率排程器狀態等。
        覆寫父類的 save_checkpoint 方法，添加對學習率排程器狀態的保存。
        Args:
            filename (str): 檢查點檔案路徑
        """
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                if self.is_ddp:
                    mod = self.network.module
                else:
                    mod = self.network
                if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod

                # 構建檢查點字典，包含原始內容和學習率排程器狀態
                checkpoint = {
                    'network_weights': mod.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'lr_scheduler_state': self.lr_scheduler.state_dict(),  # 添加學習率排程器狀態
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                    # 多模態訓練器特有資訊
                    'clinical_loss_manual_weights': self.clinical_loss_manual_weights if hasattr(self, 'clinical_loss_manual_weights') else None,
                    'grad_norm_factors': self.grad_norm_factors if hasattr(self, 'grad_norm_factors') else None,
                    'is_stage2_dataset': self.is_stage2_dataset if hasattr(self, 'is_stage2_dataset') else None,
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        """
        載入訓練檢查點，恢復模型權重、優化器狀態、學習率排程器狀態等。
        覆寫父類的 load_checkpoint 方法，添加對學習率排程器狀態的載入。
        Args:
            filename_or_checkpoint (Union[dict, str]): 檢查點檔案路徑或已載入的檢查點字典
        """
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
        else:
            checkpoint = filename_or_checkpoint

        # 處理網路權重載入
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        # 載入基本資訊
        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self.inference_allowed_mirroring_axes = checkpoint.get(
            'inference_allowed_mirroring_axes', self.inference_allowed_mirroring_axes)

        # 載入多模態訓練器特有資訊
        if hasattr(self, 'clinical_loss_manual_weights') and 'clinical_loss_manual_weights' in checkpoint:
            self.clinical_loss_manual_weights = checkpoint['clinical_loss_manual_weights']
        
        if hasattr(self, 'grad_norm_factors') and 'grad_norm_factors' in checkpoint:
            self.grad_norm_factors = checkpoint['grad_norm_factors']
        
        if hasattr(self, 'is_stage2_dataset') and 'is_stage2_dataset' in checkpoint:
            self.is_stage2_dataset = checkpoint['is_stage2_dataset']

        # 根據模型類型載入網路權重
        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)

        # 載入優化器狀態
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # 載入學習率排程器狀態
        if 'lr_scheduler_state' in checkpoint and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])
            self.print_to_log_file("學習率排程器狀態已恢復")
            
            # 檢查 cosine scheduler 的曲率是否正確恢復
            self.print_to_log_file(f"[Scheduler恢復檢查] 目前 epoch={self.current_epoch}")
            self.print_to_log_file(f"[Scheduler恢復檢查] scheduler.last_epoch={self.lr_scheduler.last_epoch}")
            self.print_to_log_file(f"[Scheduler恢復檢查] scheduler 類型: {type(self.lr_scheduler).__name__}")
            
            # 檢查 scheduler 的 last_epoch 是否與 current_epoch 一致
            expected_last_epoch = self.current_epoch - 1  # scheduler.last_epoch 通常比 current_epoch 少 1
            if hasattr(self.lr_scheduler, 'last_epoch'):
                if abs(self.lr_scheduler.last_epoch - expected_last_epoch) > 1:
                    raise RuntimeError(f"Scheduler last_epoch 異常: scheduler.last_epoch={self.lr_scheduler.last_epoch}, expected={expected_last_epoch}")
            
            # 特別針對 cosine scheduler 檢查曲率
            current_lr = self.optimizer.param_groups[0]['lr']
            if 'Cosine' in type(self.lr_scheduler).__name__:
                # 更簡單的方式：直接檢查 scheduler 執行一步後的 lr 變化
                old_last_epoch = self.lr_scheduler.last_epoch
                old_lr = current_lr
                
                # 備份當前狀態
                scheduler_state_backup = self.lr_scheduler.state_dict()
                
                # 執行一步看 lr 變化
                self.lr_scheduler.step()
                next_lr_actual = self.optimizer.param_groups[0]['lr']
                
                # 還原 scheduler 和 optimizer 狀態
                self.lr_scheduler.load_state_dict(scheduler_state_backup)
                self.optimizer.param_groups[0]['lr'] = old_lr
                
                # 計算 lr 變化率
                lr_change = abs(next_lr_actual - old_lr)
                lr_change_ratio = lr_change / (old_lr + 1e-8)
                
                self.print_to_log_file(f"[Cosine曲率檢查] current_lr={old_lr:.6e}")
                self.print_to_log_file(f"[Cosine曲率檢查] next_lr_after_step={next_lr_actual:.6e}")
                self.print_to_log_file(f"[Cosine曲率檢查] lr_change={lr_change:.6e}")
                self.print_to_log_file(f"[Cosine曲率檢查] lr_change_ratio={lr_change_ratio:.6f}")
                self.print_to_log_file(f"[Cosine曲率檢查] scheduler.last_epoch after step={old_last_epoch + 1}")
                
                # 檢查是否從頭開始（cosine 開始時變化很小）
                # 如果是恢復的中後期，lr 變化應該比較明顯
                if old_last_epoch > self.num_epochs * 0.1:  # 超過 10% 的 epoch
                    if lr_change_ratio < 0.001:  # 變化小於 0.1%
                        self.print_to_log_file(f"[警告] Cosine scheduler 可能從頭開始！在 epoch {old_last_epoch} 時 lr 變化過小")
                        self.print_to_log_file(f"[警告] 這可能表示 scheduler 曲率未正確恢復")
                        # 注意：這裡不 raise error，只是警告，因為有些情況下後期 lr 變化確實很小
                
                self.print_to_log_file("[Cosine曲率檢查] 完成 ✓")
            
            self.print_to_log_file("[Scheduler恢復檢查] 通過 ✓")

        # 載入梯度縮放器狀態
        if self.grad_scaler is not None and 'grad_scaler_state' in checkpoint and checkpoint['grad_scaler_state'] is not None:
            self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])


    def update_clinical_loss_manual_weights(self):
        """
        根據當前 epoch 從權重控制點列表動態更新臨床損失權重。
        使用分段線性插值計算當前 epoch 的權重值。
        """
        current_epoch = self.current_epoch
        
        # 對每個任務分別進行處理
        for task, schedule in self.clinical_loss_weight_schedule.items():
            # 如果當前 epoch 小於第一個控制點的 epoch，使用第一個控制點的權重
            if current_epoch <= schedule[0][0]:
                self.clinical_loss_manual_weights[task] = schedule[0][1]
                continue
                
            # 如果當前 epoch 大於等於最後一個控制點的 epoch，使用最後一個控制點的權重
            if current_epoch >= schedule[-1][0]:
                self.clinical_loss_manual_weights[task] = schedule[-1][1]
                continue
                
            # 找到當前 epoch 所在的區間
            for i in range(len(schedule) - 1):
                epoch_start, weight_start = schedule[i]
                epoch_end, weight_end = schedule[i + 1]
                
                if epoch_start <= current_epoch < epoch_end:
                    # 計算在當前區間內的插值比例
                    progress_ratio = (current_epoch - epoch_start) / (epoch_end - epoch_start)
                    # 線性插值計算當前權重
                    current_weight = weight_start + progress_ratio * (weight_end - weight_start)
                    self.clinical_loss_manual_weights[task] = current_weight
                    break

        # 可選：打印當前權重（僅在 rank 0 進程，避免重複打印）
        if self.local_rank == 0:
            # 打印當前權重
            self.print_to_log_file(f"Epoch {current_epoch}: 更新臨床損失權重為 {self.clinical_loss_manual_weights}")


    def set_deep_supervision_enabled(self, enabled: bool):
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if hasattr(mod, 'set_deep_supervision'):
            mod.set_deep_supervision(enabled)
        else:
            mod.decoder.deep_supervision = enabled

    # 改成使用 AdamW 
    def configure_optimizers(self):
        # 設置優化器初始學習率
        initial_lr = self.initial_lr/100  # 1e-2 / 100 = 1e-4
        
        optimizer = torch.optim.AdamW(self.network.parameters(),
                                      lr=initial_lr,  # 直接設置初始學習率
                                      weight_decay=self.weight_decay*100, # 3e-5 * 100= 3e-3
                                      betas=(0.9, 0.999),
                                      eps=1e-8)
        
        from torch.optim.lr_scheduler import CosineAnnealingLR
        # 使用 CosineAnnealingLR，每個 epoch 更新一次
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.num_epochs,  # 週期長度
            eta_min=initial_lr/1000,  # 最小學習率 # 1e-4 / 1000 = 1e-7
        )
        return optimizer, lr_scheduler


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
            allow_tqdm=False,
            clinical_data_dir = self.clinical_data_dir,  # 傳入臨床資料目錄
        )

        # 手動初始化 predictor，傳入訓練好的網路和相關設定
        # 這裡的 self.network 已經是 MyMultiModel，會被正確傳遞
        predictor.manual_initialization(
            self.network, 
            self.plans_manager,
            self.configuration_manager,
            None, # parameters: 在這裡不需要多折集成，因為我們只評估當前訓練的模型
            self.dataset_json,
            self.__class__.__name__,
            self.inference_allowed_mirroring_axes
        )

        # 初始化臨床編碼器
        predictor.initialize_clinical_encoder()

        print("nnUNetPredictorMultimodal 已初始化")

        # 建立多行程池用於匯出分割結果
        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)

            # 獲取驗證集鍵列表
            _, val_keys = self.do_split()

            # # [DEBUG用] 只取前五個案例進行測試
            # val_keys = val_keys[:5]


            # 如果是 DDP (分散式數據並行) 模式，則將驗證鍵分配給不同的進程
            if self.is_ddp:
                last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1
                val_keys = val_keys[self.local_rank:: dist.get_world_size()]

            # 建立驗證資料集物件 (確保使用正確的 Dataset 類別和 clinical_data_dir)
            try:
                # Stage 2 訓練：使用多模態 Dataset 並傳入臨床資料路徑
                dataset_val = self.dataset_class(
                    self.preprocessed_dataset_folder,
                    val_keys,
                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                    clinical_data_dir=self.clinical_data_dir
                )
            except TypeError as e:
                raise ValueError("perform_actual_validation 建立資料集錯誤: {}".format(e))

            next_stages = self.configuration_manager.next_stage_names
            if next_stages is not None:
                # 為每個下一個階段建立預測結果的資料夾
                _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

            results = [] # 用於儲存異步操作的結果物件

            # --- 初始化臨床指標累計變數 ---
            # 這些列表將儲存所有有效案例的真實標籤和預測標籤，以便在所有案例處理完畢後進行整體計算
            all_loc_label_labels = []
            all_t_label_labels = []
            all_n_label_labels = []
            all_m_label_labels = []

            all_pred_loc_labels = []
            all_pred_t_labels = []
            all_pred_n_labels = []
            all_pred_m_labels = []

            all_mask_loc_labels = []
            all_mask_t_labels = []
            all_mask_n_labels = []
            all_mask_m_labels = []

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

                # 載入案例資料：現在 `dataset_val.load_case(k)` 會返回 6 個值，如果 Dataset 是 Multimodal
                load_result = dataset_val.load_case(k)

                image_data_np = None
                seg_prev_np = None
                properties = None
                clinical_data = None
                clinical_mask = None

                # 判斷返回結果的長度，以區分是否為多模態 Dataset
                if len(load_result) == 6: # return data, seg, seg_prev, properties, clinical_data_dict, clinical_mask
                    image_data_np, _, seg_prev_np, properties, clinical_data, clinical_mask = load_result
                else: 
                    # 如果不是多模態 Dataset (例如 Dataset100)，則沒有臨床資料
                    image_data_np, _, seg_prev_np, properties = load_result
                    # 為臨床資料字典設定預設值，以確保程式碼能繼續執行
                    clinical_data = {
                        'location': -1,
                        't_stage': -1,
                        'n_stage': -1,
                        'm_stage': -1
                    }
                    clinical_mask = {
                        'location': False,
                        't_stage': False,
                        'n_stage': False,
                        'm_stage': False
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
                

                # 將臨床特徵字典轉換為 PyTorch 張量
                # clinical_data 是原始標籤，在 validation 我們用其同時作為輸入和標籤，因為不需要資料增強
                # 這與 train_step 和 validation_step 中區分 input 和 label 是不同的
                # 在真實的 inference 階段，實際模型輸入可能是有缺失的，但在 validation 時我們使用完整資料評估模型性能
                loc_label = torch.tensor(clinical_data['location']).to(self.device, non_blocking=True)
                t_label = torch.tensor(clinical_data['t_stage']).to(self.device, non_blocking=True)
                n_label = torch.tensor(clinical_data['n_stage']).to(self.device, non_blocking=True)
                m_label = torch.tensor(clinical_data['m_stage']).to(self.device, non_blocking=True)

                # Missing特徵的遮罩 用於計算指標時忽略
                loc_mask = torch.tensor(clinical_mask['location']).to(self.device, non_blocking=True)
                t_mask = torch.tensor(clinical_mask['t_stage']).to(self.device, non_blocking=True)
                n_mask = torch.tensor(clinical_mask['n_stage']).to(self.device, non_blocking=True)
                m_mask = torch.tensor(clinical_mask['m_stage']).to(self.device, non_blocking=True)



                self.print_to_log_file(f'案例 {k}, 影像形狀 {image_data_tensor.shape}, 當前 rank {self.local_rank}')
                
                output_filename_truncated = join(validation_output_folder, k)

                # **使用 nnUNetPredictorMultimodal 執行滑動視窗預測**
                # 它會同時返回分割的 logits 和臨床屬性預測的 logits
                prediction_seg_logits, prediction_cli_logits = predictor.predict_sliding_window_return_logits(
                    image_data_tensor, loc_label, t_label, n_label, m_label
                )
                prediction_seg_logits = prediction_seg_logits.cpu() # 將分割結果移動到 CPU
               

                # 將臨床屬性 logits 轉換為預測標籤 (argmax for multi-class, sigmoid+threshold for binary)
                pred_loc = prediction_cli_logits['location'].argmax(dim=-1).item()
                pred_t = prediction_cli_logits['t_stage'].argmax(dim=-1).item()
                pred_n = prediction_cli_logits['n_stage'].argmax(dim=-1).item()
                pred_m = prediction_cli_logits['m_stage'].argmax(dim=-1).item()

                # 累計真實和預測標籤到列表中，以便後面計算整體指標
                all_loc_label_labels.append(loc_label)
                all_t_label_labels.append(t_label)
                all_n_label_labels.append(n_label)
                all_m_label_labels.append(m_label)

                all_pred_loc_labels.append(pred_loc)
                all_pred_t_labels.append(pred_t)
                all_pred_n_labels.append(pred_n)
                all_pred_m_labels.append(pred_m)

                all_mask_loc_labels.append(loc_mask)
                all_mask_t_labels.append(t_mask)
                all_mask_n_labels.append(n_mask)
                all_mask_m_labels.append(m_mask)

                # 將器官分割的預測結果匯出為nii.gz檔案 並且工作放入背景行程池中執行 (非同步)
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
                            clinical_data_dir=self.clinical_data_dir if self.is_stage2_dataset else None
                        )

                        try:
                            # 載入下一階段的原始影像數據以獲取目標形狀
                            tmp_dataset = dataset_class_next_stage(
                                expected_preprocessed_folder,
                                [k], # 只需要載入這一個案例
                                clinical_data_dir=self.clinical_data_dir if self.is_stage2_dataset else None 
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
                if self.is_ddp:
                    clinical_vars = [
                        ("all_loc_label_labels", all_loc_label_labels),
                        ("all_t_label_labels", all_t_label_labels),
                        ("all_n_label_labels", all_n_label_labels),
                        ("all_m_label_labels", all_m_label_labels),
                        ("all_pred_loc_labels", all_pred_loc_labels),
                        ("all_pred_t_labels", all_pred_t_labels),
                        ("all_pred_n_labels", all_pred_n_labels),
                        ("all_pred_m_labels", all_pred_m_labels),
                        ("all_mask_loc_labels", all_mask_loc_labels),
                        ("all_mask_t_labels", all_mask_t_labels),
                        ("all_mask_n_labels", all_mask_n_labels),
                        ("all_mask_m_labels", all_mask_m_labels),
                    ]
                    for var_name, var_value in clinical_vars:
                        gathered = [None for _ in range(dist.get_world_size())]
                        dist.all_gather_object(gathered, var_value)
                        locals()[var_name] = [item for sublist in gathered for item in sublist]

                                    
                # --- 計算臨床指標並記錄到 Logger ---
                if self.is_stage2_dataset:
                    self.print_to_log_file("\n--- 臨床屬性驗證結果 ---")

                    # 定義要評估的臨床屬性及其對應的真實標籤、預測標籤、mask、模型輸出類別數量
                    clis_to_evaluate = {
                        'location': (all_loc_label_labels, all_pred_loc_labels, all_mask_loc_labels, self.network.missing_flag_location),
                        't_stage': (all_t_label_labels, all_pred_t_labels, all_mask_t_labels, self.network.missing_flag_t_stage),
                        'n_stage': (all_n_label_labels, all_pred_n_labels, all_mask_n_labels, self.network.missing_flag_n_stage),
                        'm_stage': (all_m_label_labels, all_pred_m_labels, all_mask_m_labels, self.network.missing_flag_m_stage),
                    }

                    for cli_type, (true_labels, pred_labels, mask_labels, num_classes) in clis_to_evaluate.items():
                        metrics = self.compute_clinical_metrics(true_labels, pred_labels, mask_labels, num_classes)
                        # logger 記錄臨床指標
                        self.print_to_log_file(f"- {cli_type} 準確率 (Accuracy): {np.round(metrics['acc'], 4) if metrics['acc'] is not None else 'N/A'}", add_timestamp=False)
                        self.print_to_log_file(f"- {cli_type} F1 分數 (F1 Score): {np.round(metrics['f1'], 4) if metrics['f1'] is not None else 'N/A'}", add_timestamp=False)
                        self.print_to_log_file(f"- {cli_type} Precision: {np.round(metrics['precision'], 4) if metrics['precision'] is not None else 'N/A'}", add_timestamp=False)
                        self.print_to_log_file(f"- {cli_type} Recall: {np.round(metrics['recall'], 4) if metrics['recall'] is not None else 'N/A'}", add_timestamp=False)
                        self.print_to_log_file(f"- {cli_type} 有效比例 (Valid Ratio): {np.round(metrics['valid_ratio'], 4)}", add_timestamp=False)
                        self.print_to_log_file(f"- {cli_type} Per-class Acc: {metrics['per_class_acc']}", add_timestamp=False)
                        if metrics['confusion_matrix'] is not None:
                            self.print_to_log_file(f"- {cli_type} Confusion Matrix:\n{metrics['confusion_matrix']}", add_timestamp=False)

                    # 保存臨床結果到 CSV 檔案
                    csv_path = os.path.join(validation_output_folder, "clinical_infer_results.csv")
                    total_cases = len(all_pred_loc_labels)
                    stat = {}
                    for name, mask_list in zip(
                        ["location", "t_stage", "n_stage", "m_stage"],
                        [all_mask_loc_labels, all_mask_t_labels, all_mask_n_labels, all_mask_m_labels]
                    ):
                        mask_np = np.array([bool(m.item()) if hasattr(m, "item") else bool(m) for m in mask_list])
                        stat[name] = {
                            "total": total_cases,
                            "valid": int(mask_np.sum()),
                            "ratio": float(mask_np.sum()) / total_cases if total_cases > 0 else 0.0
                        }
                    with open(csv_path, "w", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        # 標題
                        writer.writerow([
                            "case_id",
                            "loc_true", "loc_pred", "loc_mask",
                            "t_true", "t_pred", "t_mask",
                            "n_true", "n_pred", "n_mask",
                            "m_true", "m_pred", "m_mask"
                        ])
                        # 寫入每個案例
                        for i in range(total_cases):
                            # 取得原始 label 數值（可能是 tensor 或 int），並轉成 int
                            loc_true_idx = all_loc_label_labels[i].item() if hasattr(all_loc_label_labels[i], "item") else all_loc_label_labels[i]
                            loc_pred_idx = all_pred_loc_labels[i]
                            t_true_idx = all_t_label_labels[i].item() if hasattr(all_t_label_labels[i], "item") else all_t_label_labels[i]
                            t_pred_idx = all_pred_t_labels[i]
                            n_true_idx = all_n_label_labels[i].item() if hasattr(all_n_label_labels[i], "item") else all_n_label_labels[i]
                            n_pred_idx = all_pred_n_labels[i]
                            m_true_idx = all_m_label_labels[i].item() if hasattr(all_m_label_labels[i], "item") else all_m_label_labels[i]
                            m_pred_idx = all_pred_m_labels[i]

                            # 使用 reverse mapping 將索引轉換為文字標籤
                            encoder = dataset_val.clinical_data_label_encoder
                            loc_missing = encoder.missing_flag_location
                            t_missing = encoder.missing_flag_t_stage
                            n_missing = encoder.missing_flag_n_stage
                            m_missing = encoder.missing_flag_m_stage

                            loc_true_str = 'Missing' if loc_true_idx == loc_missing else encoder.reverse_location_mapping.get(loc_true_idx, str(loc_true_idx))
                            loc_pred_str = 'Missing' if loc_pred_idx == loc_missing else encoder.reverse_location_mapping.get(loc_pred_idx, str(loc_pred_idx))
                            t_true_str = 'Missing' if t_true_idx == t_missing else encoder.reverse_t_stage_mapping.get(t_true_idx, str(t_true_idx))
                            t_pred_str = 'Missing' if t_pred_idx == t_missing else encoder.reverse_t_stage_mapping.get(t_pred_idx, str(t_pred_idx))
                            n_true_str = 'Missing' if n_true_idx == n_missing else encoder.reverse_n_stage_mapping.get(n_true_idx, str(n_true_idx))
                            n_pred_str = 'Missing' if n_pred_idx == n_missing else encoder.reverse_n_stage_mapping.get(n_pred_idx, str(n_pred_idx))
                            m_true_str = 'Missing' if m_true_idx == m_missing else encoder.reverse_m_stage_mapping.get(m_true_idx, str(m_true_idx))
                            m_pred_str = 'Missing' if m_pred_idx == m_missing else encoder.reverse_m_stage_mapping.get(m_pred_idx, str(m_pred_idx))

                            # 寫入一行案例資料到 CSV，每個欄位都已經是文字標籤
                            writer.writerow([
                                dataset_val.identifiers[i], # 案例 ID
                                loc_true_str,               # location 真實標籤（文字）
                                loc_pred_str,               # location 預測標籤（文字）
                                "True" if (all_mask_loc_labels[i].item() if hasattr(all_mask_loc_labels[i], "item") else all_mask_loc_labels[i]) else "False", # location 是否有效
                                t_true_str,                 # t_stage 真實標籤（文字）
                                t_pred_str,                 # t_stage 預測標籤（文字）
                                "True" if (all_mask_t_labels[i].item() if hasattr(all_mask_t_labels[i], "item") else all_mask_t_labels[i]) else "False", # t_stage 是否有效
                                n_true_str,                 # n_stage 真實標籤（文字）
                                n_pred_str,                 # n_stage 預測標籤（文字）
                                "True" if (all_mask_n_labels[i].item() if hasattr(all_mask_n_labels[i], "item") else all_mask_n_labels[i]) else "False", # n_stage 是否有效
                                m_true_str,                 # m_stage 真實標籤（文字）
                                m_pred_str,                 # m_stage 預測標籤（文字）
                                "True" if (all_mask_m_labels[i].item() if hasattr(all_mask_m_labels[i], "item") else all_mask_m_labels[i]) else "False", # m_stage 是否有效
                            ])
                        # 統計欄
                        writer.writerow([])
                        writer.writerow(["特徵", "原始數量", "有效數量", "有效率"])
                        for name in ["location", "t_stage", "n_stage", "m_stage"]:
                            writer.writerow([
                                name,
                                stat[name]["total"],
                                stat[name]["valid"],
                                f"{stat[name]['ratio']:.4f}"
                            ])
                    self.print_to_log_file(f"臨床推論結果已保存至 {csv_path}", add_timestamp=False)

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

    def compute_clinical_metrics(self, true_labels, pred_labels, mask, num_classes):
        """
        true_labels, pred_labels, mask: list or numpy array, shape [N]
        num_classes: int
        回傳 dict: acc, f1, precision, recall, confusion_matrix, per_class_acc, valid_count, valid_ratio
        """

        true_labels_np = np.array([t.item() if hasattr(t, 'item') else t for t in true_labels])
        pred_labels_np = np.array(pred_labels)
        mask_np = np.array([m.item() if hasattr(m, 'item') else m for m in mask])
        valid_idx = np.where(mask_np)[0]
        valid_true = true_labels_np[valid_idx]
        valid_pred = pred_labels_np[valid_idx]
    
        valid_count = len(valid_true)
        valid_ratio = valid_count / len(mask_np) if len(mask_np) > 0 else 0
    
        metrics = {
            'valid_count': valid_count,
            'valid_ratio': valid_ratio,
            'acc': None,
            'f1': None,
            'precision': None,
            'recall': None,
            'confusion_matrix': None,
            'per_class_acc': {}
        }
    
        if valid_count > 0:
            metrics['acc'] = accuracy_score(valid_true, valid_pred)
            metrics['f1'] = f1_score(valid_true, valid_pred, average='weighted', zero_division=0)
            metrics['precision'] = precision_score(valid_true, valid_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(valid_true, valid_pred, average='weighted', zero_division=0)
            metrics['confusion_matrix'] = confusion_matrix(valid_true, valid_pred, labels=list(range(num_classes)))
            # 每類別準確率
            for c in range(num_classes):
                idx_c = np.where(valid_true == c)[0]
                if len(idx_c) > 0:
                    correct_pred_c = np.sum(valid_pred[idx_c] == c)
                    class_acc = correct_pred_c / len(idx_c)
                else:
                    class_acc = 0.0
                metrics['per_class_acc'][str(c)] = class_acc
        return metrics