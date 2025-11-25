# nnunetv2/training/logging/nnunet_logger_multimodal.py

import matplotlib
matplotlib.use('agg') # 確保在無頭環境下也能工作
import seaborn as sns
import matplotlib.pyplot as plt
from batchgenerators.utilities.file_and_folder_operations import join

# 從 nnUNet 原有日誌檔案中導入 nnUNetLogger
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger

class nnUNetLoggerMultimodal(nnUNetLogger):
    """
    擴展 nnUNetLogger，以包含臨床資料相關的損失和準確度指標。
    """
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        print("nnUNetLoggerMultimodal 初始化。")
        # 添加新的日誌項目
        self.my_fantastic_logging.update({
            'train_total_losses': list(),

            'train_seg_losses': list(),
            'train_loc_losses': list(),
            'train_t_losses': list(),
            'train_n_losses': list(),
            'train_m_losses': list(),
            'train_dataset_losses': list(),

            'val_total_losses': list(),
            'val_seg_losses': list(),
            'val_loc_losses': list(),
            'val_t_losses': list(),
            'val_n_losses': list(),
            'val_m_losses': list(),
            'val_dataset_losses': list(),

            'val_loc_accs': list(),
            'val_t_accs': list(),
            'val_n_accs': list(),
            'val_m_accs': list(),
            'val_dataset_accs': list(),

            'tr_loss_weights_seg': list(),
            'tr_loss_weights_loc': list(),
            'tr_loss_weights_t': list(),
            'tr_loss_weights_n': list(),
            'tr_loss_weights_m': list(),
            'tr_loss_weights_dataset': list(),

            'tr_final_loss_seg': list(),
            'tr_final_loss_loc': list(),
            'tr_final_loss_t': list(),
            'tr_final_loss_n': list(),
            'tr_final_loss_m': list(),
            'tr_final_loss_dataset': list(),

            'val_final_loss_seg': list(),
            'val_final_loss_loc': list(),
            'val_final_loss_t': list(),
            'val_final_loss_n': list(),
            'val_final_loss_m': list(),
            'val_final_loss_dataset': list(),
        })


    def plot_progress_png(self, output_folder):
        """
        繪製訓練進度圖，包括分割和臨床資料的損失及指標。
        覆寫父類的 plot_progress_png 方法。
        """
        # 從內部日誌推斷當前 epoch
        epoch = min([len(i) for i in self.my_fantastic_logging.values()]) - 1
        # 確保 epoch 不為負值
        if epoch < 0:
            print("警告: 尚未有足夠的日誌數據來繪製進度圖。")
            return

        sns.set(font_scale=2.5)
        # 這裡設 11 個子圖，依序對應：分割損失與dice、5個臨床損失(loc,t,n,m,dataset)、臨床acc、epoch耗時、學習率、權重、最終損失
        # fig, ax_all = plt.subplots(8, 1, figsize=(30, 18*8))
        fig, ax_all = plt.subplots(11, 1, figsize=(30, 18*11))
        x_values = list(range(epoch + 1))

        # --- 1. 分割損失和 Dice ---
        ax = ax_all[0]
        ax2 = ax.twinx()
        # 平移分割損失，將最小值從 -1 調整到 0，使其與其他損失在相同尺度
        ax.plot(x_values, [loss + 1.0 for loss in self.my_fantastic_logging['train_seg_losses'][:epoch + 1]], color='b', ls='-', label="loss_tr (+1)", linewidth=4)
        ax.plot(x_values, [loss + 1.0 for loss in self.my_fantastic_logging['val_seg_losses'][:epoch + 1]], color='r', ls='-', label="loss_val (+1)", linewidth=4)
        if len(self.my_fantastic_logging['mean_fg_dice']) > 0:
            ax2.plot(x_values, self.my_fantastic_logging['mean_fg_dice'][:epoch + 1], color='g', ls='dotted', label="pseudo dice", linewidth=3)
        if len(self.my_fantastic_logging['ema_fg_dice']) > 0:
            ax2.plot(x_values, self.my_fantastic_logging['ema_fg_dice'][:epoch + 1], color='g', ls='-', label="pseudo dice (mov. avg.)", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax2.set_ylabel("val pseudo dice")
        ax.legend(loc=(0, 1))
        ax2.legend(loc=(0.2, 1))
        ax.set_title("Segmentation Loss and Dice")

        # --- 2. 臨床分類損失與指標（合併在5張子圖中）---
        loss_names = ['loc', 't', 'n', 'm', 'dataset']
        # loss_names = ['loc', 't']
        for i, loss_name in enumerate(loss_names):
            ax = ax_all[1+i] # ax_all[1] 到 ax_all[5] 分別對應 loc, t, n, m, dataset 的損失+準確度
            ax2 = ax.twinx()

            ax.plot(x_values, self.my_fantastic_logging[f'train_{loss_name}_losses'][:epoch + 1],
                    color='b', ls='-', label=f'Train {loss_name.upper()} Loss', linewidth=4)
            ax.plot(x_values, self.my_fantastic_logging[f'val_{loss_name}_losses'][:epoch + 1],
                    color='r', ls='-', label=f'Val {loss_name.upper()} Loss', linewidth=4)

            ax2.plot(x_values, self.my_fantastic_logging[f'val_{loss_name}_accs'][:epoch + 1],
                    color='g', ls='-', label=f'{loss_name.upper()} Accuracy', linewidth=3)

            ax.set_xlabel('epoch')
            ax.set_ylabel(f'{loss_name.upper()} Loss')
            ax2.set_ylabel(f'{loss_name.upper()} Acc')
            ax.set_title(f'{loss_name.upper()} Loss & Accuracy')

            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax2.set_ylim(0, 1)  # 準確度範圍 0～1

        # --- 3. 臨床分類準確度 ---
        ax = ax_all[6]
        # ax = ax_all[3]
        ax.plot(x_values, self.my_fantastic_logging['val_loc_accs'][:epoch + 1], color='r', ls='-', label="Loc Acc", linewidth=3)
        ax.plot(x_values, self.my_fantastic_logging['val_t_accs'][:epoch + 1], color='g', ls='-', label="T Acc", linewidth=3)
        ax.plot(x_values, self.my_fantastic_logging['val_n_accs'][:epoch + 1], color='c', ls='-', label="N Acc", linewidth=3)
        ax.plot(x_values, self.my_fantastic_logging['val_m_accs'][:epoch + 1], color='m', ls='-', label="M Acc", linewidth=3)
        ax.plot(x_values, self.my_fantastic_logging['val_dataset_accs'][:epoch + 1], color='y', ls='-', label="Dataset Acc", linewidth=3)
        ax.set_xlabel("epoch")
        ax.set_ylabel("Clinical Accuracy")
        ax.legend(loc=(0, 1))
        ax.set_ylim(0, 1) # 準確度範圍在 0-1 之間
        ax.set_title("Clinical Classification Accuracies")

        # --- 4. Epoch 耗時 ---
        ax = ax_all[7]
        # ax = ax_all[4]
        ax.plot(x_values, [i - j for i, j in zip(self.my_fantastic_logging['epoch_end_timestamps'][:epoch + 1],
                                                self.my_fantastic_logging['epoch_start_timestamps'])][:epoch + 1], color='b',
                ls='-', label="epoch duration", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("time [s]")
        ax.legend(loc=(0, 1))
        ax.set_title("Epoch Duration")

        # --- 5. 學習率 ---
        ax = ax_all[8]
        # ax = ax_all[5]
        ax.plot(x_values, self.my_fantastic_logging['lrs'][:epoch + 1], color='b', ls='-', label="learning rate", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.legend(loc=(0, 1))
        ax.set_title("Learning Rate")

        # --- 6. 各損失權重
        ax = ax_all[9]
        ax.plot(x_values, self.my_fantastic_logging['tr_loss_weights_seg'][:epoch + 1], color='b', ls='-', label="seg weight", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['tr_loss_weights_loc'][:epoch + 1], color='r', ls='-', label="loc weight", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['tr_loss_weights_t'][:epoch + 1], color='g', ls='-', label="t weight", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['tr_loss_weights_n'][:epoch + 1], color='c', ls='-', label="n weight", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['tr_loss_weights_m'][:epoch + 1], color='m', ls='-', label="m weight", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['tr_loss_weights_dataset'][:epoch + 1], color='y', ls='-', label="dataset weight", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss weight")
        ax.legend(loc=(0, 1))
        ax.set_title("Loss Weights (Grad Norm ema * Manual Weight)")

        # ---7. 權重*損失
        ax = ax_all[10]
        # 平移分割最終損失和總損失，使最小值從 -1 調整到 0
        ax.plot(x_values, [loss + 1.0 for loss in self.my_fantastic_logging['train_total_losses'][:epoch + 1]], color='purple', ls='-', label="total final loss (+1)", linewidth=4)
        ax.plot(x_values, [loss + 1.0 for loss in self.my_fantastic_logging['tr_final_loss_seg'][:epoch + 1]], color='b', ls='-', label="seg final loss (+1)", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['tr_final_loss_loc'][:epoch + 1], color='r', ls='-', label="loc final loss", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['tr_final_loss_t'][:epoch + 1], color='g', ls='-', label="t final loss", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['tr_final_loss_n'][:epoch + 1], color='c', ls='-', label="n final loss", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['tr_final_loss_m'][:epoch + 1], color='m', ls='-', label="m final loss", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['tr_final_loss_dataset'][:epoch + 1], color='y', ls='-', label="dataset final loss", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("final loss")
        ax.legend(loc=(0, 1))
        ax.set_title("Final Losses (Weight * Loss)")

        # 畫圖
        plt.tight_layout()
        fig.savefig(join(output_folder, "progress_multimodal.png")) # 另存為不同檔案名
        plt.close()