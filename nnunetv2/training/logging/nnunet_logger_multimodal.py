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
            'loc_losses': list(),
            't_losses': list(),
            'n_losses': list(),
            'm_losses': list(),
            'missing_flags_losses': list(), # 針對缺失標記的損失

            'loc_accs': list(),
            't_accs': list(),
            'n_accs': list(),
            'm_accs': list(),
            'missing_flags_accs': list() # 針對缺失標記的準確度
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
        # 這裡設 5 個子圖，依序對應：分割損失與dice、臨床損失、臨床acc、epoch耗時、學習率
        fig, ax_all = plt.subplots(5, 1, figsize=(30, 90))
        x_values = list(range(epoch + 1))

        # --- 1. 分割損失和 Dice ---
        ax = ax_all[0]
        ax2 = ax.twinx()
        ax.plot(x_values, self.my_fantastic_logging['train_losses'][:epoch + 1], color='b', ls='-', label="loss_tr", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['val_losses'][:epoch + 1], color='r', ls='-', label="loss_val", linewidth=4)
        if len(self.my_fantastic_logging['mean_fg_dice']) > 0:
            ax2.plot(x_values, self.my_fantastic_logging['mean_fg_dice'][:epoch + 1], color='g', ls='dotted', label="pseudo dice", linewidth=3)
        if len(self.my_fantastic_logging['ema_fg_dice']) > 0:
            ax2.plot(x_values, self.my_fantastic_logging['ema_fg_dice'][:epoch + 1], color='g', ls='-', label="pseudo dice (mov. avg.)", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax2.set_ylabel("pseudo dice")
        ax.legend(loc=(0, 1))
        ax2.legend(loc=(0.2, 1))
        ax.set_title("Segmentation Loss and Dice")

        # --- 2. 臨床分類損失 ---
        ax = ax_all[1]
        ax.plot(x_values, self.my_fantastic_logging['loc_losses'][:epoch + 1], color='purple', ls='-', label="Loc Loss", linewidth=3)
        ax.plot(x_values, self.my_fantastic_logging['t_losses'][:epoch + 1], color='orange', ls='-', label="T Loss", linewidth=3)
        ax.plot(x_values, self.my_fantastic_logging['n_losses'][:epoch + 1], color='brown', ls='-', label="N Loss", linewidth=3)
        ax.plot(x_values, self.my_fantastic_logging['m_losses'][:epoch + 1], color='pink', ls='-', label="M Loss", linewidth=3)
        ax.plot(x_values, self.my_fantastic_logging['missing_flags_losses'][:epoch + 1], color='gray', ls='-', label="Missing Flags Loss", linewidth=3)
        ax.set_xlabel("epoch")
        ax.set_ylabel("Clinical Loss")
        ax.legend(loc=(0, 1))
        ax.set_title("Clinical Classification Losses")

        # --- 3. 臨床分類準確度 ---
        ax = ax_all[2]
        ax.plot(x_values, self.my_fantastic_logging['loc_accs'][:epoch + 1], color='purple', ls='--', label="Loc Acc", linewidth=3)
        ax.plot(x_values, self.my_fantastic_logging['t_accs'][:epoch + 1], color='orange', ls='--', label="T Acc", linewidth=3)
        ax.plot(x_values, self.my_fantastic_logging['n_accs'][:epoch + 1], color='brown', ls='--', label="N Acc", linewidth=3)
        ax.plot(x_values, self.my_fantastic_logging['m_accs'][:epoch + 1], color='pink', ls='--', label="M Acc", linewidth=3)
        ax.plot(x_values, self.my_fantastic_logging['missing_flags_accs'][:epoch + 1], color='gray', ls='--', label="Missing Flags Acc", linewidth=3)
        ax.set_xlabel("epoch")
        ax.set_ylabel("Clinical Accuracy")
        ax.legend(loc=(0, 1))
        ax.set_ylim(0, 1) # 準確度範圍在 0-1 之間
        ax.set_title("Clinical Classification Accuracies")

        # --- 4. Epoch 耗時 ---
        ax = ax_all[3]
        ax.plot(x_values, [i - j for i, j in zip(self.my_fantastic_logging['epoch_end_timestamps'][:epoch + 1],
                                                self.my_fantastic_logging['epoch_start_timestamps'])][:epoch + 1], color='b',
                ls='-', label="epoch duration", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("time [s]")
        ax.legend(loc=(0, 1))
        ax.set_title("Epoch Duration")

        # --- 5. 學習率 ---
        ax = ax_all[4]
        ax.plot(x_values, self.my_fantastic_logging['lrs'][:epoch + 1], color='b', ls='-', label="learning rate", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.legend(loc=(0, 1))
        ax.set_title("Learning Rate")

        plt.tight_layout()
        fig.savefig(join(output_folder, "progress_multimodal.png")) # 另存為不同檔案名
        plt.close()
