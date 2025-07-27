# nnunetv2/training/loss/focal_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

class FocalLoss(nn.Module):
    """
    多類別 Focal Loss 實現。
    Args:
        gamma (float): 聚焦參數，用於調整易分樣本的權重。
        alpha (Union[float, list, None]): 平衡參數，可以是單一浮點數（二分類），
                                         類別權重列表，或 None（不使用）。
        reduction (str): 損失的彙總方式 ('mean', 'sum', 'none')。
    """
    def __init__(self, gamma: float = 2.0, alpha: Union[float, list, None] = None, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        計算 Focal Loss。
        Args:
            input (torch.Tensor): 模型的原始輸出 (logits)，形狀為 [N, C] (N: 批次大小, C: 類別數)。
            target (torch.Tensor): 真實標籤，形狀為 [N]，包含 0 到 C-1 的類別索引。
        Returns:
            torch.Tensor: 計算出的 Focal Loss 值。
        """
        # 確保 target 是長整型
        target = target.long()

        # 計算對數機率 (log_softmax) 和機率 (softmax)
        log_prob = F.log_softmax(input, dim=-1)
        prob = torch.exp(log_prob)

        # 獲取真實類別的機率
        # 使用 gather 根據 target 索引從 prob 中提取對應的機率
        # unsqueeze(1) 將 target 變為 [N, 1]，然後 gather 得到 [N, 1]，最後 squeeze(1) 變回 [N]
        prob_target = prob.gather(1, target.unsqueeze(1)).squeeze(1)

        # 計算聚焦項 (1 - pt)^gamma
        focal_term = (1 - prob_target) ** self.gamma

        # 計算交叉熵項 -log(pt)
        ce_loss = -log_prob.gather(1, target.unsqueeze(1)).squeeze(1)

        # 結合聚焦項和交叉熵項
        loss = focal_term * ce_loss

        # 如果設定了 alpha 參數，應用類別權重
        if self.alpha is not None:
            # 將 alpha 權重張量移到與輸入相同的設備上
            if not isinstance(self.alpha, torch.Tensor):
                # 如果 alpha 是浮點數或列表，先轉換為 Tensor
                self.alpha = torch.tensor(self.alpha, dtype=torch.float32, device=input.device)
            elif self.alpha.device != input.device:
                self.alpha = self.alpha.to(input.device)
            
            # 獲取每個樣本對應類別的 alpha 權重
            alpha_t = self.alpha.gather(0, target)
            loss = alpha_t * loss

        # 應用約簡方式
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else: # 'none'
            return loss