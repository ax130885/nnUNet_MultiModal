from typing import Callable

import torch
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from torch import nn


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True, clip_tp: float = None):
        """
        標準的 Soft Dice Loss 實作。
        
        參數:
            apply_nonlin: 用於預測輸出的非線性激活函數 (例如 softmax 或 sigmoid)。
            batch_dice: 
                若為 True，則會將整個 batch 視為一個大樣本來計算 Dice (適用於小目標，可減少波動)。
                若為 False，則對 batch 中的每個樣本單獨計算 Dice 後取平均。
            do_bg: 是否將背景類別 (通常是 index 0) 納入 Loss 計算。
            smooth: 平滑項 (Smoothing factor)，防止分母為 0，並在訓練初期穩定梯度。
            ddp: 是否在分散式訓練 (Distributed Data Parallel) 模式下運行。
            clip_tp: 是否對 True Positive (TP) 進行截斷 (通常較少使用)。
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.clip_tp = clip_tp
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        # x: 網路的輸出 (Logits 或已過激活函數)，形狀通常是 (Batch, Class, X, Y, Z)
        # y: Ground Truth 標籤
        shp_x = x.shape

        # --- 步驟 1: 決定在哪些維度上進行求和 (Summation) ---
        if self.batch_dice:
            # 如果是 Batch Dice，我們要跨越 "Batch 維度 (dim 0)" 以及所有 "空間維度 (dim 2+)" 進行聚合。
            # 也就是把整個 Batch 看作這一個類別的一個大體積。
            axes = [0] + list(range(2, len(shp_x)))
        else:
            # 一般情況，只在 "空間維度 (dim 2+)" 聚合，保留 Batch 維度，每個樣本單獨算 Dice。
            axes = list(range(2, len(shp_x)))

        # --- 步驟 2: 應用激活函數 ---
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # --- 步驟 3: 計算 TP, FP, FN ---
        # 呼叫輔助函數計算 True Positive, False Positive, False Negative
        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        # --- 步驟 4: 處理分散式訓練 (DDP) ---
        if self.ddp and self.batch_dice:
            # 如果是 DDP 且開啟 Batch Dice，需要把所有 GPU 上的 TP, FP, FN 收集起來加總。
            # AllGatherGrad 是一個自定義的 autograd function，支援梯度的反向傳播。
            tp = AllGatherGrad.apply(tp).sum(0)
            fp = AllGatherGrad.apply(fp).sum(0)
            fn = AllGatherGrad.apply(fn).sum(0)

        # --- 步驟 5: (選用) 截斷 TP ---
        if self.clip_tp is not None:
            tp = torch.clip(tp, min=self.clip_tp , max=None)

        # --- 步驟 6: 計算 Dice 公式 ---
        # Dice = (2 * TP) / (2 * TP + FP + FN)
        nominator = 2 * tp
        denominator = 2 * tp + fp + fn

        # 計算 Dice 系數，加上 smooth 防止除以零
        # clip(..., 1e-8) 是為了數值穩定性
        dc = (nominator + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))

        # --- 步驟 7: 處理背景類別與平均 ---
        if not self.do_bg:
            # 如果不計算背景 (通常 index 0 是背景)，則從 index 1 開始取值
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        
        # 對所有剩餘的類別 (以及 batch 中的樣本，如果 batch_dice=False) 取平均
        dc = dc.mean()

        # 返回負的 Dice，因為優化器是做 minimize (最小化)，而我們希望 Dice 最大化。
        return -dc


class MemoryEfficientSoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        記憶體優化版的 Soft Dice Loss。
        
        原本的 SoftDiceLoss 會計算並儲存完整的 fp 和 fn 張量，這在 3D 分割且 Patch 很大時非常佔顯存。
        這個版本利用數學等價公式來減少中間變數的儲存：
        分母 (2*TP + FP + FN) 等價於 (預測總和 + GT 總和)。
        
        註釋提到：在 Dataset017 3d_lowres 任務上可以節省 1.6 GB 顯存。
        """
        super(MemoryEfficientSoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        # --- 步驟 1: 應用激活函數 ---
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # 確定空間維度 (例如 2D 影像就是 (2,3)，3D 影像就是 (2,3,4))
        axes = tuple(range(2, x.ndim))

        # --- 步驟 2: 處理 Ground Truth (GT) ---
        # 使用 no_grad，因為我們不需要對 GT 的處理過程計算梯度，這能節省顯存和計算量
        with torch.no_grad():
            # 確保維度匹配，如果是 (B, X, Y) 轉成 (B, 1, X, Y)
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            # 將 GT 轉換為 One-hot 編碼
            if x.shape == y.shape:
                # 如果形狀一樣，假設已經是 One-hot
                y_onehot = y
            else:
                # 否則進行 scatter 操作轉 One-hot
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)

            # 如果不計算背景，移除第一個 channel
            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            # 計算 GT 的總和 (這是 Dice 分母的一部分)
            # sum_gt 對應數學公式中的 (TP + FN)
            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        # --- 步驟 3: 處理預測值 (Pred) ---
        # 這裡必須在 no_grad 之外，因為我們需要 x 的梯度來更新網路
        if not self.do_bg:
            x = x[:, 1:]

        # 計算交集 (TP) 和 預測總和 (TP + FP)
        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes) # TP
            sum_pred = x.sum(axes)               # TP + FP
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes)
            sum_pred = (x * loss_mask).sum(axes)

        # --- 步驟 4: 處理 DDP 與 Batch Dice ---
        if self.batch_dice:
            if self.ddp:
                # 跨 GPU 聚合數據
                intersect = AllGatherGrad.apply(intersect).sum(0)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0)
            
            # 在 Batch 維度上求和
            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        # --- 步驟 5: 計算 Dice ---
        # 這裡使用了優化後的公式：
        # 分子: 2 * TP
        # 分母: (TP + FP) + (TP + FN) = Sum_Pred + Sum_GT
        # 這避免了顯式計算 FP 和 FN 的大張量
        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))

        dc = dc.mean()
        return dc

class NormalizedSoftDiceLoss(nn.Module):
    """
    正規化版本的 Soft Dice Loss。
    先將 TP, FP, FN 正規化為相對比例（0-1範圍），再計算 Dice。
    這樣可以消除不同解析度層之間的影響，使 smooth 參數在所有層保持一致的效果。
    
    例如：smooth=0.1 表示在所有層都貢獻 0.1 的穩定項
    - 背景patch (TP=0, FP=0.2): Dice = 0.1 / (0.2 + 0.1) = 0.33
    - 前景patch (TP=0.25, FP=0.09, FN=0.06): Dice = (0.5+0.1) / (0.9+0.1) = 0.6
    """
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        super(NormalizedSoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        # --- 步驟 1: 應用激活函數 ---
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # 確定空間維度
        axes = tuple(range(2, x.ndim))

        # --- 步驟 2: 決定在哪些維度上進行求和 ---
        shp_x = x.shape
        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        # --- 步驟 3: 計算 TP, FP, FN, TN ---
        tp, fp, fn, tn = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        # --- 步驟 4: 處理分散式訓練 (DDP) ---
        if self.ddp and self.batch_dice:
            tp = AllGatherGrad.apply(tp).sum(0)
            fp = AllGatherGrad.apply(fp).sum(0)
            fn = AllGatherGrad.apply(fn).sum(0)
            tn = AllGatherGrad.apply(tn).sum(0)  # 【新增】TN 也需要收集

        # --- 步驟 5: 正規化 TP, FP, FN（包含 TN）---
        # 計算總像素數（包含所有：TP + FP + FN + TN）
        total_pixels = tp + fp + fn + tn
        
        # 正規化為相對比例（0-1範圍），所有項加起來 = 1
        tp_normalized = tp / (total_pixels + 1e-8)
        fp_normalized = fp / (total_pixels + 1e-8)
        fn_normalized = fn / (total_pixels + 1e-8)
        tn_normalized = tn / (total_pixels + 1e-8)  # 【新增】TN 也正規化
        
        
        # --- 步驟 6: 使用正規化後的值計算 Dice ---
        # 注意：Dice 公式中不包含 TN，但正規化時已經考慮了 TN
        # 這樣背景patch（TP=0, FN=0）的分母會包含 TN 的影響
        nominator = 2 * tp_normalized
        denominator_normalized = 2 * tp_normalized + fp_normalized + fn_normalized
        
        # 計算 Dice 系數，smooth 直接使用設定值
        dc = (nominator + self.smooth) / (torch.clip(denominator_normalized + self.smooth, 1e-8))

        # --- 步驟 7: 處理背景類別與平均 ---
        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
                # # 【調適損失】打印正規化後的統計（包含 TN）
                # tp_norm_fg = tp_normalized[1:].detach().cpu().numpy() * 100
                # fp_norm_fg = fp_normalized[1:].detach().cpu().numpy() * 100
                # fn_norm_fg = fn_normalized[1:].detach().cpu().numpy() * 100
                # tn_norm_fg = tn_normalized[1:].detach().cpu().numpy() * 100  # 【新增】顯示 TN%
                # smooth_pct = self.smooth * 100
                # print(f"    [Normalized Dice] TP%: {tp_norm_fg}, FP%: {fp_norm_fg}, FN%: {fn_norm_fg}, TN%: {tn_norm_fg}, "
                #       f"Smooth: {smooth_pct}%, Dice: {dc.detach().cpu().numpy()}")
            else:
                dc = dc[:, 1:]
        
        dc = dc.mean()
        return 1-dc

def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    輔助函數：計算 True Positive, False Positive, False Negative, True Negative
    
    參數:
        net_output: 網路輸出 (b, c, x, y(, z))
        gt: Ground Truth 標籤圖 (b, 1, x, y(, z)) 或 one hot (b, c, x, y(, z))
        axes: 要進行求和的維度。如果是 None，則預設為所有空間維度。
        mask: 遮罩，1 為有效像素，0 為無效像素。
        square: 是否在求和前對 tp, fp, fn 進行平方 (Squared Dice Loss 的變體)。
    """
    if axes is None:
        axes = tuple(range(2, net_output.ndim))

    # --- 準備 GT (One-hot 轉換) ---
    with torch.no_grad():
        if net_output.ndim != gt.ndim:
            gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

        if net_output.shape == gt.shape:
            y_onehot = gt
        else:
            y_onehot = torch.zeros(net_output.shape, device=net_output.device, dtype=torch.bool)
            y_onehot.scatter_(1, gt.long(), 1)

    # --- 計算混淆矩陣元素 ---
    # TP: 預測為正 且 真實為正
    tp = net_output * y_onehot
    # FP: 預測為正 但 真實為負 (net_output * (1-y_onehot))
    fp = net_output * (~y_onehot)
    # FN: 預測為負 但 真實為正 ((1-net_output) * y_onehot)
    fn = (1 - net_output) * y_onehot
    # TN: 預測為負 且 真實為負
    tn = (1 - net_output) * (~y_onehot)

    # --- 應用遮罩 (Masking) ---
    if mask is not None:
        with torch.no_grad():
            # 將 mask 複製(tile)到與 tp 相同的 channel 數，以便進行相乘
            mask_here = torch.tile(mask, (1, tp.shape[1], *[1 for _ in range(2, tp.ndim)]))
        tp *= mask_here
        fp *= mask_here
        fn *= mask_here
        tn *= mask_here

    # --- (選用) 平方操作 ---
    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    # --- 聚合 (Summation) ---
    # 在指定的空間維度 (axes) 上將像素值加總，得到每個樣本、每個類別的具體數值
    if len(axes) > 0:
        tp = tp.sum(dim=axes, keepdim=False)
        fp = fp.sum(dim=axes, keepdim=False)
        fn = fn.sum(dim=axes, keepdim=False)
        tn = tn.sum(dim=axes, keepdim=False)

    return tp, fp, fn, tn


if __name__ == '__main__':
    # --- 測試區塊 ---
    from nnunetv2.utilities.helpers import softmax_helper_dim1
    
    # 建立一個隨機預測: batch=2, class=3, size=32x32x32
    pred = torch.rand((2, 3, 32, 32, 32))
    # 建立一個隨機標籤: 值域 0~2
    ref = torch.randint(0, 3, (2, 32, 32, 32))

    # 初始化舊版 Loss
    dl_old = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False)
    # 初始化新版 (省記憶體) Loss
    dl_new = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False)
    
    # 計算並列印結果，理論上兩者數值應該完全相同 (或極度接近)
    res_old = dl_old(pred, ref)
    res_new = dl_new(pred, ref)
    print(res_old, res_new)