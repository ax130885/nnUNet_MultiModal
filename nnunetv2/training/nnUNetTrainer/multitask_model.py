# nnunetv2/training/nnUNetTrainer/multitask_segmentation_model.py
import os
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.instancenorm import InstanceNorm3d
from torch.nn.modules.activation import LeakyReLU
from nnunetv2.preprocessing.clinical_data_label_encoder import ClinicalDataLabelEncoder
from sentence_transformers import SentenceTransformer

import matplotlib.pyplot as plt
import numpy as np


class TextEncoderModule(nn.Module):
    """
    獨立的文字編碼模組，使用 torch._dynamo.disable 避免被編譯
    
    重要：BERT 權重已凍結，但會創建可訓練的輸出副本供下游層使用
    """
    def __init__(self, freeze_bert: bool = True):
        super().__init__()
        self.sentence_transformer = SentenceTransformer("neuml/pubmedbert-base-embeddings")
        
        # 凍結權重
        if freeze_bert:
            for param in self.sentence_transformer.parameters():
                param.requires_grad = False
            # 設置為 eval 模式（這是關鍵！）
            self.sentence_transformer.eval()
    
    @torch._dynamo.disable
    def forward(self, text_descriptions):
        """
        對文字描述進行編碼
        
        策略：
        1. BERT 在 no_grad 下執行（省記憶體，完全凍結）
        2. 創建新的 tensor 副本，允許下游層訓練
        """
        # 使用 no_grad 明確告訴 PyTorch 不追蹤 BERT 的計算圖
        with torch.no_grad():
            embeddings = self.sentence_transformer.encode(
                text_descriptions, 
                convert_to_tensor=True
            )
        
        # 創建一個全新的 tensor，數據相同但可以參與梯度計算
        # 這樣 BERT 完全不參與反向傳播，但下游層可以訓練
        return embeddings.clone().detach().requires_grad_()

class GatedCrossAttentionFusion(nn.Module):
    """
    整合 Gated Attention 機制的 Cross Attention 融合模組
    
    核心創新：基於 NeurIPS 2025 Oral 論文的 Gated Attention 機制
    
    論文要點：
    1. 非線性門控 (Non-linearity): 在 attention 輸出後應用 sigmoid 門控，引入非線性變換
    2. 稀疏性 (Sparsity): 通過門控實現輸入依賴的稀疏性，動態調節資訊流
    3. 無 Attention Sink (Attention-Sink-Free): 避免 attention 過度集中在某些 token
    
    實現方式：
    - Headwise Gating: 每個 attention head 擁有獨立的門控標量
    - 門控信號從 query 投影中額外輸出，與原始 query 一起計算
    - 公式:  attn_output = attn_output * sigmoid(gate_score)
    
    適用場景：
    - 影像特徵:  [B, C, D, H, W] 保留空間資訊
    - 臨床特徵: [B, cli_dim] 全局語義資訊
    - 讓每個空間位置自適應地決定需要多少臨床資訊
    """
    def __init__(self, channels:  int, cli_dim: int = 320, num_heads: int = 8, 
                 dropout: float = 0.1, use_gating: bool = True):
        """
        Args:
            channels:  影像特徵的通道數
            cli_dim: 臨床特徵的維度
            num_heads: attention head 的數量
            dropout: dropout 比率
            use_gating: 是否啟用 gated attention 機制
        """
        super().__init__()
        self.channels = channels
        self. num_heads = num_heads
        self.head_dim = channels // num_heads
        self.use_gating = use_gating
        
        assert channels % num_heads == 0, f"channels ({channels}) 必須能被 num_heads ({num_heads}) 整除"
        
        # ========== Gated Attention 核心實現 ==========
        # 影像特徵投影 (作為 Query)
        # 如果啟用 gating，額外輸出門控信號 (每個 head 一個標量)
        if self.use_gating:
            # 輸出維度:  channels (query) + num_heads (gate scores)
            self.img_query = nn.Conv3d(channels, channels + num_heads, 1)
        else:
            self.img_query = nn.Conv3d(channels, channels, 1)
        
        # 臨床特徵投影 (作為 Key 和 Value)
        # 將單一臨床向量擴展為多個"專家"，每個專家關注不同的臨床aspect
        self.num_clinical_tokens = 4  # 臨床資訊分解為4個token
        self.cli_key = nn.Sequential(
            nn.Linear(cli_dim, channels * self.num_clinical_tokens),
            nn.LayerNorm(channels * self. num_clinical_tokens)
        )
        self.cli_value = nn.Sequential(
            nn.Linear(cli_dim, channels * self.num_clinical_tokens),
            nn.LayerNorm(channels * self. num_clinical_tokens)
        )
        
        # 輸出投影
        self.out_proj = nn.Conv3d(channels, channels, 1)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # 後處理：融合原始影像特徵與注意力結果
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(channels * 2, channels, 3, padding=1),
            InstanceNorm3d(channels),
            nn. GELU(),
            nn.Dropout(dropout),
            nn.Conv3d(channels, channels, 3, padding=1),
            InstanceNorm3d(channels)
        )
        
        # 縮放因子
        self. scale = self.head_dim ** -0.5
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """初始化權重"""
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu', a=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (InstanceNorm3d, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, img_feat, cli_feat):
        """
        img_feat :  [B, C, D, H, W]  影像空間特徵
        cli_feat : [B, cli_dim]     臨床全局特徵
        
        Gated Attention 流程：
        1. 從 query 投影中分離出 gate_score (每個 head 一個標量)
        2. 計算標準的 scaled dot-product attention
        3. 將 attention 輸出與 sigmoid(gate_score) 相乘，實現門控
        4. 門控後的輸出與原始影像特徵融合
        """
        B, C, D, H, W = img_feat.shape
        spatial_size = D * H * W
        
        # ========== 步驟 1: 投影 Query 並分離門控信號 ==========
        if self.use_gating:
            # 投影輸出:  [B, C + num_heads, D, H, W]
            query_and_gate = self.img_query(img_feat)
            
            # 分離 query 和 gate_score
            # query: [B, C, D, H, W]
            # gate_score: [B, num_heads, D, H, W]
            query = query_and_gate[:, : self.channels, :, : , :]
            gate_score = query_and_gate[:, self.channels:, :, : , :]  # [B, num_heads, D, H, W]
            
            # 重塑 gate_score 為 [B, num_heads, D*H*W, 1]
            # 每個 head 在每個空間位置都有一個門控標量
            gate_score = gate_score.view(B, self.num_heads, spatial_size, 1)
        else:
            query = self.img_query(img_feat)  # [B, C, D, H, W]
            gate_score = None
        
        # 重塑 query 為 multi-head 格式
        # [B, C, D, H, W] -> [B, num_heads, head_dim, D*H*W]
        query = query.view(B, self.num_heads, self.head_dim, spatial_size)
        
        # ========== 步驟 2: 投影 Key 和 Value ==========
        # 臨床特徵投影為多個 Key 和 Value tokens
        key_raw = self.cli_key(cli_feat)      # [B, C * num_tokens]
        value_raw = self. cli_value(cli_feat)  # [B, C * num_tokens]
        
        # 重塑為 multi-head 格式
        # [B, C * num_tokens] -> [B, num_heads, head_dim, num_tokens]
        key = key_raw.view(B, self.num_heads, self.head_dim, self.num_clinical_tokens)
        value = value_raw.view(B, self.num_heads, self.head_dim, self.num_clinical_tokens)
        
        # ========== 步驟 3: 計算 Scaled Dot-Product Attention ==========
        # query:   [B, num_heads, head_dim, D*H*W]
        # key:    [B, num_heads, head_dim, num_tokens]
        # scores: [B, num_heads, D*H*W, num_tokens]
        attn_scores = torch.matmul(query. transpose(-2, -1), key) * self.scale
        
        # 在臨床 tokens 維度上做 softmax
        # 每個空間位置決定從哪些臨床 tokens 獲取資訊
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, num_heads, D*H*W, num_tokens]
        attn_weights = self.attn_dropout(attn_weights)
        
        # ========== 步驟 4: 應用 Attention 到 Value ==========
        # attn_weights: [B, num_heads, D*H*W, num_tokens]
        # value:        [B, num_heads, head_dim, num_tokens]
        # attn_output:   [B, num_heads, head_dim, D*H*W]
        attn_output = torch.matmul(value, attn_weights.transpose(-2, -1))
        
        # ========== 步驟 5: 應用 Gated Attention 機制 ==========
        if self.use_gating and gate_score is not None: 
            # gate_score: [B, num_heads, D*H*W, 1]
            # attn_output: [B, num_heads, head_dim, D*H*W]
            
            # 轉置 attn_output 以匹配 gate_score 的維度
            # [B, num_heads, head_dim, D*H*W] -> [B, num_heads, D*H*W, head_dim]
            attn_output = attn_output.transpose(-2, -1)
            
            # 應用門控：attn_output * sigmoid(gate_score)
            # gate_score 會 broadcast 到 head_dim 維度
            # 每個 head 的每個空間位置都有自己的門控值
            attn_output = attn_output * torch.sigmoid(gate_score)
            
            # 轉回原始維度
            # [B, num_heads, D*H*W, head_dim] -> [B, num_heads, head_dim, D*H*W]
            attn_output = attn_output.transpose(-2, -1)
        
        # ========== 步驟 6: 重塑回空間形狀 ==========
        # [B, num_heads, head_dim, D*H*W] -> [B, C, D, H, W]
        attn_output = attn_output.contiguous().view(B, C, D, H, W)
        attn_output = self.out_proj(attn_output)
        attn_output = self.proj_dropout(attn_output)
        
        # ========== 步驟 7: 融合原始影像特徵與注意力增強特徵 ==========
        fused = torch.cat([img_feat, attn_output], dim=1)  # [B, 2C, D, H, W]
        out = self.fusion_conv(fused)  # [B, C, D, H, W]
        
        # ========== 步驟 8: 殘差連接 ==========
        out = out + img_feat
        
        return out


class CrossAttentionFusion(nn.Module):
    """
    使用 Cross Attention 機制融合影像特徵和臨床特徵
    適用於影像分割任務：
    - 影像特徵: [B, C, D, H, W] 保留空間資訊
    - 臨床特徵: [B, cli_dim] 全局語義資訊
    
    策略改進：讓每個空間位置自適應地決定需要多少臨床資訊
    - 影像特徵作為 Query（詢問：我需要什麼臨床資訊？）
    - 臨床特徵作為 Key/Value（提供：這些是可用的臨床資訊）
    """
    def __init__(self, channels: int, cli_dim: int = 320, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, f"channels ({channels}) 必須能被 num_heads ({num_heads}) 整除"
        
        # 影像特徵投影 (作為 Query)
        self.img_query = nn.Conv3d(channels, channels, 1)
        
        # 臨床特徵投影 (作為 Key 和 Value)
        # 將單一臨床向量擴展為多個"專家"，每個專家關注不同的臨床aspect
        self.num_clinical_tokens = 4  # 臨床資訊分解為4個token
        self.cli_key = nn.Sequential(
            nn.Linear(cli_dim, channels * self.num_clinical_tokens),
            nn.LayerNorm(channels * self.num_clinical_tokens)
        )
        self.cli_value = nn.Sequential(
            nn.Linear(cli_dim, channels * self.num_clinical_tokens),
            nn.LayerNorm(channels * self.num_clinical_tokens)
        )
        
        # 輸出投影
        self.out_proj = nn.Conv3d(channels, channels, 1)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # 後處理：融合原始影像特徵與注意力結果
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(channels * 2, channels, 3, padding=1),
            InstanceNorm3d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv3d(channels, channels, 3, padding=1),
            InstanceNorm3d(channels)
        )
        
        # 縮放因子
        self.scale = self.head_dim ** -0.5
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """初始化權重"""
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu', a=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (InstanceNorm3d, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, img_feat, cli_feat):
        """
        img_feat : [B, C, D, H, W]  影像空間特徵
        cli_feat : [B, cli_dim]     臨床全局特徵
        
        改進策略：
        1. 影像每個空間位置作為 Query
        2. 臨床特徵擴展為多個 token 作為 Key/Value
        3. 每個位置自適應地從臨床 tokens 中提取相關資訊
        """
        B, C, D, H, W = img_feat.shape
        spatial_size = D * H * W
        
        # 1. 影像特徵投影為 Query (每個空間位置都是一個 query)
        query = self.img_query(img_feat)  # [B, C, D, H, W]
        # 重塑為 multi-head 格式
        # [B, C, D, H, W] -> [B, num_heads, head_dim, D*H*W]
        query = query.view(B, self.num_heads, self.head_dim, spatial_size)
        
        # 2. 臨床特徵投影為多個 Key 和 Value tokens
        key_raw = self.cli_key(cli_feat)      # [B, C * num_tokens]
        value_raw = self.cli_value(cli_feat)  # [B, C * num_tokens]
        
        # 重塑為 multi-head 格式
        # [B, C * num_tokens] -> [B, num_heads, head_dim, num_tokens]
        key = key_raw.view(B, self.num_heads, self.head_dim, self.num_clinical_tokens)
        value = value_raw.view(B, self.num_heads, self.head_dim, self.num_clinical_tokens)
        
        # 3. 計算 Attention scores
        # query:  [B, num_heads, head_dim, D*H*W]
        # key:    [B, num_heads, head_dim, num_tokens]
        # scores: [B, num_heads, D*H*W, num_tokens]
        attn_scores = torch.matmul(query.transpose(-2, -1), key) * self.scale
        
        # 在臨床 tokens 維度上做 softmax
        # 每個空間位置決定從哪些臨床 tokens 獲取資訊
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, num_heads, D*H*W, num_tokens]
        attn_weights = self.attn_dropout(attn_weights)
        
        # 4. 應用 Attention 到 Value
        # attn_weights: [B, num_heads, D*H*W, num_tokens]
        # value:        [B, num_heads, head_dim, num_tokens]
        # attn_output:  [B, num_heads, head_dim, D*H*W]
        attn_output = torch.matmul(value, attn_weights.transpose(-2, -1))
        
        # 5. 重塑回空間形狀
        # [B, num_heads, head_dim, D*H*W] -> [B, C, D, H, W]
        attn_output = attn_output.contiguous().view(B, C, D, H, W)
        attn_output = self.out_proj(attn_output)
        attn_output = self.proj_dropout(attn_output)
        
        # 6. 融合原始影像特徵與注意力增強特徵
        fused = torch.cat([img_feat, attn_output], dim=1)  # [B, 2C, D, H, W]
        out = self.fusion_conv(fused)  # [B, C, D, H, W]
        
        # 7. 殘差連接
        out = out + img_feat
        
        return out


class CrossAttentionFusion_Reverse(nn.Module):
    """
    反向 Cross Attention: 使用臨床特徵查詢影像特徵
    適用於影像分割任務：
    - 臨床特徵作為 Query（詢問：影像中哪些區域與臨床資訊相關？）
    - 影像特徵作為 Key/Value（提供：這些是影像的空間資訊）
    
    目標：利用臨床資料來指導模型關注影像中的相關區域,提升分割精度
    """
    def __init__(self, channels: int, cli_dim: int = 320, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, f"channels ({channels}) 必須能被 num_heads ({num_heads}) 整除"
        
        # 臨床特徵投影 (作為 Query)
        # 將臨床特徵擴展為多個 query token
        self.num_query_tokens = 4  # 臨床資訊分解為4個query
        self.cli_query = nn.Sequential(
            nn.Linear(cli_dim, channels * self.num_query_tokens),
            nn.LayerNorm(channels * self.num_query_tokens)
        )
        
        # 影像特徵投影 (作為 Key 和 Value)
        self.img_key = nn.Conv3d(channels, channels, 1)
        self.img_value = nn.Conv3d(channels, channels, 1)
        
        # 輸出投影 - 將注意力結果投影回影像空間
        self.out_proj = nn.Sequential(
            nn.Linear(channels * self.num_query_tokens, channels),
            nn.LayerNorm(channels)
        )
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # 後處理：融合原始影像特徵與注意力結果
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(channels * 2, channels, 3, padding=1),
            InstanceNorm3d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv3d(channels, channels, 3, padding=1),
            InstanceNorm3d(channels)
        )
        
        # 縮放因子
        self.scale = self.head_dim ** -0.5
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """初始化權重"""
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu', a=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (InstanceNorm3d, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, img_feat, cli_feat):
        """
        img_feat : [B, C, D, H, W]  影像空間特徵
        cli_feat : [B, cli_dim]     臨床全局特徵
        
        反向策略：
        1. 臨床特徵擴展為多個 Query tokens
        2. 影像每個空間位置作為 Key/Value
        3. 每個臨床 query 自適應地從影像空間中提取相關資訊
        4. 將提取的資訊廣播回影像空間
        """
        B, C, D, H, W = img_feat.shape
        spatial_size = D * H * W
        
        # 1. 臨床特徵投影為多個 Query tokens
        query_raw = self.cli_query(cli_feat)  # [B, C * num_query_tokens]
        # 重塑為 multi-head 格式
        # [B, C * num_query_tokens] -> [B, num_heads, head_dim, num_query_tokens]
        query = query_raw.view(B, self.num_heads, self.head_dim, self.num_query_tokens)
        
        # 2. 影像特徵投影為 Key 和 Value (每個空間位置)
        key = self.img_key(img_feat)    # [B, C, D, H, W]
        value = self.img_value(img_feat)  # [B, C, D, H, W]
        
        # 重塑為 multi-head 格式
        # [B, C, D, H, W] -> [B, num_heads, head_dim, D*H*W]
        key = key.view(B, self.num_heads, self.head_dim, spatial_size)
        value = value.view(B, self.num_heads, self.head_dim, spatial_size)
        
        # 3. 計算 Attention scores
        # query:  [B, num_heads, head_dim, num_query_tokens]
        # key:    [B, num_heads, head_dim, D*H*W]
        # scores: [B, num_heads, num_query_tokens, D*H*W]
        attn_scores = torch.matmul(query.transpose(-2, -1), key) * self.scale
        
        # 在影像空間位置維度上做 softmax
        # 每個臨床 query 決定關注影像中的哪些位置
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, num_heads, num_query_tokens, D*H*W]
        attn_weights = self.attn_dropout(attn_weights)
        
        # 4. 應用 Attention 到 Value
        # attn_weights: [B, num_heads, num_query_tokens, D*H*W]
        # value:        [B, num_heads, head_dim, D*H*W]
        # attn_output:  [B, num_heads, head_dim, num_query_tokens]
        attn_output = torch.matmul(value, attn_weights.transpose(-2, -1))
        
        # 5. 合併多頭輸出
        # [B, num_heads, head_dim, num_query_tokens] -> [B, C * num_query_tokens]
        attn_output = attn_output.contiguous().view(B, -1)
        
        # 6. 投影回通道維度並廣播到空間維度
        attn_output = self.out_proj(attn_output)  # [B, C]
        attn_output = self.proj_dropout(attn_output)
        
        # 廣播到空間維度 [B, C] -> [B, C, D, H, W]
        attn_output = attn_output.view(B, C, 1, 1, 1).expand(-1, -1, D, H, W)
        
        # 7. 融合原始影像特徵與注意力增強特徵
        fused = torch.cat([img_feat, attn_output], dim=1)  # [B, 2C, D, H, W]
        out = self.fusion_conv(fused)  # [B, C, D, H, W]
        
        # 8. 殘差連接
        out = out + img_feat
        
        return out


class FiLMAddFusion(nn.Module):
    """
    用 FiLM 把臨床向量變成 γ、β,對影像 feature 做逐通道仿射變換。
    變換後的 feature 再與 skip 連接的 feature **concat**（不改 nnUNet 邏輯）。
    """
    def __init__(self, channels: int, cli_dim: int):
        super().__init__()
        # 影像投影 (保持通道數)
        self.img_proj = nn.Sequential(
            nn.Conv3d(channels, channels, 1),
            nn.LeakyReLU(0.01),
            InstanceNorm3d(channels)
        )

        # 臨床 → γ、β 兩組通道權重
        self.film = nn.Sequential(
            nn.Linear(cli_dim, channels * 2),
            nn.LeakyReLU(0.01)
        )

        # 後處理卷積
        self.final_fusion = nn.Sequential(
            nn.Conv3d(channels, channels * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(channels * 2, channels, 3, padding=1),
            InstanceNorm3d(channels)
        )
        
        # 使用統一的初始化方法
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """初始化權重，符合 apply 方法的參數要求"""
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu', a=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu', a=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, InstanceNorm3d):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, img_feat, cli_feat):
        """
        img_feat : [B, C, D, H, W]
        cli_feat : [B, cli_dim]  (已經是 320 或 128 維)
        """
        # 1. 影像投影
        x = self.img_proj(img_feat)           # [B, C, D, H, W]

        # 2. FiLM 產生 γ, β
        g_b = self.film(cli_feat)             # [B, 2C]
        gamma, beta = g_b.chunk(2, dim=1)     # 拆成兩個 [B, C]
        B, C = gamma.shape
        gamma = gamma.view(B, C, 1, 1, 1)
        beta  = beta.view(B, C, 1, 1, 1)

        # 3. 仿射變換
        x = x * gamma + beta                  # 逐通道調製

        # 4. 後處理 + 残差
        out = x + self.final_fusion(x)
        return out

class GatedFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        # 改進1：非線性投影層
        # 影像特徵使用Conv3D進行投影
        self.img_proj = nn.Sequential(
            nn.Conv3d(channels, channels, 1),
            nn.LeakyReLU(0.01),
            InstanceNorm3d(channels)
        )
        # 臨床特徵使用MLP而非Conv3D，因為結構化數據無空間關係
        self.cli_proj = nn.Sequential(
            nn.Linear(320, 128), # 先將320維原始臨床特徵投影到128維
            nn.LeakyReLU(0.01),
            nn.Linear(128, channels), # 再將128維投影到與影像特徵相同的通道數 (避免320直接壓到32差距太大)
            nn.LayerNorm(channels)
        )
        
        # 改進2：門控機制: 輸入 拼接兩個特徵的proj，輸出 兩個特徵將乘上的權重張量
        # 混合方式在forwarding內實現，目前是在channel維度進行concate，channel維度會翻倍
        self.gate_conv = nn.Sequential(
            nn.Conv3d(channels*2, channels//2, 3, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv3d(channels//2, channels, 1),
            nn.Sigmoid()
        )

        # 改進3：門控後的結果 套一層MLP再輸出
        self.final_fusion = nn.Sequential(
            nn.Conv3d(channels, channels*2, 3, padding=1),
            nn.GELU(), # 更複雜的 activation function，可以換為 leaky ReLU
            nn.Conv3d(channels*2, channels, 3, padding=1),
            InstanceNorm3d(channels)
        )
        
        # 使用統一的初始化方法
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """初始化權重，符合 apply 方法的參數要求"""
        if isinstance(m, nn.Conv3d):
            # 對於門控機制使用不同的初始化
            if any(m is layer for layer in self.gate_conv):
                # 門控層的權重初始化為接近零的值，使初始門控接近0.5
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            else:
                # 其他卷積層使用Kaiming初始化
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu', a=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu', a=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (InstanceNorm3d, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, img_feat, cli_feat):
        # 影像特徵投影 (conv)
        proj_img = self.img_proj(img_feat)
        # 臨床特徵投影 (MLP)
        proj_cli_1d = self.cli_proj(cli_feat)

        # 將臨床投影改為符合影像投影的維度
         # [B, C] -> [B, C, 1, 1, 1]
        proj_cli = proj_cli_1d.view(*proj_cli_1d.shape, 1, 1, 1)
        # [B, C, 1, 1, 1] -> [B, C, D, H, W]
        proj_cli = proj_cli.expand(-1, -1, *img_feat.shape[2:])  # -1代表不變, [2:] 相當於輸入三個參數, 指定DHW與影像的維度相同
        

        # 雙路門控
        combined = torch.cat([proj_img, proj_cli], dim=1)  # 沿通道維拼接
        gate = self.gate_conv(combined) # 計算權重
        fused = (1 - gate) * proj_img + gate * proj_cli  # 套用權重得到embedding
        out = fused + self.final_fusion(fused) # 套用殘差結構 逐元素相加

        return out



class MyMultiModel(nn.Module):
    def __init__(self,
                 input_channels: int = 1,
                 num_classes: int = 2,
                 deep_supervision: bool = True,
                 clinical_csv_dir: str = '/home/admin/yuxin/data/Lab/model/UNet_base/nnunet_ins_data/data_test/nnUNet_raw/Dataset101',
                 freeze_bert: bool = True):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        # 用 DummyDecoder 只存 deep_supervision 屬性，避免遞迴
        class _DummyDecoder:
            def __init__(self, deep_supervision):
                self.deep_supervision = deep_supervision
        self.decoder = _DummyDecoder(self.deep_supervision)

        ## label encoder
        encoder = ClinicalDataLabelEncoder(clinical_csv_dir)

        # 讀取臨床資料的類別數
        self.num_location_classes = encoder.num_location_classes # 8
        self.num_t_stage_classes = encoder.num_t_stage_classes # 6
        self.num_n_stage_classes = encoder.num_n_stage_classes # 4
        self.num_m_stage_classes = encoder.num_m_stage_classes # 3
        self.num_dataset_classes = encoder.num_dataset_classes # 3

        # 讀取臨床資料的缺失idx
        self.missing_flag_location = encoder.missing_flag_location # 7
        self.missing_flag_t_stage = encoder.missing_flag_t_stage # 5
        self.missing_flag_n_stage = encoder.missing_flag_n_stage # 3
        self.missing_flag_m_stage = encoder.missing_flag_m_stage # 2
        self.missing_flag_dataset = encoder.missing_flag_dataset # 2






        # ---------- 文字編碼器 ----------
        # 創建獨立的文字編碼模組，不會被 torch.compile 編譯
        self.text_encoder = TextEncoderModule(freeze_bert)


        # ---------- 影像編碼器 ----------
        self.encoder_stages = nn.ModuleList()
        
        # 階段 0: 輸入通道=1, 輸出通道=32, 步長=[1,1,1]
        stage0 = nn.Sequential(
            self._create_conv_block(input_channels, 32, kernel_size=3, stride=[1,1,1]),
            self._create_conv_block(32, 32, kernel_size=3, stride=[1,1,1])
        )
        self.encoder_stages.append(stage0)
        
        # 階段 1: 輸入通道=32, 輸出通道=64, 步長=[2,2,2]
        stage1 = nn.Sequential(
            self._create_conv_block(32, 64, kernel_size=3, stride=[2,2,2]),
            self._create_conv_block(64, 64, kernel_size=3, stride=[1,1,1])
        )
        self.encoder_stages.append(stage1)
        
        # 階段 2: 輸入通道=64, 輸出通道=128, 步長=[2,2,2]
        stage2 = nn.Sequential(
            self._create_conv_block(64, 128, kernel_size=3, stride=[2,2,2]),
            self._create_conv_block(128, 128, kernel_size=3, stride=[1,1,1])
        )
        self.encoder_stages.append(stage2)
        
        # 階段 3: 輸入通道=128, 輸出通道=256, 步長=[2,2,2]
        stage3 = nn.Sequential(
            self._create_conv_block(128, 256, kernel_size=3, stride=[2,2,2]),
            self._create_conv_block(256, 256, kernel_size=3, stride=[1,1,1])
        )
        self.encoder_stages.append(stage3)
        
        # 階段 4: 輸入通道=256, 輸出通道=320, 步長=[2,2,2]
        stage4 = nn.Sequential(
            self._create_conv_block(256, 320, kernel_size=3, stride=[2,2,2]),
            self._create_conv_block(320, 320, kernel_size=3, stride=[1,1,1])
        )
        self.encoder_stages.append(stage4)
        
        # 階段 5: 輸入通道=320, 輸出通道=320, 步長=[1,2,2]
        stage5 = nn.Sequential(
            self._create_conv_block(320, 320, kernel_size=3, stride=[1,2,2]),
            self._create_conv_block(320, 320, kernel_size=3, stride=[1,1,1])
        )
        self.encoder_stages.append(stage5)

        # ---------- 臨床資料編碼器 ----------
        # 使用 nn.Embedding 將離散的臨床特徵轉換為連續向量
        # padding_idx 確保缺失值的嵌入向量為零
        self.emb_loc = nn.Embedding(self.num_location_classes, 8, padding_idx=self.missing_flag_location)
        self.emb_t   = nn.Embedding(self.num_t_stage_classes, 8, padding_idx=self.missing_flag_t_stage)
        self.emb_n   = nn.Embedding(self.num_n_stage_classes, 8, padding_idx=self.missing_flag_n_stage)
        self.emb_m   = nn.Embedding(self.num_m_stage_classes, 8, padding_idx=self.missing_flag_m_stage)

        # text_descriptions 投影層 - 改進版
        self.text_expand = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 320),
            nn.LayerNorm(320)
        )

        # # ---------- Dataset 獨立預測頭（從瓶頸層直接預測）----------
        # # 只使用純影像特徵，不融合文字，避免被文字長度等 spurious correlation 影響
        # self.dataset_predictor_from_bottleneck = nn.Sequential(
        #     nn.AdaptiveAvgPool3d(1),  # [B, 320, D, H, W] -> [B, 320, 1, 1, 1]
        #     nn.Flatten(),              # [B, 320, 1, 1, 1] -> [B, 320]
        #     nn.Linear(320, 128),
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU(0.01, inplace=True),
        #     nn.Dropout(0.2),
        #     nn.Linear(128, 64),
        #     nn.BatchNorm1d(64),
        #     nn.LeakyReLU(0.01, inplace=True),
        #     nn.Dropout(0.2),
        #     nn.Linear(64, self.missing_flag_dataset)  # 輸出 dataset 類別數（不含 Missing）
        # )
        
        # # 臨床特徵投影層，將所有embedding concate 後(dim=8*5)投影到與影像瓶頸層(dim=320)相同的通道數
        # self.clinical_expand = nn.Sequential(
        #     nn.Linear(8 * 5, 256), # 8是每個嵌入的維度，5是臨床特徵的數量
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(256, 320) # 輸出維度與影像編碼器瓶頸層通道數一致
        # )


        # # ---------- Gated Cross Attention 融合模組 ----------
        # # 根據 NeurIPS 2025 Oral 論文實現的 Gated Attention
        # # 論文核心貢獻：
        # # 1. 非線性門控提升模型表達能力
        # # 2. 輸入依賴的稀疏性提高訓練穩定性
        # # 3. 避免 Attention Sink 現象，改善長序列處理
        # self.use_gated_attention = True  # 設為 False 可切換回原始版本

        # self.multiscale_fusions = nn.ModuleList([
        #     GatedCrossAttentionFusion(32,  cli_dim=320, num_heads=4, use_gating=self.use_gated_attention),
        #     GatedCrossAttentionFusion(64,  cli_dim=320, num_heads=8, use_gating=self. use_gated_attention),
        #     GatedCrossAttentionFusion(128, cli_dim=320, num_heads=8, use_gating=self. use_gated_attention),
        #     GatedCrossAttentionFusion(256, cli_dim=320, num_heads=8, use_gating=self. use_gated_attention),
        #     GatedCrossAttentionFusion(320, cli_dim=320, num_heads=8, use_gating=self. use_gated_attention),
        #     GatedCrossAttentionFusion(320, cli_dim=320, num_heads=8, use_gating=self. use_gated_attention)
        # ])

        # ---------- Cross Attention 混合 跳躍連結(skip)與臨床特徵 ----------
        # 原版: 影像 Query 臨床 Key/Value (空間適應性)
        self.multiscale_fusions = nn.ModuleList([
            CrossAttentionFusion(32,  cli_dim=320, num_heads=4),   # 32 / 4 = 8 (head_dim)
            CrossAttentionFusion(64,  cli_dim=320, num_heads=8),   # 64 / 8 = 8
            CrossAttentionFusion(128, cli_dim=320, num_heads=8),   # 128 / 8 = 16
            CrossAttentionFusion(256, cli_dim=320, num_heads=8),   # 256 / 8 = 32
            CrossAttentionFusion(320, cli_dim=320, num_heads=8),   # 320 / 8 = 40
            CrossAttentionFusion(320, cli_dim=320, num_heads=8)    # 320 / 8 = 40
        ])

        # # 反向版: 臨床 Query 影像 Key/Value (臨床指導)
        # self.multiscale_fusions = nn.ModuleList([
        #     CrossAttentionFusion_Reverse(32,  cli_dim=320, num_heads=4),   # 32 / 4 = 8 (head_dim)
        #     CrossAttentionFusion_Reverse(64,  cli_dim=320, num_heads=8),   # 64 / 8 = 8
        #     CrossAttentionFusion_Reverse(128, cli_dim=320, num_heads=8),   # 128 / 8 = 16
        #     CrossAttentionFusion_Reverse(256, cli_dim=320, num_heads=8),   # 256 / 8 = 32
        #     CrossAttentionFusion_Reverse(320, cli_dim=320, num_heads=8),   # 320 / 8 = 40
        #     CrossAttentionFusion_Reverse(320, cli_dim=320, num_heads=8)    # 320 / 8 = 40
        # ])

        # # ---------- 門控混合 跳躍連結(skip)與臨床特徵 ----------
        # self.multiscale_fusions = nn.ModuleList([
        #     GatedFusion(32),
        #     GatedFusion(64),
        #     GatedFusion(128),
        #     GatedFusion(256),
        #     GatedFusion(320),
        #     GatedFusion(320)
        # ])

        # # ---------- FiLM混合,  Feature-wise Linear Modulation 跳躍連結(skip)與臨床特徵 ----------
        # self.multiscale_fusions = nn.ModuleList([
        #     FiLMAddFusion(32,  320),
        #     FiLMAddFusion(64,  320),
        #     FiLMAddFusion(128, 320),
        #     FiLMAddFusion(256, 320),
        #     FiLMAddFusion(320, 320),
        #     FiLMAddFusion(320, 320)
        # ])

        # ---------- 解碼器 ----------
        self.transpconvs = nn.ModuleList() # 用於上採樣的轉置卷積層
        self.decoder_stages = nn.ModuleList() # 解碼器階段的卷積層
        self.seg_layers = nn.ModuleList() # 影像資料分割頭
        self.cli_layers = nn.ModuleList() # 臨床資料分類頭

        # 解碼器階段 0 (對應編碼器階段5->4)
        # 多尺度門控融合影像和臨床特徵
        # 上採樣層
        self.transpconvs.append(nn.ConvTranspose3d(320, 320, kernel_size=[1,2,2], stride=[1,2,2]))
        # (上採樣結果 concate 跳躍連接) 降維
        self.decoder_stages.append(self._create_conv_block(320*3, 320, kernel_size=3, stride=1, num_convs=2))
        # 用於深度監督的分割頭
        self.seg_layers.append(nn.Conv3d(320, self.num_classes, kernel_size=1))
        # 深度監督的臨床資料預測頭
        self.cli_layers.append(self._create_cli_predictor(320))

        # 解碼器階段 1 (對應編碼器階段4->3)
        self.transpconvs.append(nn.ConvTranspose3d(320, 256, kernel_size=2, stride=2))
        self.decoder_stages.append(self._create_conv_block(256*3, 256, kernel_size=3, stride=1, num_convs=2))
        self.seg_layers.append(nn.Conv3d(256, num_classes, kernel_size=1))
        self.cli_layers.append(self._create_cli_predictor(256))

        mul_coef = 3
        # 解碼器階段 2 (對應編碼器階段3->2)
        self.transpconvs.append(nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2))
        self.decoder_stages.append(self._create_conv_block(128*mul_coef, 128, kernel_size=3, stride=1, num_convs=2))
        self.seg_layers.append(nn.Conv3d(128, num_classes, kernel_size=1))
        self.cli_layers.append(self._create_cli_predictor(128))

        # 解碼器階段 3 (對應編碼器階段2->1)        
        self.transpconvs.append(nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2))
        self.decoder_stages.append(self._create_conv_block(64*mul_coef, 64, kernel_size=3, stride=1, num_convs=2))
        self.seg_layers.append(nn.Conv3d(64, num_classes, kernel_size=1))
        self.cli_layers.append(self._create_cli_predictor(64))

        # 解碼器階段 4 (對應編碼器階段1->0)        
        self.transpconvs.append(nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2))
        self.decoder_stages.append(self._create_conv_block(32*mul_coef, 32, kernel_size=3, stride=1, num_convs=2))
        self.seg_layers.append(nn.Conv3d(32, num_classes, kernel_size=1))
        self.cli_layers.append(self._create_cli_predictor(32))

        # # 分類頭
        # self.loc_head = nn.Linear(128, self.missing_flag_location)  # 輸出所有類別（不包括Missing對應的索引）
        # self.t_head   = nn.Linear(128, self.missing_flag_t_stage)
        # self.n_head   = nn.Linear(128, self.missing_flag_n_stage)
        # self.m_head   = nn.Linear(128, self.missing_flag_m_stage)

        self.loc_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, self.missing_flag_location)
        )
        self.t_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, self.missing_flag_t_stage)
        )
        self.n_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, self.missing_flag_n_stage)
        )
        self.m_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, self.missing_flag_m_stage)
        )
        self.dataset_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, self.missing_flag_dataset)
        )


        # 初始化權重
        self.apply(self._init_weights)

    # ---------- 工具 ----------
    def _create_conv_block(self, in_ch, out_ch, kernel_size, stride, num_convs=1):
        """自動計算模型的卷積層參數"""
        layers = []
        for i in range(num_convs):
            layers += [nn.Conv3d(in_ch if i == 0 else out_ch,
                                 out_ch, kernel_size, stride if i == 0 else 1,
                                 kernel_size // 2, bias=True),
                       InstanceNorm3d(out_ch),
                       LeakyReLU(0.01, True)]
        return nn.Sequential(*layers)

    def _create_cli_predictor(self, in_channels: int):
        """
        深度監督用: 對每層 創建臨床屬性預測的模塊
        固定輸出128維 交給每個特徵的分類頭進行分類
        """
        return nn.Sequential(
            nn.AdaptiveAvgPool3d(1), # 全局平均池化
            nn.Flatten(),
            nn.Linear(in_channels, 64), # 可以是更小的 MLP
            LeakyReLU(0.01, inplace=True),
            nn.Linear(64, 128) # 輸出到一個固定維度，再分發到各個臨床頭
        )



    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, img, loc, t, n, m, dataset, text_descriptions):
        """
        Args: 輸入模型時 需要全部為 tensor
            img:          [B, C, D, H, W]
            loc:          [B]  整數索引 (含 Missing 索引)
            t:            [B]  整數索引 (含 Missing 索引)
            n:            [B]  整數索引 (含 Missing 索引)
            m:            [B]  整數索引 (含 Missing 索引)
            dataset:      [B]  整數索引 (含 Missing 索引)
            text_descriptions:     [B]  文字描述列表
        Returns:
            seg_out:       [B, C, D, H, W] 分割頭
            cli_out:       dict 包含臨床資料分類頭輸出
                {
                    'location': list of [B, C=6]  # 深度監督
                    't_stage':  list of [B, C=5]
                    'n_stage':  list of [B, C=3]
                    'm_stage':  list of [B, C=2]
                    'dataset':  [B, C=2]  # 單一預測，不使用深度監督
                }
        """
        # ---------- 影像編碼 ----------
        skips = [] # 保存每層下採樣的embedding 用於跳躍連結


        x = img
        i=0
        for stage in self.encoder_stages:
            x = stage(x)
            skips.append(x) # 保存每層下採樣的embedding 用於跳躍連結
            # print(f"Stage: {i}, Input shape: {x.shape}")  # 調試輸出
            i += 1
        bottleneck = skips[-1]  # [B, 320, D, H, W]
        
        # # ✅ 新增：從純影像瓶頸層預測 dataset（不受文字影響）
        # # 這裡在文字融合之前就完成預測，確保 dataset 分類只依賴影像特徵
        # dataset_pred_from_bottleneck = self.dataset_predictor_from_bottleneck(bottleneck)  # [B, num_dataset_classes]

        # # ---------- 臨床 Embedding ----------
        # loc_emb = self.emb_loc(loc) # [B] -> [B, C=8]
        # t_emb   = self.emb_t(t)
        # n_emb   = self.emb_n(n)
        # m_emb   = self.emb_m(m)
        # dataset_emb = self.emb_dataset(dataset)
        # clinical_vec = torch.cat([loc_emb, t_emb, n_emb, m_emb, dataset_emb], dim=1) # [B, C=8] -> [B, 8*5]
        # clinical_feat = self.clinical_expand(clinical_vec) # [B, 8*5] -> [B, 320] (符合瓶頸層維度)

        # ---------- 文字 Embedding ----------
        # 使用獨立的文字編碼模組，不會被 torch.compile 編譯
        text_embeddings = self.text_encoder(text_descriptions) # [B, 768]
        text_embeddings = text_embeddings.to(img.device) # 確保與模型在同一設備上
        clinical_feat = self.text_expand(text_embeddings) # [B, 768] -> [B, 320] (符合瓶頸層維度)


        # ---------- 融合 ----------
        fused_skips = []
        for i, skip in enumerate(skips):
            # # skips = [stage0, stage1, stage2, stage3, stage4, stage5]
            # # fused_skips[i] 對應 encoder_stages[i] 的輸出

            # # 跳過淺層 計算門控混合的過程
            # # 只對深層(低解析度層)進行融合，淺層直接使用原始跳躍連接
            # if i < 3:  # stage0, stage1, stage2 (高解析度層)
            #     fused_skips.append(None)  # 佔位，保持索引一致
            #     continue
            
            fused_skip = self.multiscale_fusions[i](skip, clinical_feat)
            fused_skips.append(fused_skip)

        # ---------- 解碼器 ----------
        lres = fused_skips[-1] # 解碼器輸入
        seg_out = [] # 分割頭輸出 影像

        loc_out = [] # 分類頭輸出 位置
        t_out = []   # 分類頭輸出 時間
        n_out = []   # 分類頭輸出 數量
        m_out = []   # 分類頭輸出 模式
        # ✅ dataset_out 初始化為 None，只在最後一層解碼器賦值
        dataset_out = None # 分類頭輸出 資料集來源

        # 遍歷所有解碼器階段
        for i in range(len(self.decoder_stages)):
            # 1. 上採樣
            lres = self.transpconvs[i](lres)

            ## 2. 拼接 embedding (注意檢查 _create_conv_block 的輸入維度)

            # 注意: 跳躍連接索引是從後往前取
            # print(f"lres shape: {lres.shape}, skip shape: {skips[-(i+2)].shape}")  # 調試輸出

            ## 2.1 原版 直接拚接 1.跳躍連接 2.上採樣結果
            # skip = skips[-(i+2)]  # 從-2開始 下次-3...，跳過瓶頸層
            # lres = torch.cat((lres, skip), dim=1)

            ## 2.2 二版 拼接 1.門控結果 2.上採樣結果
            # skip_to_concat = fused_skips[-(i+2)]  # 從-2開始 下次-3...，跳過瓶頸層
            # lres = torch.cat((lres, skip_to_concat), dim=1)

            # 2.3 三版 拼接 1.門控結果 2.跳躍連接 3.上採樣結果
            skip_to_concat = fused_skips[-(i+2)]  # 從-2開始 下次-3...，跳過瓶頸層
            skip = skips[-(i+2)]
            # 對齊空間維度（處理轉置卷積的邊界對齊問題）
            if lres.shape[2:] != skip.shape[2:]:
                lres = F.interpolate(lres, size=skip.shape[2:], mode='trilinear', align_corners=False)
            lres = torch.cat((lres, skip_to_concat, skip), dim=1)
            

            # # 2.4 根據解析度層級選擇不同的拼接策略
            # skip = skips[-(i+2)]  # 從-2開始 下次-3...，跳過瓶頸層
            
            # if i < 2:  # 前兩層解碼器（低解析度層）
            #     skip_to_concat = fused_skips[-(i+2)]  # 使用門控融合結果
            #     lres = torch.cat((lres, skip_to_concat, skip), dim=1)  # 三路拼接
            # else:  # 高解析度層
            #     lres = torch.cat((lres, skip), dim=1)  # 原始nnUNet風格，二路拼接



            # 3. 使用conv 對拼接後的embedding 上採樣
            lres = self.decoder_stages[i](lres)

            # 4. 深度監督輸出
            if self.deep_supervision:
                # 分割輸出
                seg_out.append(self.seg_layers[i](lres))
                # 臨床資料輸出
                cli_raw_out = (self.cli_layers[i](lres))
                # 臨床資料分類頭（只有 location/T/N/M，不包括 dataset）
                loc_out.append(self.loc_head(cli_raw_out))
                t_out.append(self.t_head(cli_raw_out))
                n_out.append(self.n_head(cli_raw_out))
                m_out.append(self.m_head(cli_raw_out))
                # ✅ dataset 只在最後一層輸出，不使用深度監督
                if i == (len(self.decoder_stages) - 1):
                    dataset_out = self.dataset_head(cli_raw_out)  # [B, num_dataset_classes]


            # 如果關閉深度監督 但是最後一層解碼器 還是要輸出
            elif i == (len(self.decoder_stages) - 1):
                # 分割輸出
                seg_out.append(self.seg_layers[-1](lres))
                # 臨床資料輸出
                cli_raw_out = (self.cli_layers[-1](lres))
                # 臨床資料分類頭（只有 location/T/N/M，不包括 dataset）
                loc_out.append(self.loc_head(cli_raw_out))
                t_out.append(self.t_head(cli_raw_out))
                n_out.append(self.n_head(cli_raw_out))
                m_out.append(self.m_head(cli_raw_out))
                dataset_out = self.dataset_head(cli_raw_out)  # [B, num_dataset_classes]

        # 反轉輸出順序（只有使用深度監督的特徵需要反轉）
        seg_out = seg_out[::-1] #[start, end, step]
        loc_out = loc_out[::-1]
        t_out = t_out[::-1]
        n_out = n_out[::-1]
        m_out = m_out[::-1]
        # dataset_out 是單一張量，不需要反轉


        # ---------- 屬性預測 ----------
        # cli_out = {
        #     'location': loc_out,  # list of [B, C=num_loc_classes]（深度監督）
        #     't_stage':  t_out,    # list of [B, C=num_t_classes]
        #     'n_stage':  n_out,    # list of [B, C=num_n_classes]
        #     'm_stage':  m_out,    # list of [B, C=num_m_classes]
        #     'dataset':  dataset_pred_from_bottleneck  # ✅ 單一張量 [B, C=num_dataset_classes]，不是 list
        # }
        cli_out = {
            'location': loc_out,  # list of [B, C=num_loc_classes]（深度監督）
            't_stage':  t_out,    # list of [B, C=num_t_classes]
            'n_stage':  n_out,    # list of [B, C=num_n_classes]
            'm_stage':  m_out,    # list of [B, C=num_m_classes]
            'dataset':  dataset_out  # ✅ 單一張量 [B, C=num_dataset_classes]，不是 list
        }

        # 如果關閉深度監督 只返回最後一層的輸出 (已經在前面設定 不用在這裡判斷是否啟用)
        return seg_out, cli_out
        

if __name__ == "__main__":
    # -------------------- 超參數 --------------------
    epochs = 10
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------- 模型 & 假資料 --------------------
    model = MyMultiModel(input_channels=1, num_classes=2).to(device)
    img = torch.randn(2, 1, 64, 64, 64).to(device)          # B C D H W
    loc = torch.tensor([5, 2]).to(device)  # B 1，batch 1 輸入 5，batch 2 輸入 2
    t = torch.tensor([3, 1]).to(device)    # B 1，batch 1 輸入 3，batch 2 輸入 1
    n = torch.tensor([2, 0]).to(device)    # B 1，batch 1 輸入 2，batch 2 輸入 0
    m = torch.tensor([1, 0]).to(device)    # B 1，batch 1 輸入 1，batch 2 輸入 0
    dataset = torch.tensor([1, 0]).to(device)  # B 1，batch 1 輸入 1，batch 2 輸入 0
    
    # 文字描述 (訓練用假資料)
    text_descriptions_train = [
        "A computerized tomography scan reveals a colorectal cancer.",
        "A computerized tomography scan reveals a colorectal cancer."
    ]

    # 假 GT（分割與臨床標籤都用 0/1 隨便填）
    seg_gt = torch.randint(0, 2, (2, 64, 64, 64)).long().to(device)  # 與最終層同空間尺寸
    loc_gt = torch.randint(0, model.missing_flag_location, (2,)).to(device)
    t_gt   = torch.randint(0, model.missing_flag_t_stage,   (2,)).to(device)
    n_gt   = torch.randint(0, model.missing_flag_n_stage,   (2,)).to(device)
    m_gt   = torch.randint(0, model.missing_flag_m_stage,   (2,)).to(device)
    dataset_gt = torch.randint(0, model.missing_flag_dataset, (2,)).to(device)

    # -------------------- loss & optimizer --------------------
    seg_criterion = nn.CrossEntropyLoss()
    cls_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -------------------- 10 epoch 訓練 --------------------
    model.train()
    for epoch in range(1, epochs+1):
        optimizer.zero_grad()

        seg_out, cli_out = model(img, loc, t, n, m, dataset, text_descriptions_train)  # 前向傳播

        # 分割 loss：取最後一層（關閉 deep_supervision 時）
        if isinstance(seg_out, list):
            seg_loss = seg_criterion(seg_out[0], seg_gt)
        else:
            seg_loss = seg_criterion(seg_out, seg_gt)

        # 臨床 loss：取最後一層預測
        loc_loss = cls_criterion(cli_out['location'][-1], loc_gt)
        t_loss   = cls_criterion(cli_out['t_stage'][-1],   t_gt)
        n_loss   = cls_criterion(cli_out['n_stage'][-1],   n_gt)
        m_loss   = cls_criterion(cli_out['m_stage'][-1],   m_gt)
        dataset_loss = cls_criterion(cli_out['dataset'], dataset_gt)

        loss = seg_loss + loc_loss + t_loss + n_loss + m_loss + dataset_loss
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch:02d}/{epochs} | loss={loss.item():.4f}")

    # -------------------- 推論示範 --------------------
    with torch.no_grad():
        model.eval()
        seg_out, cli_out = model(img, loc, t, n, m, dataset, text_descriptions_train)  # 前向傳播
        print("\n=== eval shapes ===")
        for seg in (seg_out if isinstance(seg_out, list) else [seg_out]):
            print("seg:", seg.shape)
        for k, v in cli_out.items():
            print(k, [x.shape for x in v])

    print("Toy training done.")


    # ========== 參數檢查 ==========
    print("\n" + "="*50)
    print("模型參數結構分析")
    print("="*50)
    
    all_params = list(model.parameters())
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    trainable_params_set = set(trainable_params_list)  # 修正這行

    total_params = sum(p.numel() for p in all_params)
    trainable_params_count = sum(p.numel() for p in trainable_params_list)
    frozen_params_count = total_params - trainable_params_count

    print(f"總參數量: {total_params / 1e6:.2f}M")
    print(f"可訓練參數量: {trainable_params_count / 1e6:.2f}M")
    print(f"凍結參數量: {frozen_params_count / 1e6:.2f}M")
    
    if frozen_params_count > 0:
        print("\n=== 凍結的參數層 ===")
        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(f"  ❄️  {name:50} | {str(param.shape):20} | {param.numel()/1e3:.1f}K")
    
    print("\n=== 模型結構概覽 (類似 print(model)) ===")
    def print_module_structure(module, indent=0):
        prefix = "  " * indent
        if list(module.children()):
            print(f"{prefix}📦 {module.__class__.__name__}")
            for name, child in module.named_children():
                print(f"{prefix}  ├─ {name}: ", end="")
                if list(child.children()):
                    print()
                    print_module_structure(child, indent + 2)
                else:
                    param_count = sum(p.numel() for p in child.parameters())
                    trainable_count = sum(p.numel() for p in child.parameters() if p.requires_grad)
                    status = " ✓" if trainable_count > 0 else " ❄️"
                    print(f"{child.__class__.__name__} ({param_count/1e3:.1f}K params){status}")
        else:
            param_count = sum(p.numel() for p in module.parameters())
            trainable_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            status = " ✓" if trainable_count > 0 else " ❄️"
            print(f"{module.__class__.__name__} ({param_count/1e3:.1f}K params){status}")
    
    print_module_structure(model)

    # ========== 完整輸入範例 (包含文字描述) ==========
    print("\n" + "="*50)
    print("完整輸入範例測試 (包含文字描述)")
    print("="*50)
    
    # 重新建立模型實例，確保乾淨狀態
    model_test = MyMultiModel(input_channels=1, num_classes=2).to(device)
    
    # 準備完整輸入數據
    batch_size = 2
    img_test = torch.randn(batch_size, 1, 64, 64, 64).to(device)  # [B, C, D, H, W]
    
    # 臨床特徵 (確保在有效範圍內)
    loc_test = torch.tensor([2, 5]).to(device)  # location indices
    t_test = torch.tensor([1, 3]).to(device)    # t_stage indices  
    n_test = torch.tensor([0, 2]).to(device)    # n_stage indices
    m_test = torch.tensor([0, 1]).to(device)    # m_stage indices
    
    # 文字描述範例
    text_descriptions = [
        "A computerized tomography scan reveals a colorectal cancer located in the rectum region, with T stage T2, N stage N0, without distant metastasis.",
        "A computerized tomography scan reveals a colorectal cancer located in the sigmoid colon region, with T stage T4, N stage N2, with distant metastasis."
    ]
    
    print(f"輸入形狀檢查:")
    print(f"  - 影像: {img_test.shape}")
    print(f"  - Location: {loc_test.shape} -> {loc_test.tolist()}")
    print(f"  - T_stage: {t_test.shape} -> {t_test.tolist()}")
    print(f"  - N_stage: {n_test.shape} -> {n_test.tolist()}")
    print(f"  - M_stage: {m_test.shape} -> {m_test.tolist()}")
    print(f"  - 文字描述數量: {len(text_descriptions)}")
    
    # 測試前向傳播
    model_test.eval()
    with torch.no_grad():
        try:
            seg_pred, cli_pred = model_test(img_test, loc_test, t_test, n_test, m_test, text_descriptions)
            
            print(f"\n✅ 模型前向傳播成功!")
            print(f"分割輸出形狀:")
            if isinstance(seg_pred, list):
                for i, seg in enumerate(seg_pred):
                    print(f"  - 層 {i}: {seg.shape}")
            else:
                print(f"  - {seg_pred.shape}")
            
            print(f"臨床預測輸出形狀:")
            for feature, predictions in cli_pred.items():
                print(f"  - {feature}:")
                for i, pred in enumerate(predictions):
                    print(f"    層 {i}: {pred.shape}")
            
            # 顯示預測結果
            print(f"\n預測結果範例 (最後一層):")
            print(f"  Location 預測機率分佈: {torch.softmax(cli_pred['location'][-1], dim=1)}")
            print(f"  T_stage 預測機率分佈: {torch.softmax(cli_pred['t_stage'][-1], dim=1)}")
            print(f"  N_stage 預測機率分佈: {torch.softmax(cli_pred['n_stage'][-1], dim=1)}")
            print(f"  M_stage 預測機率分佈: {torch.softmax(cli_pred['m_stage'][-1], dim=1)}")
            
        except Exception as e:
            print(f"❌ 模型前向傳播失敗: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*50)
    print("測試完成!")
    print("="*50)