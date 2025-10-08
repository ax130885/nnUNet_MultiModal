# nnunetv2/training/nnUNetTrainer/multitask_segmentation_model.py
import os
import torch
import numpy as np
from torch import nn
from torch.nn.modules.instancenorm import InstanceNorm3d
from torch.nn.modules.activation import LeakyReLU
from nnunetv2.preprocessing.clinical_data_label_encoder import ClinicalDataLabelEncoder
from sentence_transformers import SentenceTransformer

import matplotlib.pyplot as plt
import numpy as np


class TextEncoderModule(nn.Module):
    """
    獨立的文字編碼模組，使用 torch._dynamo.disable 避免被編譯
    """
    def __init__(self, freeze_bert: bool = True):
        super().__init__()
        self.sentence_transformer = SentenceTransformer("neuml/pubmedbert-base-embeddings")
        
        # 凍結權重
        if freeze_bert:
            for param in self.sentence_transformer.parameters():
                param.requires_grad = False
    
    @torch._dynamo.disable  # 這是關鍵：告訴 torch.compile 不要編譯這個方法
    def forward(self, text_descriptions):
        """
        對文字描述進行編碼，此方法不會被 torch.compile 編譯
        """
        embeddings = self.sentence_transformer.encode(
            text_descriptions, 
            convert_to_tensor=True
        )
        return embeddings.detach()


class ASPP3D(nn.Module):
    """
    3D Atrous Spatial Pyramid Pooling (ASPP) 模塊
    使用不同的膨脹率來捕獲多尺度上下文信息
    """
    def __init__(self, in_channels: int, out_channels: int, rates: list = [1, 2, 3, 4]):
        super().__init__()
        
        # 1x1x1 卷積分支
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )
        
        # 多個膨脹卷積分支
        self.atrous_convs = nn.ModuleList()
        for rate in rates:
            self.atrous_convs.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, 3, 
                             padding=rate, dilation=rate, bias=False),
                    InstanceNorm3d(out_channels),
                    nn.LeakyReLU(0.01, inplace=True)
                )
            )
        
        # 全局平均池化分支
        # 注意：池化後是 [B, C, 1, 1, 1]，不能使用 InstanceNorm3d（需要 >1 空間元素）
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels, 1, bias=True),  # 使用 bias 補償沒有 norm
            nn.LeakyReLU(0.01, inplace=True)
        )
        
        # 融合所有分支後的投影層
        # 總共有: 1個1x1卷積 + len(rates)個膨脹卷積 + 1個全局池化 = len(rates)+2 個分支
        total_branches = len(rates) + 2
        self.project = nn.Sequential(
            nn.Conv3d(out_channels * total_branches, out_channels, 1, bias=False),
            InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout3d(0.1)
        )
        
        # 初始化權重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu', a=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, InstanceNorm3d):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        x: [B, C, D, H, W]
        """
        # 保存輸入的空間尺寸
        size = x.shape[2:]
        
        # 1x1 卷積分支
        feat1 = self.conv1(x)
        
        # 膨脹卷積分支
        atrous_feats = [conv(x) for conv in self.atrous_convs]
        
        # 全局平均池化分支
        global_feat = self.global_avg_pool(x)
        # 上採樣到原始尺寸
        global_feat = torch.nn.functional.interpolate(
            global_feat, size=size, mode='trilinear', align_corners=False
        )
        
        # 拼接所有分支
        concat_feats = torch.cat([feat1] + atrous_feats + [global_feat], dim=1)
        
        # 投影到輸出維度
        out = self.project(concat_feats)
        
        return out


class FiLMAddFusion(nn.Module):
    """
    用 FiLM 把臨床向量變成 γ、β，對影像 feature 做逐通道仿射變換。
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

        # 讀取臨床資料的缺失idx
        self.missing_flag_location = encoder.missing_flag_location # 7
        self.missing_flag_t_stage = encoder.missing_flag_t_stage # 5
        self.missing_flag_n_stage = encoder.missing_flag_n_stage # 3
        self.missing_flag_m_stage = encoder.missing_flag_m_stage # 2




        # # ---------- 文字編碼器 ----------
        # # 創建獨立的文字編碼模組，不會被 torch.compile 編譯
        # self.text_encoder = TextEncoderModule(freeze_bert)


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
        
        # 臨床特徵投影層，將所有embedding concate 後(dim=8*4)投影到與影像瓶頸層(dim=320)相同的通道數
        self.clinical_expand = nn.Sequential(
            nn.Linear(8 * 4, 256), # 8是每個嵌入的維度，4是臨床特徵的數量
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 320) # 輸出維度與影像編碼器瓶頸層通道數一致
        )

        # ---------- ASPP 模塊（多層次設計）----------
        # 在不同層使用不同強度的 ASPP
        # 深層（低解析度）：使用完整 ASPP 捕獲大範圍上下文
        # 淺層（高解析度）：不使用 ASPP，保持細節
        self.aspp_modules = nn.ModuleList([
            None,  # stage0 (32通道) - 最高解析度，不用 ASPP
            None,  # stage1 (64通道) - 高解析度，不用 ASPP  
            None,  # stage2 (128通道) - 中解析度，不用 ASPP
            ASPP3D(256, 256, rates=[1, 2]),      # stage3 (256通道) - 輕量 ASPP
            ASPP3D(320, 320, rates=[1, 2, 3]),   # stage4 (320通道) - 中度 ASPP
            ASPP3D(320, 320, rates=[1, 2, 3, 4]) # stage5 (320通道) - 完整 ASPP (瓶頸層)
        ])

        # ---------- 門控混合 跳躍連結(skip)與臨床特徵 ----------
        self.multiscale_fusions = nn.ModuleList([
            GatedFusion(32),
            GatedFusion(64),
            GatedFusion(128),
            # None,
            # None,
            # None,
            GatedFusion(256),
            GatedFusion(320),
            GatedFusion(320)
        ])


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

    def forward(self, img, loc, t, n, m, text_descriptions):
        """
        Args: 輸入模型時 需要全部為 tensor
            img:          [B, C, D, H, W]
            loc:          [B]  整數索引 (含 Missing 索引)
            t:            [B]  整數索引 (含 Missing 索引)
            n:            [B]  整數索引 (含 Missing 索引)
            m:            [B]  整數索引 (含 Missing 索引)
            text_descriptions:     [B]  文字描述列表
        Returns:
            seg_out:       [B, C, D, H, W] 分割頭
            cli_out:       dict 包含臨床資料分類頭輸出
                {
                    'location': [B, C=6]
                    't_stage':  [B, C=5]
                    'n_stage':  [B, C=3]
                    'm_stage':  [B, C=2]
                }
        """
        # ---------- 影像編碼 ----------
        skips = [] # 保存每層下採樣的embedding 用於跳躍連結


        x = img
        for i, stage in enumerate(self.encoder_stages):
            x = stage(x)
            skips.append(x) # 保存每層下採樣的embedding 用於跳躍連結

        bottleneck = skips[-1]

        # ---------- 臨床 Embedding ----------
        loc_emb = self.emb_loc(loc) # [B] -> [B, C=8]
        t_emb   = self.emb_t(t)
        n_emb   = self.emb_n(n)
        m_emb   = self.emb_m(m)
        clinical_vec = torch.cat([loc_emb, t_emb, n_emb, m_emb], dim=1) # [B, C=8] -> [B, 8*4]
        clinical_feat = self.clinical_expand(clinical_vec) # [B, 8*4] -> [B, 320] (符合瓶頸層維度)

        # # ---------- 文字 Embedding ----------
        # # 使用獨立的文字編碼模組，不會被 torch.compile 編譯
        # text_embeddings = self.text_encoder(text_descriptions) # [B, 768]
        # text_embeddings = text_embeddings.to(img.device) # 確保與模型在同一設備上
        # clinical_feat = self.text_expand(text_embeddings) # [B, 768] -> [B, 320] (符合瓶頸層維度)


        # # ---------- 融合 ----------
        # fused_skips = []
        # for i, skip in enumerate(skips):
        #     # # skips = [stage0, stage1, stage2, stage3, stage4, stage5]
        #     # # fused_skips[i] 對應 encoder_stages[i] 的輸出

        #     # # 跳過淺層 計算門控混合的過程
        #     # # 只對深層(低解析度層)進行融合，淺層直接使用原始跳躍連接
        #     # if i < 3:  # stage0, stage1, stage2 (高解析度層)
        #     #     fused_skips.append(None)  # 佔位，保持索引一致
        #     #     continue
            
        #     fused_skip = self.multiscale_fusions[i](skip, clinical_feat)
        #     fused_skips.append(fused_skip)


        # ---------- ASPP + 融合 ----------
        # 關鍵改進：先對影像特徵進行 ASPP 增強，再與臨床特徵融合
        # 這樣 ASPP 只作用於有空間語義的影像特徵
        enhanced_skips = []
        for i, skip in enumerate(skips):
            # 1. 先對影像特徵進行 ASPP（如果該層有配置）
            if self.aspp_modules[i] is not None:
                skip_enhanced = self.aspp_modules[i](skip)
            else:
                skip_enhanced = skip  # 淺層不使用 ASPP
            
            enhanced_skips.append(skip_enhanced)
        
        # 2. 再將增強後的影像特徵與臨床特徵融合
        fused_skips = []
        for i, enhanced_skip in enumerate(enhanced_skips):
            fused_skip = self.multiscale_fusions[i](enhanced_skip, clinical_feat)
            fused_skips.append(fused_skip)

        # ---------- 解碼器 ----------
        lres = fused_skips[-1] # 解碼器輸入
        seg_out = [] # 分割頭輸出 影像

        loc_out = [] # 分類頭輸出 位置
        t_out = []   # 分類頭輸出 時間
        n_out = []   # 分類頭輸出 數量
        m_out = []   # 分類頭輸出 模式

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
                # 臨床資料分類頭
                loc_out.append(self.loc_head(cli_raw_out))
                t_out.append(self.t_head(cli_raw_out))
                n_out.append(self.n_head(cli_raw_out))
                m_out.append(self.m_head(cli_raw_out))

            # 如果關閉深度監督 但是最後一層解碼器 還是要輸出
            elif i == (len(self.decoder_stages) - 1):
                # 分割輸出
                seg_out.append(self.seg_layers[-1](lres))
                # 臨床資料輸出
                cli_raw_out = (self.cli_layers[-1](lres))
                # 臨床資料分類頭
                loc_out.append(self.loc_head(cli_raw_out))
                t_out.append(self.t_head(cli_raw_out))
                n_out.append(self.n_head(cli_raw_out))
                m_out.append(self.m_head(cli_raw_out))

        # 反轉輸出順序
        seg_out = seg_out[::-1] #[start, end, step]
        loc_out = loc_out[::-1]
        t_out = t_out[::-1]
        n_out = n_out[::-1]
        m_out = m_out[::-1]

        # ---------- 屬性預測 ----------
        cli_out = {
            'location': loc_out, # loc_out[0] = [B, C=6]  (若啟動深度監督會有5個解析度的輸出)
            't_stage':  t_out,   # t_out[0] = [B, C=5]
            'n_stage':  n_out,   # n_out[0] = [B, C=3]
            'm_stage':  m_out    # m_out[0] = [B, C=2]
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

    # -------------------- loss & optimizer --------------------
    seg_criterion = nn.CrossEntropyLoss()
    cls_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -------------------- 10 epoch 訓練 --------------------
    model.train()
    for epoch in range(1, epochs+1):
        optimizer.zero_grad()

        seg_out, cli_out = model(img, loc, t, n, m, text_descriptions_train)  # 前向傳播

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

        loss = seg_loss + loc_loss + t_loss + n_loss + m_loss
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch:02d}/{epochs} | loss={loss.item():.4f}")

    # -------------------- 推論示範 --------------------
    with torch.no_grad():
        model.eval()
        seg_out, cli_out = model(img, loc, t, n, m, text_descriptions_train)  # 前向傳播
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