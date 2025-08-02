# nnunetv2/training/nnUNetTrainer/multitask_segmentation_model.py
import os
import torch
import numpy as np
from torch import nn
from torch.nn.modules.instancenorm import InstanceNorm3d
from torch.nn.modules.activation import LeakyReLU
from nnunetv2.preprocessing.clinical_data_label_encoder import ClinicalDataLabelEncoder


import matplotlib.pyplot as plt
import numpy as np

# def visualize_gate(gate,           # [B, C, D, H, W]
#                    batch_idx=0,    # 要看哪一個 batch
#                    channel_idx=0,  # 要看哪一個 channel（可設 None 取平均）
#                    save_path=None):
#     """
#     把 3D gate 權重圖投影成 2D 並顯示 / 儲存
#     gate: torch.Tensor, shape (B, C, D, H, W)
#     """
#     gate_np = gate[batch_idx].detach().cpu().numpy()  # [C, D, H, W]

#     # 如果 channel_idx=None → 取所有 channel 平均
#     if channel_idx is None:
#         gate_2d = np.mean(gate_np, axis=0)            # [D, H, W]
#     else:
#         gate_2d = gate_np[channel_idx]                # [D, H, W]

#     # 沿深度方向再平均 ⇒ [H, W]
#     gate_2d = np.mean(gate_2d, axis=0)

#     plt.figure(figsize=(4, 4))
#     plt.title(f"Gate (avg over depth) - batch {batch_idx}")
#     plt.imshow(gate_2d, cmap='jet', vmin=0, vmax=1)
#     plt.colorbar()
#     plt.axis('off')
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight')
#         print(f"Gate visualization saved to {save_path}")
#         plt.close()
#     else:
#         plt.show()
#         plt.close()
#     torch.cuda.empty_cache()

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

        # # ------- 可視化 gate（只在 eval 或 debug 時用） -------
        # # 避免訓練時拖慢速度
        # if not self.training:
        #     visualize_gate(gate, batch_idx=0, channel_idx=0, save_path=os.path.join(os.getcwd(), f"visualize_gate.png"))

        return out



class MyMultiModel(nn.Module):
    def __init__(self,
                 input_channels: int = 1,
                 num_classes: int = 2,
                 deep_supervision: bool = True,
                 clinical_csv_dir: str = '/home/admin/yuxin/data/Lab/model/UNet_base/nnunet_ins_data/data_test/nnUNet_raw/Dataset101/crcCTlist.csv'):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        # 用 DummyDecoder 只存 deep_supervision 屬性，避免遞迴
        class _DummyDecoder:
            def __init__(self, deep_supervision):
                self.deep_supervision = deep_supervision
        self.decoder = _DummyDecoder(self.deep_supervision)

        encoder = ClinicalDataLabelEncoder(clinical_csv_dir)

        # 讀取臨床資料的類別數
        self.num_location_classes = encoder.num_location_classes # 7
        self.num_t_stage_classes = encoder.num_t_stage_classes # 6
        self.num_n_stage_classes = encoder.num_n_stage_classes # 4
        self.num_m_stage_classes = encoder.num_m_stage_classes # 3

        # 讀取臨床資料的缺失idx
        self.missing_flag_location = encoder.missing_flag_location # 6
        self.missing_flag_t_stage = encoder.missing_flag_t_stage # 5
        self.missing_flag_n_stage = encoder.missing_flag_n_stage # 3
        self.missing_flag_m_stage = encoder.missing_flag_m_stage # 2

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

        # 臨床特徵投影層，將所有embedding concate 後(dim=8*4)投影到與影像瓶頸層(dim=320)相同的通道數
        self.clinical_expand = nn.Sequential(
            nn.Linear(8 * 4, 256), # 8是每個嵌入的維度，4是臨床特徵的數量
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 320) # 輸出維度與影像編碼器瓶頸層通道數一致
        )


        # ---------- 門控混合 跳躍連結(skip)與臨床特徵 ----------
        self.multiscale_fusions = nn.ModuleList([
            GatedFusion(32),
            GatedFusion(64),
            GatedFusion(128),
            GatedFusion(256),
            GatedFusion(320),
            GatedFusion(320)
        ])

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
        self.decoder_stages.append(self._create_conv_block(320*2, 320, kernel_size=3, stride=1, num_convs=2))
        # 用於深度監督的分割頭
        self.seg_layers.append(nn.Conv3d(320, self.num_classes, kernel_size=1))
        # 深度監督的臨床資料預測頭
        self.cli_layers.append(self._create_cli_predictor(320))

        # 解碼器階段 1 (對應編碼器階段4->3)
        self.transpconvs.append(nn.ConvTranspose3d(320, 256, kernel_size=2, stride=2))
        self.decoder_stages.append(self._create_conv_block(256*2, 256, kernel_size=3, stride=1, num_convs=2))
        self.seg_layers.append(nn.Conv3d(256, num_classes, kernel_size=1))
        self.cli_layers.append(self._create_cli_predictor(256))


        # 解碼器階段 2 (對應編碼器階段3->2)
        self.transpconvs.append(nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2))
        self.decoder_stages.append(self._create_conv_block(128*2, 128, kernel_size=3, stride=1, num_convs=2))
        self.seg_layers.append(nn.Conv3d(128, num_classes, kernel_size=1))
        self.cli_layers.append(self._create_cli_predictor(128))

        # 解碼器階段 3 (對應編碼器階段2->1)        
        self.transpconvs.append(nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2))
        self.decoder_stages.append(self._create_conv_block(64*2, 64, kernel_size=3, stride=1, num_convs=2))
        self.seg_layers.append(nn.Conv3d(64, num_classes, kernel_size=1))
        self.cli_layers.append(self._create_cli_predictor(64))

        # 解碼器階段 4 (對應編碼器階段1->0)        
        self.transpconvs.append(nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2))
        self.decoder_stages.append(self._create_conv_block(32*2, 32, kernel_size=3, stride=1, num_convs=2))
        self.seg_layers.append(nn.Conv3d(32, num_classes, kernel_size=1))
        self.cli_layers.append(self._create_cli_predictor(32))

        # 分類頭
        self.loc_head = nn.Linear(128, self.missing_flag_location)  # 輸出所有類別（不包括Missing對應的索引）
        self.t_head   = nn.Linear(128, self.missing_flag_t_stage)
        self.n_head   = nn.Linear(128, self.missing_flag_n_stage)
        self.m_head   = nn.Linear(128, self.missing_flag_m_stage)

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

    def forward(self, img, loc, t, n, m):
        """
        Args: 輸入模型時 需要全部為 tensor
            img:          [B, C, D, H, W]
            loc:          [B]  整數索引 (含 Missing 索引)
            t:            [B]  整數索引 (含 Missing 索引)
            n:            [B]  整數索引 (含 Missing 索引)
            m:            [B]  整數索引 (含 Missing 索引)
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
        i=0
        for stage in self.encoder_stages:
            x = stage(x)
            skips.append(x) # 保存每層下採樣的embedding 用於跳躍連結
            # print(f"Stage: {i}, Input shape: {x.shape}")  # 調試輸出
            i += 1
        bottleneck = skips[-1]

        # ---------- 臨床 Embedding ----------
        loc_emb = self.emb_loc(loc) # [B] -> [B, C=8]
        t_emb   = self.emb_t(t)
        n_emb   = self.emb_n(n)
        m_emb   = self.emb_m(m)
        clinical_vec = torch.cat([loc_emb, t_emb, n_emb, m_emb], dim=1) # [B, C=8] -> [B, 8*4]
        clinical_feat = self.clinical_expand(clinical_vec) # [B, 8*4] -> [B, 320] (符合瓶頸層維度)

        # ---------- 門控融合 ----------
        fused_skips = []
        for i, skip in enumerate(skips):
            # skips = [stage0, stage1, stage2, stage3, stage4, stage5]
            # fused_skips[i] 對應 encoder_stages[i] 的輸出
            fused_skip = self.multiscale_fusions[i](skip, clinical_feat)
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

            # 2. 拼接跳躍連接與上採樣結果 (此處的跳躍連結=影像編碼+臨床編碼的門控融合結果)
            # 注意: 跳躍連接索引是從後往前取
            # print(f"lres shape: {lres.shape}, skip shape: {skips[-(i+2)].shape}")  # 調試輸出
            skip_to_concat = fused_skips[-(i+2)]  # 從-2開始 下次-3...，跳過瓶頸層
            lres = torch.cat((lres, skip_to_concat), dim=1)
            # skip = skips[-(i+2)]  # 從-2開始 下次-3...，跳過瓶頸層
            # lres = torch.cat((lres, skip), dim=1)

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

        # print(f"seg_out len: {len(seg_out)}")  # 假設有啟動深度監督 
        # print(f"loc_out len: {len(loc_out)}")  # 返回的是一個列表 保存每個解析度的輸出
        # print(f"seg_out[0] shape: {seg_out[0].shape}") # [B, C, D, H, W]
        # print(f"loc_out[0] shape: {loc_out[0].shape}") # [B, C=6]
        # print(f"t_out[0] shape: {t_out[0].shape}")   # [B, C=5]
        # print(f"n_out[0] shape: {n_out[0].shape}") # [B, C=3]
        # print(f"m_out[0] shape: {m_out[0].shape}") # [B, C=2]

        # breakpoint()  # 調試用

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

        if self.deep_supervision:
            return seg_out, cli_out
        else:
            return seg_out[0], cli_out[0]
        

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

        seg_out, cli_out = model(img, loc, t, n, m)  # 前向傳播

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
        seg_out, cli_out = model(img, loc, t, n, m)  # 前向傳播
        print("\n=== eval shapes ===")
        for seg in (seg_out if isinstance(seg_out, list) else [seg_out]):
            print("seg:", seg.shape)
        for k, v in cli_out.items():
            print(k, [x.shape for x in v])

    print("Toy training done.")