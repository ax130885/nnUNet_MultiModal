# nnunetv2/training/nnUNetTrainer/multitask_segmentation_model.py
import torch
import numpy as np
from torch import nn
from torch.nn.modules.instancenorm import InstanceNorm3d
from torch.nn.modules.activation import LeakyReLU

class MyModel(nn.Module):
    def __init__(self, input_channels=1, num_classes=2, deep_supervision=True, 
                 prompt_dim=14, location_classes=7, t_stage_classes=6, 
                 n_stage_classes=4, m_stage_classes=3, 
                 missing_flags_dim=4):
        """
        Args:
            input_channels: 影像輸入通道數
            num_classes: 分割類別數
            deep_supervision: 是否使用深度監督
            prompt_dim: 結構化提示的維度 (7位置類別 + 3 TNM類別 + 4缺失標記 = 14)
            location_classes: 腫瘤位置類別數 (7: ascending, transverse, descending, sigmoid, rectal, rectosigmoid, NONE)
            t_stage_classes: T分期類別數 (6: T0, T1, T2, T3, T4, NONE)
            n_stage_classes: N分期類別數 (4: N0, N1, N2, NONE)
            m_stage_classes: M分期類別數 (3: M0, M1, NONE)
            missing_flags_dim: 缺失標記維度 (4: 位置缺失, T缺失, N缺失, M缺失)
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        self.prompt_dim = prompt_dim
        self.missing_flags_dim = missing_flags_dim
        

        self.init_kwargs = {
            'input_channels': input_channels,
            'num_classes': num_classes,
            'deep_supervision': deep_supervision,
            'prompt_dim': prompt_dim,
            'location_classes': location_classes,
            't_stage_classes': t_stage_classes,
            'n_stage_classes': n_stage_classes,
            'm_stage_classes': m_stage_classes,
            'missing_flags_dim': missing_flags_dim
        }

        # ====================== 影像編碼器 ======================
        # 保持原有的影像編碼器結構不變
        self.encoder_stages = nn.ModuleList()
        
        # 階段 0-5 (與原始代碼相同)
        stage0 = nn.Sequential(
            self._create_conv_block(input_channels, 32, kernel_size=3, stride=[1,1,1]),
            self._create_conv_block(32, 32, kernel_size=3, stride=[1,1,1])
        )
        self.encoder_stages.append(stage0)
        
        stage1 = nn.Sequential(
            self._create_conv_block(32, 64, kernel_size=3, stride=[2,2,2]),
            self._create_conv_block(64, 64, kernel_size=3, stride=[1,1,1])
        )
        self.encoder_stages.append(stage1)
        
        stage2 = nn.Sequential(
            self._create_conv_block(64, 128, kernel_size=3, stride=[2,2,2]),
            self._create_conv_block(128, 128, kernel_size=3, stride=[1,1,1])
        )
        self.encoder_stages.append(stage2)
        
        stage3 = nn.Sequential(
            self._create_conv_block(128, 256, kernel_size=3, stride=[2,2,2]),
            self._create_conv_block(256, 256, kernel_size=3, stride=[1,1,1])
        )
        self.encoder_stages.append(stage3)
        
        stage4 = nn.Sequential(
            self._create_conv_block(256, 320, kernel_size=3, stride=[2,2,2]),
            self._create_conv_block(320, 320, kernel_size=3, stride=[1,1,1])
        )
        self.encoder_stages.append(stage4)
        
        stage5 = nn.Sequential(
            self._create_conv_block(320, 320, kernel_size=3, stride=[1,2,2]),
            self._create_conv_block(320, 320, kernel_size=3, stride=[1,1,1])
        )
        self.encoder_stages.append(stage5)
        
        # ====================== 提示編碼器 ======================
        # 添加提示編碼器，將提示向量轉換為與影像特徵相容的表示
        self.prompt_encoder = nn.Sequential(
            nn.Linear(prompt_dim, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 320)
        )
        
        # ====================== 特徵融合模塊 ======================
        # 融合影像特徵和提示特徵
        self.fusion_block = nn.Sequential(
            nn.Conv3d(320*2, 320, kernel_size=1),  # 將拼接的特徵降維回原始通道數
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(320, 320, kernel_size=3, padding=1),
            nn.InstanceNorm3d(320),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        
        # ====================== 解碼器 ======================
        # 保持原有解碼器結構不變
        self.transpconvs = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        
        # 解碼器階段 0 (對應編碼器階段5->4)
        self.transpconvs.append(
            nn.ConvTranspose3d(320, 320, kernel_size=[1,2,2], stride=[1,2,2])
        )
        self.decoder_stages.append(
            self._create_conv_block(320*2, 320, kernel_size=3, stride=1, num_convs=2)
        )
        self.seg_layers.append(
            nn.Conv3d(320, self.num_classes, kernel_size=1)
        )
        
        # 解碼器階段 1 (對應編碼器階段4->3)
        self.transpconvs.append(
            nn.ConvTranspose3d(320, 256, kernel_size=2, stride=2)
        )
        self.decoder_stages.append(
            self._create_conv_block(256*2, 256, kernel_size=3, stride=1, num_convs=2)
        )
        self.seg_layers.append(
            nn.Conv3d(256, self.num_classes, kernel_size=1)
        )
        
        self.transpconvs.append(nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2))
        self.decoder_stages.append(self._create_conv_block(128*2, 128, kernel_size=3, stride=1, num_convs=2))
        self.seg_layers.append(nn.Conv3d(128, self.num_classes, kernel_size=1))
        
        self.transpconvs.append(nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2))
        self.decoder_stages.append(self._create_conv_block(64*2, 64, kernel_size=3, stride=1, num_convs=2))
        self.seg_layers.append(nn.Conv3d(64, self.num_classes, kernel_size=1))
        
        self.transpconvs.append(nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2))
        self.decoder_stages.append(self._create_conv_block(32*2, 32, kernel_size=3, stride=1, num_convs=2))
        self.seg_layers.append(nn.Conv3d(32, self.num_classes, kernel_size=1))
        
        # ====================== 屬性預測頭 ======================
        # 用於預測腫瘤位置和TNM分期
        self.attribute_predictor = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(320, 256),  # 瓶頸層通道數為320
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        
        # 各屬性預測分支
        self.location_head = nn.Linear(128, location_classes)
        self.t_stage_head = nn.Linear(128, t_stage_classes)
        self.n_stage_head = nn.Linear(128, n_stage_classes)
        self.m_stage_head = nn.Linear(128, m_stage_classes)
        
        # 缺失標記預測分支 (新增)
        self.missing_flags_head = nn.Linear(128, missing_flags_dim) if missing_flags_dim > 0 else None
        
        # 權重初始化
        self.apply(self._init_weights)

        # 用 DummyDecoder 只存 deep_supervision 屬性，避免遞迴
        class _DummyDecoder:
            def __init__(self, deep_supervision):
                self.deep_supervision = deep_supervision
        self.decoder = _DummyDecoder(self.deep_supervision)
    
    def _create_conv_block(self, in_channels, out_channels, kernel_size, stride, num_convs=1):
        """創建卷積塊 (與原始代碼相同)"""
        blocks = []
        for i in range(num_convs):
            conv = nn.Conv3d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride if i == 0 else [1, 1, 1],
                padding=[k//2 for k in kernel_size] if isinstance(kernel_size, (list, tuple)) else kernel_size//2,
                bias=True
            )
            blocks.append(conv)
            blocks.append(InstanceNorm3d(out_channels))
            blocks.append(LeakyReLU(negative_slope=0.01, inplace=True))
        return nn.Sequential(*blocks)
    
    def _init_weights(self, m):
        """初始化權重"""
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, a=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, prompt=None):
        """
        前向傳播函數
        Args:
            x: 可以是字典、張量或影像輸入
            - 如果是字典，則必須包含 'data' 作為影像輸入，可選包含 'clinical_features' 和 'has_clinical_data'
            prompt: 可選，直接提供的提示特徵
        Returns:
            seg_outputs: 分割輸出
            attributes: 屬性預測結果的字典
        """

        # 處理字典輸入
        if isinstance(x, dict):
            image = x['data']  # 確保 'data' 是 torch.Tensor
            clinical_features = x.get('clinical_features')
            has_clinical_data = x.get('has_clinical_data')
        else:
            image = x
            clinical_features = prompt
            has_clinical_data = None

        # 確保 clinical_features 是張量
        if clinical_features is not None and not isinstance(clinical_features, torch.Tensor):
            clinical_features = torch.tensor(clinical_features, dtype=torch.float32, device=image.device)
        
        # 如果沒有提供 clinical_features，則使用預設值
        if clinical_features is None:
            batch_size = image.shape[0]
            clinical_features = torch.zeros((batch_size, self.prompt_dim), device=image.device)
        
        # 確保 has_clinical_data 是張量
        if has_clinical_data is not None and not isinstance(has_clinical_data, torch.Tensor):
            has_clinical_data = torch.tensor(has_clinical_data, dtype=torch.bool, device=image.device)
        
        # 確保影像數據是張量
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32, device=clinical_features.device)
        
        # ====================== 影像編碼器前向傳播 ======================
        skips = []
        x_img = image  # 確保 x_img 是 torch.Tensor
        for stage in self.encoder_stages:
            x_img = stage(x_img)  # 每個 stage 都應接收 torch.Tensor
            skips.append(x_img)

        # 瓶頸層特徵
        bottleneck = skips[-1]

        # ====================== 提示編碼器前向傳播 ======================
        prompt_embedding = self.prompt_encoder(clinical_features)
        prompt_expanded = prompt_embedding.view(prompt_embedding.size(0), prompt_embedding.size(1), 1, 1, 1)
        prompt_expanded = prompt_expanded.expand(-1, -1, bottleneck.size(2), bottleneck.size(3), bottleneck.size(4))

        # ====================== 特徵融合 ======================
        fused = torch.cat([bottleneck, prompt_expanded], dim=1)
        fused_bottleneck = self.fusion_block(fused)
        skips[-1] = fused_bottleneck

        # ====================== 解碼器前向傳播 ======================
        lres_input = skips[-1]
        seg_outputs = []
        for i in range(len(self.decoder_stages)):
            x = self.transpconvs[i](lres_input)
            skip = skips[-(i+2)]
            x = torch.cat((x, skip), dim=1)
            x = self.decoder_stages[i](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[i](x))
            elif i == (len(self.decoder_stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x
        seg_outputs = seg_outputs[::-1]

        # ====================== 屬性預測 ======================
        attribute_features = self.attribute_predictor(fused_bottleneck)
        attributes = {
            'location': self.location_head(attribute_features),
            't_stage': self.t_stage_head(attribute_features),
            'n_stage': self.n_stage_head(attribute_features),
            'm_stage': self.m_stage_head(attribute_features)
        }
        if self.missing_flags_head is not None:
            attributes['missing_flags'] = self.missing_flags_head(attribute_features)


        if self.deep_supervision: # 訓練時 啟用深度監督 回傳每層解析度的輸出 並且存為列表
            # seg_outputs shape: torch.Size([1, 2, 112, 160, 128])
            return seg_outputs, attributes
                    
        else:
            # seg_outputs[0] shape: torch.Size([2, 112, 160, 128])
            return seg_outputs[0], attributes # 驗證時 只回傳最後一層的輸出
    
    def compute_conv_feature_map_size(self, input_size):
        """計算卷積特徵圖大小 (與原始代碼相同)"""
        skip_sizes = []
        current_size = input_size
        
        for stride in [[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2], [1,2,2]]:
            skip_sizes.append([i // j for i, j in zip(current_size, stride)])
            current_size = skip_sizes[-1]
        
        output = 0
        
        for i, size in enumerate(skip_sizes):
            for j in range(2):
                output += np.prod([32 * (2**min(i, 4)) if i < 5 else 320, *size], dtype=np.int64)
        
        for i in range(5):
            output += np.prod([self.seg_layers[i].in_channels, *skip_sizes[-(i+1)]], dtype=np.int64)
            for j in range(2):
                output += np.prod([self.decoder_stages[i][j][0].in_channels, *skip_sizes[-(i+1)]], dtype=np.int64)
            output += np.prod([self.num_classes, *skip_sizes[-(i+1)]], dtype=np.int64)
        
        return output


    def set_deep_supervision(self, enabled: bool):
        self.deep_supervision = enabled
        self.decoder.deep_supervision = enabled

# if __name__ == "__main__":
#     import torch
    
#     # 設置設備
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"使用設備: {device}")
    
#     # 設定輸入參數
#     batch_size = 1
#     input_channels = 1
#     depth, height, width = 112, 160, 128  # 輸入的體積大小
#     prompt_dim = 14
    
#     # 創建模型
#     model = MyModel(
#         input_channels=input_channels, 
#         deep_supervision=True, 
#         prompt_dim=prompt_dim
#     ).to(device)
    
#     # 創建隨機輸入數據
#     image = torch.randn(batch_size, input_channels, depth, height, width).to(device)
#     prompt = torch.randn(batch_size, prompt_dim).to(device)
#     print(prompt)
#     # breakpoint()  # 在這裡設置斷點以便調試
    
#     # 顯示輸入形狀
#     print(f"輸入影像形狀: {image.shape}")
#     print(f"輸入提示形狀: {prompt.shape}")
    
#     # 追蹤編碼器每個階段後的大小
#     encoder_outputs = []
#     x_img = image
    
#     print("\n編碼器階段尺寸變化:")
#     print("-" * 50)
    
#     for i, stage in enumerate(model.encoder_stages):
#         x_img = stage(x_img)
#         encoder_outputs.append(x_img)
#         print(f"編碼器階段 {i}: {x_img.shape}")
    
#     # 取得瓶頸層特徵
#     bottleneck = encoder_outputs[-1]
#     print(f"\n瓶頸層形狀: {bottleneck.shape}")
    
#     # 提示編碼器前向傳播
#     prompt_embedding = model.prompt_encoder(prompt)
#     print(f"提示嵌入形狀: {prompt_embedding.shape}")
    
#     # 將提示嵌入擴展為與瓶頸層相同的空間維度
#     prompt_expanded = prompt_embedding.view(prompt_embedding.size(0), prompt_embedding.size(1), 1, 1, 1)
#     prompt_expanded = prompt_expanded.expand(
#         -1, -1, 
#         bottleneck.size(2), 
#         bottleneck.size(3), 
#         bottleneck.size(4)
#     )
#     print(f"擴展後的提示形狀: {prompt_expanded.shape}")
    
#     # 特徵融合
#     fused = torch.cat([bottleneck, prompt_expanded], dim=1)
#     print(f"拼接後的特徵形狀: {fused.shape}")
    
#     fused_bottleneck = model.fusion_block(fused)
#     print(f"融合後的瓶頸層形狀: {fused_bottleneck.shape}")
    
#     # 解碼器每個階段後的大小
#     print("\n解碼器階段尺寸變化:")
#     print("-" * 50)
    
#     # 替換原始瓶頸層
#     encoder_outputs[-1] = fused_bottleneck
    
#     # 追蹤解碼器階段
#     lres_input = encoder_outputs[-1]
    
#     for i in range(len(model.decoder_stages)):
#         # 轉置卷積
#         x = model.transpconvs[i](lres_input)
#         print(f"解碼器階段 {i} (轉置卷積後): {x.shape}")
        
#         # 與跳躍連接特徵拼接
#         skip = encoder_outputs[-(i+2)]
#         print(f"解碼器階段 {i} (跳躍連接特徵): {skip.shape}")
        
#         x = torch.cat((x, skip), dim=1)
#         print(f"解碼器階段 {i} (拼接後): {x.shape}")
        
#         # 卷積塊
#         x = model.decoder_stages[i](x)
#         print(f"解碼器階段 {i} (卷積後): {x.shape}")
        
#         # 分割層
#         seg = model.seg_layers[i](x)
#         print(f"解碼器階段 {i} (分割輸出): {seg.shape}")
        
#         lres_input = x
#         print("-" * 50)
    
#     # 屬性預測
#     attribute_features = model.attribute_predictor(fused_bottleneck)
#     print(f"屬性特徵形狀: {attribute_features.shape}")
    
#     attributes = {
#         'location': model.location_head(attribute_features),
#         't_stage': model.t_stage_head(attribute_features),
#         'n_stage': model.n_stage_head(attribute_features),
#         'm_stage': model.m_stage_head(attribute_features)
#     }
    
#     for attr_name, attr_value in attributes.items():
#         print(f"{attr_name}預測輸出形狀: {attr_value.shape}")
        
#     # 完整前向傳播測試
#     print("\n完整模型前向傳播測試:")
#     print("-" * 50)
    
#     with torch.no_grad():
#         outputs, attr_outputs = model(image, prompt)
        
#         if model.deep_supervision:
#             print(f"分割輸出數量: {len(outputs)}")
#             for i, output in enumerate(outputs):
#                 print(f"分割輸出 {i} 形狀: {output.shape}")
#         else:
#             print(f"分割輸出形狀: {outputs.shape}")
        
#         for attr_name, attr_value in attr_outputs.items():
#             print(f"{attr_name}屬性輸出形狀: {attr_value.shape}")



def calc_receptive_field():
    # 每層卷積的 kernel size 與 stride 配置
    # 按照 encoder 的順序，每層兩個 conv block
    stages = [
        # stage0
        [((3,3,3), (1,1,1)), ((3,3,3), (1,1,1))],
        # stage1
        [((3,3,3), (2,2,2)), ((3,3,3), (1,1,1))],
        # stage2
        [((3,3,3), (2,2,2)), ((3,3,3), (1,1,1))],
        # stage3
        [((3,3,3), (2,2,2)), ((3,3,3), (1,1,1))],
        # stage4
        [((3,3,3), (2,2,2)), ((3,3,3), (1,1,1))],
        # stage5
        [((3,3,3), (1,2,2)), ((3,3,3), (1,1,1))],
    ]

    rf = [1, 1, 1]      # receptive field for d, h, w
    jump = [1, 1, 1]    # actual stride in d, h, w for each output position

    print("Stage, Conv, RF(d,h,w), Jump(d,h,w)")
    for si, stage in enumerate(stages):
        for ci, (kernel, stride) in enumerate(stage):
            rf = [r + (k-1)*j for r, k, j in zip(rf, kernel, jump)]
            jump = [j * s for j, s in zip(jump, stride)]
            print(f"{si:>2}, {ci+1}, {rf}, {jump}")
    print(f"\n最終感受野(Depth, Height, Width): {rf}")
    print(f"最終 jump (實際一格對應原圖幾格): {jump}")

if __name__ == "__main__":
    # ...原本你的測試程式...
    print("\n===== 計算Encoder感受野 =====")
    calc_receptive_field()