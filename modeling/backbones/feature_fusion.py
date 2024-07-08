import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src):
        return self.transformer_encoder(src)

class FeatureFusion(nn.Module):
    def __init__(self, d_model=768, nhead=8, num_layers=1):
        super(FeatureFusion, self).__init__()
        self.transformer = TransformerEncoder(d_model*2, nhead, num_layers)
        self.fc = nn.Linear(d_model*2, d_model)

    def forward(self, feature1, feature2):
        combined_feature = torch.cat((feature1, feature2), dim=1)
        
        # 通过 Transformer 编码器
        fused_feature = self.transformer(combined_feature)
        
        # 使用一个全连接层融合特征
        fused_feature = self.fc(fused_feature)
        return fused_feature

# 示例使用
batch_size = 10
feature1 = torch.rand(batch_size, 768)
feature2 = torch.rand(batch_size, 768)
num_layers = 3  # 可以调整层数

fusion_model = FeatureFusion(num_layers=num_layers)
fused_feature = fusion_model(feature1, feature2)
