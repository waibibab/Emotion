import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelWiseAttention(nn.Module):
    """特征维度注意力机制"""

    def __init__(self, feat_dim=512):
        super().__init__()
        self.channel_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 16),
            nn.ReLU(),
            nn.Linear(feat_dim // 16, feat_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """输入形状: (B, L, D)"""
        channel_weights = self.channel_proj(x.mean(dim=1))  # (B,D)
        return x * channel_weights.unsqueeze(1)  # (B,1,D)广播到(B,L,D)


class TemporalConvolution(nn.Module):
    """时序特征提取器"""

    def __init__(self, feat_dim=512):
        super().__init__()
        # 深度可分离时序卷积
        self.dw_conv = nn.Conv1d(
            in_channels=feat_dim,
            out_channels=feat_dim,
            kernel_size=3,
            padding=1,
            groups=feat_dim
        )
        # 动态空洞卷积
        self.dilation_conv = nn.Conv1d(
            feat_dim, feat_dim,
            kernel_size=3,
            padding=2,
            dilation=2
        )
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, x):
        """输入形状: (B, L, D)"""
        x = x.transpose(1, 2)  # (B,D,L)
        x = self.dw_conv(x) + self.dilation_conv(x)
        x = self.norm(x.transpose(1, 2))  # (B,L,D)
        return x


class AdaptiveGateFusion(nn.Module):
    """自适应门控融合模块"""

    def __init__(self, feat_dim=512):
        super().__init__()
        # 门控生成器
        self.gate_net = nn.Sequential(
            nn.Linear(2 * feat_dim, 4 * feat_dim),
            nn.ReLU(),
            nn.Linear(4 * feat_dim, feat_dim),
            nn.Sigmoid()
        )

    def forward(self, x, g):
        """融合主输入x和门控信号g"""
        combined = torch.cat([x, g], dim=-1)  # (B,L,2D)
        gate = self.gate_net(combined)  # (B,L,D)
        return gate * x + (1 - gate) * g


class MultiScaleGateAttn(nn.Module):
    def __init__(self, feat_dim=512):
        super().__init__()
        # 时序特征提取
        self.temp_conv = TemporalConvolution(feat_dim)
        # 通道注意力
        self.channel_attn = ChannelWiseAttention(feat_dim)
        # 门控融合
        self.gate_fusion = AdaptiveGateFusion(feat_dim)
        # 输出投影
        self.proj = nn.Linear(feat_dim, feat_dim)

    def forward(self, x, g):
        """输入格式：
        x: 主特征 (B, L, D) = (32,50,512)
        g: 门控特征 (B, L, D)
        """
        # 时序特征提取
        temp_feat = self.temp_conv(x)  # (B,50,512)

        # 通道注意力
        channel_attn_feat = self.channel_attn(temp_feat)  # (B,50,512)

        # 门控融合
        fused = self.gate_fusion(channel_attn_feat, g)  # (B,50,512)

        # 残差连接
        return self.proj(fused) + x


# 测试用例 ---------------------------------------------------
if __name__ == '__main__':
    # 输入格式 (batch, seq_len, feat_dim)
    x = torch.randn(32, 1, 2560)  # 主输入
    g = torch.randn(32, 1, 2560)  # 门控信号

    model = MultiScaleGateAttn(feat_dim=2560)
    out = model(x, g)

    print("输入形状验证:")
    print("x:", x.shape)  # 应输出 torch.Size([32, 50, 512])
    print("输出:", out.shape)  # 应输出 torch.Size([32, 50, 512])
    print("维度一致性检查:", out.shape == x.shape)  # 应输出 True