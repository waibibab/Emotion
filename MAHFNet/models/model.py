import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from block import *
from MAG.MultiScaleGateAttn import MultiScaleGateAttn


class CrossGatedFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 双路门控生成器
        self.gate_linear1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid() 
        )
        self.gate_linear2 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        nn.init.xavier_uniform_(self.gate_linear1[0].weight)
        nn.init.xavier_uniform_(self.gate_linear2[0].weight)

    def forward(self, x1, x2):
        gate1 = self.gate_linear1(x1)  
        gate2 = self.gate_linear2(x2)  
        return gate1 * x2 + gate2 * x1 

def contrastive_loss(contrast_t, contrast_i, temperature=0.07):
    """
    跨模态对比损失函数
    """
    contrast_t = F.normalize(contrast_t, p=2, dim=1)
    contrast_i = F.normalize(contrast_i, p=2, dim=1)
    logits = torch.matmul(contrast_t, contrast_i.T) / temperature
    batch_size = contrast_t.size(0)
    labels = torch.arange(batch_size).to(contrast_t.device)
    loss_t = F.cross_entropy(logits, labels)
    loss_i = F.cross_entropy(logits.T, labels)
    return (loss_t + loss_i) / 2

class AGAI(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        
        self.w_c = nn.Sequential(
            nn.Linear(2*d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
        # 交叉注意力模块
        self.cross_attn_t = Cross_Attention(d_model,num_heads=8)
        self.cross_attn_i = Cross_Attention(d_model,num_heads=8)
        self.attn = Cross_Attention(d_model*2, num_heads=8)
        self.proj = nn.Sequential(
            nn.Linear(2*d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(d_model)
        )
        self.contrast_proj_t = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128)
        )
        self.contrast_proj_i = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128)
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, X_t, X_i):

        X_c = torch.cat((X_t, X_i), dim=-1)
        theta = torch.sigmoid(self.w_c(X_c))
        

        X_t_prime = theta * X_t
        X_i_prime = (1 - theta) * X_i
        

        X_t_tilde = self.cross_attn_t(X_t_prime,X_i)
        X_i_tilde = self.cross_attn_i(X_i_prime,X_t)
        

        fused = torch.cat((X_t_tilde, X_i_tilde), dim=-1)
        fused = self.proj(fused)
        
        # 对比学习损失
        contrast_t = self.contrast_proj_t(X_i_tilde.mean(dim=1))  
        contrast_i = self.contrast_proj_i(X_t_tilde.mean(dim=1))
        contrast_loss = contrastive_loss(contrast_t, contrast_i) * 0.001
        
        return fused, contrast_loss

class HF(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.attn1 = Attention(d_model, 8)
        self.attn2 = Attention(d_model, 8)
        self.linear = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        self.MAG_fusion = MultiScaleGateAttn(d_model)
        self.CrossGated_fusion = CrossGatedFusion(d_model)
    def forward(self, X_st, X_si, X_c):

        X_st = self.attn1(X_st,X_st)
        X_si = self.attn1(X_si,X_si)

        X_s = self.CrossGated_fusion(X_si,X_st)
        X_c_prime = self.MAG_fusion(X_s, X_c)
        return self.linear(X_c_prime)


class BMT(nn.Module):  # BIT
    def __init__(self, dim=512):
        super().__init__()
        self.aai = AGAI(dim)
        self.sa_text =  Cross_Attention(dim, num_heads=8)
        self.sa_image =  Cross_Attention(dim, num_heads=8)
        self.hf = HF(dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(2*dim, 256),
            nn.ReLU(),
            nn.Linear(256, cls_num)
        )
    def forward(self, X_t, X_i):
        # 注意力交互
        X_c,loss = self.aai(X_t, X_i)
        # 情感关联表示
        X_st = self.sa_text(X_t, X_t)
        X_si = self.sa_image(X_i, X_i)
        # 分层融合
        X_final = self.hf(X_st, X_si, X_c)
        # 分类
        logits = self.classifier(X_final)

        return logits,loss

