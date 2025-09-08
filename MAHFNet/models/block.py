import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    '''
    self-attention
    '''
    def __init__(self, dim=512, num_heads=8, attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x1, x2):
        B, N_1, C = x1.shape
        N_2 = x2.shape[1]  # n2 may not = n1
        # print(x1.device)
        # print(x2.device)
        q = self.q(x1).reshape(B,  N_1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # local input
        # b,h,n,64
        k = self.k(x2).reshape(B,  N_2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x2).reshape(B,  N_2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
       # b , h , N , dim/h
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N_1, C) # q's shape.
        return x


class Cross_Attention(nn.Module):
    '''
    cross-attention
    inputs: X and Y from image and text. X represent local input, Y represent other input.
            shapes of X and Y may be different. X as q, Y as k,v. X interact with Y to query sentimental information
            embedded in Y.
    output:
    '''
    def __init__(self, dim=512, num_heads=8, attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x1, x2):
        B, N_1, C = x1.shape
        N_2 = x2.shape[1]  # n2 may not = n1
        # print(x1.device)
        # print(x2.device)
        q = self.q(x1).reshape(B,  N_1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # local input
        # b,h,n,64
        k = self.k(x2).reshape(B,  N_2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x2).reshape(B,  N_2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
       # b , h , N , dim/h
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N_1, C) # q's shape.
        return x


class CrossModalBlock(nn.Module):
    '''完整的跨模态Transformer块'''

    def __init__(self, dim=512, num_heads=8, attn_drop=0.1, mlp_ratio=4):
        super().__init__()
        self.cross_attn = Cross_Attention(dim, num_heads, attn_drop)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # FFN部分
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(attn_drop),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(attn_drop)
        )

    def forward(self, x1, x2):
        # 跨模态注意力 + 残差
        x1 = self.norm1(x1 + self.cross_attn(x1, x2))
        # FFN + 残差
        x1 = self.norm2(x1 + self.mlp(x1))
        return x1
class Cross_Attention_without_pro(nn.Module):
    def __init__(self, dim=512, num_heads=8, groups=32, attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.groups = groups
        self.head_dim = dim // num_heads
        self.scale = self.head_dim  **  -0.5

        # 查询分支
        self.q_proj = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=groups),
            nn.GroupNorm(groups, dim),
            nn.Conv1d(dim, dim, kernel_size=1)
        )

        # 键值分支
        self.kv_proj = nn.Sequential(
            nn.Conv1d(dim, dim * 2, kernel_size=1, groups=groups),
            nn.GroupNorm(groups * 2, dim * 2),
            nn.GELU()
        )

        # 空间注意力门控
        self.spatial_gate = nn.Sequential(
            nn.Conv1d(dim, dim // 4, kernel_size=1),
            nn.GroupNorm(groups // 4, dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 通道注意力门控（修正分组维度）
        self.channel_gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GroupNorm(num_heads, dim // 4),  # 关键修改：按注意力头数分组
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, num_heads),  # 输出每个头的权重
            nn.Sigmoid()
        )

        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x1, x2):
        B, N1, C = x1.shape
        N2 = x2.shape[1]

        # 查询投影
        q = self.q_proj(x1.transpose(1, 2)).transpose(1, 2)  # [B, N1, C]

        # 键值投影
        kv = self.kv_proj(x2.transpose(1, 2))  # [B, 2C, N2]
        k, v = torch.chunk(kv, 2, dim=1)
        k = k.transpose(1, 2)  # [B, N2, C]
        v = v.transpose(1, 2)

        # 多头拆分
        q = q.view(B, N1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, h, N1, d]
        k = k.view(B, N2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, N2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 基础注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, h, N1, N2]

        # 空间注意力（修正广播机制）
        spatial_weight = self.spatial_gate(x1.transpose(1, 2))  # [B, 1, N1]
        attn = attn * spatial_weight.view(B, 1, N1, 1)  # [B, h, N1, N2]

        # 通道注意力（关键修正）
        channel_weight = self.channel_gate(x1.mean(dim=1))  # [B, h]
        attn = attn * channel_weight.view(B, self.num_heads, 1, 1)  # [B, h, 1, 1]

        # 归一化输出
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        return x

class Cross_Attention_with_pro(nn.Module):
    '''
    cross-attention
    inputs: X and Y from image and text. X represent local input, Y represent other input.
            shapes of X and Y may be different. X as q, Y as k,v. X interact with Y to query sentimental information
            embedded in Y.
    output:
    '''
    def __init__(self, dim=512, num_heads=8, attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.project = nn.Linear(dim, dim)

    def forward(self, x1, x2):
        B, N_1, C = x1.shape
        N_2 = x2.shape[1]  # n2 may not = n1
        # print(x1.device)
        # print(x2.device)
        q = self.q(x1).reshape(B,  N_1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # local input
        # b,h,n,64
        k = self.k(x2).reshape(B,  N_2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x2).reshape(B,  N_2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
       # b , h , N , dim/h
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N_1, C) # q's shape.
        x = self.project(x)
        return x

class MemoryInit(nn.Module): # prefix init
    def __init__(self, n_memory_cells, dim):
        super(MemoryInit, self).__init__()
        # init memory
        self.n_memory_cells = n_memory_cells
        self.init_memory_bias = nn.Parameter(
            torch.randn(1, n_memory_cells, 1))  # (1, M, D)
        self.init_memory_fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(0.1)
        )

    def forward(self, input_states):
        """ initialize the model with the first input states
            input_states: (N, L, D)
            :returns  (N, M, D)
        """
        pooled_input_states = torch.sum(input_states, dim=1)   # (N, D)
        pooled_input_states = pooled_input_states.unsqueeze(1).repeat(1, self.n_memory_cells, 1)  # (N, M, D)
        pooled_input_states = pooled_input_states + self.init_memory_bias  # (N, M, D)
        init_memory = self.init_memory_fc(pooled_input_states)  # (N, M, D)
        return init_memory





class Memory_attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0.1, proj_drop=0.1, compress_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim  ** -0.5

        # 线性变换
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        # 记忆压缩模块
        self.memory_compressor = nn.Sequential(
            nn.Linear(dim, dim // compress_ratio),
            nn.GELU(),
            nn.Linear(dim // compress_ratio, dim)
        )
        self.memory_gate = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        # 注意力和投影
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1, x2, memory_k, memory_v, perfix=True):
        B, N1, C = x1.shape

        # 生成Q
        q = self.q(x1).reshape(B, N1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 动态记忆处理
        if perfix and memory_k is not None:
            # 记忆压缩 (B, M, D) -> (B, 1, D)
            mem_k = self.memory_compressor(memory_k.mean(dim=1, keepdim=True))
            mem_v = self.memory_compressor(memory_v.mean(dim=1, keepdim=True))

            # 门控加权
            gate = self.memory_gate(mem_k)  # (B,1,1)
            mem_k = mem_k * gate
            mem_v = mem_v * gate

            # 拼接独立记忆token
            k_ctx = self.k(x2)  # (B, N2, D)
            v_ctx = self.v(x2)
            k = torch.cat([k_ctx, mem_k], dim=1)  # (B, N2+1, D)
            v = torch.cat([v_ctx, mem_v], dim=1)

        else:
            k = self.k(x2)
            v = self.v(x2)


        # 生成K, V
        k = k.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B,H,N,D)
        v = v.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  #(32,8,78,64)

        # 稀疏注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B,H,N1,N2+1) (32,8,197,64)

        # Top-K稀疏化 (保留30%-50%连接)
        k = max(1, min(int(attn.size(-1) * 0.5), 64))  # 动态调整K值
        topk_val, topk_idx = torch.topk(attn, k=k, dim=-1)

        # 使用负无穷填充非Top-K位置
        sparse_attn = torch.full_like(attn, float('-inf'))
        sparse_attn.scatter_(-1, topk_idx, topk_val)

        # 归一化
        attn = F.softmax(sparse_attn, dim=-1)
        attn = self.attn_drop(attn)

        # 结果聚合
        x = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        x = self.proj(x)

        return self.proj_drop(x)



class Adaptive_Memory_Gate_Fusion(nn.Module): # sub PIA
    def __init__(self, dim=512, num_heads=8):
        super().__init__()
        self.cross_attn = Cross_Attention_without_pro(dim=dim, num_heads=num_heads)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x1, x2):
        q_m = self.cross_attn(x1, x2)
        gate = torch.sigmoid(q_m + self.linear1(x1))
        out = gate * self.linear2(x1)
        return out


class Cross_Memory_Block(nn.Module):
    """增强版跨记忆块，新增跨模态残差连接"""

    def __init__(self, dim=512, num_heads=8):
        super().__init__()
        self.cross_attn = Memory_attention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )
        self.norm2 = nn.LayerNorm(dim)
        # 跨模态跳跃连接
        self.skip_gate = nn.Sequential(
            nn.Linear(dim * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2, memory_k, memory_v, perfix=True):
        # 动态门控生成（兼容不同长度）
        gate_x1 = x1.mean(dim=1)  # (B,D)
        gate_x2 = x2.mean(dim=1)
        gate = self.skip_gate(torch.cat([gate_x1, gate_x2], dim=-1)).unsqueeze(1)  # (B,1,1)

        # 带记忆更新的注意力
        attn_out = self.cross_attn(x1, x2, memory_k, memory_v, perfix)

        # 残差连接
        out = self.norm1(x1 + gate * attn_out)
        mlp_out = self.mlp(out)
        return self.norm2(out + mlp_out)



class Memory_augmented_Interactive_Block(nn.Module): # HMPIA
    '''
    HMPIA
    '''
    def __init__(self, dim=512, num_heads=8):
        super().__init__()
        self.image_block = Cross_Memory_Block(dim, num_heads=num_heads)
        self.text_block = Cross_Memory_Block(dim, num_heads=num_heads)

    def forward(self, x1, x2, mk_i, mv_i, mk_t, mv_t, perfix=True):
        image_out = self.image_block(x1, x2, mk_i, mv_i, perfix=perfix)
        text_out = self.text_block(x2, x1, mk_t, mv_t, perfix=perfix)
        return image_out, text_out


class Ada_Bi_fusion_v2(nn.Module):  # TAGF with Cross-Gating
    def __init__(self, dim, down_dim, num_cls):
        super().__init__()
        # 前向/后向特征降维层
        self.f_dowm_linear = nn.Linear(dim*2, down_dim)
        self.b_dowm_linear = nn.Linear(dim*2, down_dim)

        # 特征增强层（添加非线性）
        self.linear_f = nn.Sequential(
            nn.Linear(2 * dim + down_dim, dim),
            nn.ReLU()
        )
        self.linear_b = nn.Sequential(
            nn.Linear(2 * dim + down_dim, dim),
            nn.ReLU()
        )

        # 交叉门控机制
        self.gate_f = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.gate_b = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

        # 最终分类层
        self.linear_o = nn.Sequential(
            nn.Linear(dim, num_cls)
        )

    def forward(self, Fi, Ft, Bi, Bt):
        # 前向路径处理
        cat_f = torch.cat([Fi, Ft], dim=-1)
        Fo = torch.cat([cat_f, self.f_dowm_linear(cat_f)], dim=-1)
        Fo = self.linear_f(Fo)  # (B, ..., dim)

        # 后向路径处理
        cat_b = torch.cat([Bi, Bt], dim=-1)
        Bo = torch.cat([cat_b, self.b_dowm_linear(cat_b)], dim=-1)
        Bo = self.linear_b(Bo)  # (B, ..., dim)

        # 交叉门控融合
        gate_f = self.gate_f(Bo)  # 用后向特征生成前向门控
        gate_b = self.gate_b(Fo)  # 用前向特征生成后向门控
        fused = gate_f * Fo + gate_b * Bo

        # 分类输出
        return self.linear_o(fused)


class QFormer(nn.Module):
    def __init__(self, dim, num_queries=32, num_heads=8):
        super().__init__()
        self.num_queries = num_queries
        self.query = nn.Parameter(torch.randn(num_queries, dim))

        # 修改后的注意力层（增加batch_first参数）
        self.image_cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.text_cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, image_feats, text_feats):
        # 输入维度处理 (保持3维)
        # image_feats: [bs, 1, dim]
        # text_feats: [bs, 1, dim]

        batch_size = image_feats.size(0)
        queries = self.query.unsqueeze(0).expand(batch_size, -1, -1)  # [bs, num_queries, dim]

        # 图像交叉注意力（使用3维输入）
        image_attn, _ = self.image_cross_attn(
            query=queries,
            key=image_feats,
            value=image_feats
        )

        # 文本交叉注意力
        text_attn, _ = self.text_cross_attn(
            query=queries,
            key=text_feats,
            value=text_feats
        )

        # 合并特征
        combined = image_attn + text_attn

        # 自注意力
        self_attn, _ = self.self_attn(
            query=combined,
            key=combined,
            value=combined
        )

        # 残差连接
        output = self.norm1(queries + self_attn)
        output = self.norm2(output + combined)

        return output  # [bs, num_queries, dim]


class ContrastiveHead(nn.Module):
    def __init__(self, dim, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.image_proj = nn.Linear(dim, dim)
        self.text_proj = nn.Linear(dim, dim)

    def forward(self, image_feats, text_feats):
        # Project features
        image_emb = F.normalize(self.image_proj(image_feats), dim=-1)
        text_emb = F.normalize(self.text_proj(text_feats), dim=-1)

        # Compute similarity matrix
        logits = torch.matmul(image_emb, text_emb.T) / self.temperature

        # Create labels
        batch_size = image_emb.size(0)
        labels = torch.arange(batch_size, device=image_emb.device)

        # Compute loss
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        return (loss_i + loss_t) / 2


def load_glove(glove_path):
    """加载GloVe词向量"""
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0].lower()  # 统一转小写
            vector = np.array(values[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings

def build_label_embeddings(label_words, glove_embeddings, embed_dim=300):
    if not label_words:
        raise ValueError("label_words列表不能为空")
    
    matrix = []
    for word in label_words:
        lword = word.lower()
        vec = glove_embeddings.get(lword, None)
        
        if vec is None:
            vec = np.zeros(embed_dim, dtype=np.float32)
        elif len(vec) != embed_dim:
            raise ValueError(
                f"GloVe向量'{lword}'维度错误: expected {embed_dim}, got {len(vec)}"
            )
        matrix.append(vec)
    
    return torch.tensor(np.array(matrix), dtype=torch.float32)

if __name__ == '__main__':
    x1 = torch.randn(16, 77, 512)
    x2 = torch.randn(16, 197, 512)
    memory = torch.randn(16, 50, 512)

