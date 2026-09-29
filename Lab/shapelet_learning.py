"""
Shapelet Learning Modules
包含shapelet学习所需的基础模块和学习模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.Time2HGModel import InceptionModule


# ==================== Shapelet Embedding Layer ====================

class ShapeletEmbedLayer(nn.Module):
    """
    原项目的shapelet嵌入层，使用滑动窗口方式获取shapelet
    特点：stride < shape_size，窗口有重叠
    """
    def __init__(self, seq_len, shape_size=8, in_chans=1, embed_dim=128, stride=4):
        super().__init__()
        self.stride = stride
        num_patches = int((seq_len - shape_size) / stride + 1)
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=shape_size, stride=stride)
    
    def forward(self, x):
        """
        输入: x, shape=(batch_size, in_chans, seq_len)
        输出: x_out, shape=(batch_size, num_patches, embed_dim)
        """
        x_out = self.proj(x).flatten(2).transpose(1, 2)
        return x_out


# ==================== 归一化模块 ====================

class RMSNorm(nn.Module):
    """RMS Normalization"""
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.gamma * self.scale


# ==================== MoE 相关模块 ====================

class MLP(nn.Module):
    """MoE 中的专家网络"""
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.gelu = nn.GELU()
        self.out_drop = nn.Dropout(p=0.15)

    def forward(self, x):
        out = self.fc1(x)
        out = self.gelu(out)
        out = self.out_drop(out)
        out = self.fc2(out)
        return out


class SparseDispatcher(object):
    """MoE 分发器"""
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())

        return combined

    def expert_to_gates(self):
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MoE_Block(nn.Module):
    """混合专家模块"""
    def __init__(self, input_size, output_size, num_experts, hidden_size, k=1):
        super(MoE_Block, self).__init__()
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k

        self.experts = nn.ModuleList([MLP(self.input_size, self.output_size, self.hidden_size) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.empty(input_size, num_experts))
        nn.init.normal_(self.w_gate, std=0.02)
        self.rmsnorm = RMSNorm(dim=self.output_size)
        self.act = nn.GELU()

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert (self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def top_k_gating(self, x):
        logits = x @ self.w_gate
        logits = self.softmax(logits)
        
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        if self.k == 1:
            top_k_gates = top_k_logits / top_k_logits.detach().clamp_min(1e-6)
        else:
            top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)  # normalization

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        load = self._gates_to_load(gates)
      
        return gates, load, logits

    def forward(self, x):
        batch_size, num_patches, feature_size = x.shape  # x: (batch_size, num_patches, input_size)
        x_flat = x.reshape(batch_size * num_patches, feature_size)  # Flatten patches for processing
        gates, load, probabilities = self.top_k_gating(x_flat)

        # calculate importance loss
        importance = probabilities.mean(0)
        loss = self.num_experts * (importance * (load.detach() / x_flat.size(0))).sum()

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x_flat)
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]

        y = x_flat + dispatcher.combine(expert_outputs)
        y = self.rmsnorm(y.view((batch_size, num_patches, feature_size)))
        
        return self.act(y), loss


# ==================== 辅助函数 ====================

def coml_index(input_indx, dim):
    """计算补集索引"""
    full_idx = torch.arange(dim)
    mask = torch.ones(input_indx.size(0), dim, dtype=torch.bool)

    for i in range(input_indx.size(0)):
        mask[i, input_indx[i]] = False

    complement_idx = torch.stack([full_idx[mask[i]] for i in range(input_indx.size(0))])
    return complement_idx


# ==================== Shapelet 学习层 ====================

class ShapeletLearningLayer(nn.Module):
    """
    Shapelet学习层 - 简化版 Time2HGNet_layer
    保留：MoE、稀疏化、残差连接、Attention
    去除：Inception
    """
    def __init__(self, dim, moe_nets=None, atten_head=None):
        super().__init__()
        
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.attention_head = atten_head
        self.moe = moe_nets
        self.act = nn.GELU()
        self.inception = InceptionModule(dim, 32)
    
    def forward(self, x, remain_ratio=1.0):
        """
        参数:
            x: (batch_size, num_patches, dim)
            remain_ratio: 保留的 token 比例（1.0 表示不稀疏化）
        返回:
            x: (batch_size, num_patches_new, dim)
            moe_loss: MoE 负载均衡损失
            index_map: (batch_size, num_patches_in) - 原始patch索引到当前输出序列索引的映射
        """
        batch_size, num_patches_in, dim = x.shape

        if remain_ratio < 1.0:
            # 稀疏化模式：Top-k 选择 + token 聚合
            x = self.norm1(x)
            attn_x_score = self.attention_head(x)
            
            left_patch_tokens = math.ceil(remain_ratio * x.shape[1])
            _, left_idx = torch.topk(attn_x_score, left_patch_tokens, dim=1, largest=True, sorted=True)
            
            # 获取补集索引
            compl_left_indx = coml_index(input_indx=left_idx.squeeze(-1), dim=x.shape[1])
            compl_left_indx = compl_left_indx.unsqueeze(2)
            
            # 选择 top-k tokens
            sorted_left_idx, _ = torch.sort(left_idx, dim=1)
            left_index = sorted_left_idx.expand(-1, -1, x.shape[2])
            compl = compl_left_indx.to(left_index.device)

            # 构建原始索引到当前输出序列索引的映射
            # 对于被保留的 token，按照排序后的位置 0..left_patch_tokens-1 映射；
            # 对于被聚合的 token，统一映射到聚合 token 的位置 left_patch_tokens。
            orig_to_new = torch.full(
                (batch_size, num_patches_in),
                fill_value=left_patch_tokens,
                device=x.device,
                dtype=torch.long,
            )
            sorted_left_idx_flat = sorted_left_idx.squeeze(-1)  # (B, left_patch_tokens)
            new_indices = torch.arange(left_patch_tokens, device=x.device, dtype=torch.long)
            new_indices = new_indices.unsqueeze(0).expand(batch_size, -1)
            orig_to_new.scatter_(1, sorted_left_idx_flat, new_indices)
            
            # 聚合非 top-k tokens
            non_topk = torch.gather(x * attn_x_score, dim=1, index=compl.expand(-1, -1, x.shape[2]))
            extra_token = torch.sum(non_topk, dim=1, keepdim=True)  # [B, 1, C]
            left_x = torch.gather(x * attn_x_score, dim=1, index=left_index)  # [B, left_tokens, C]
            
            # 拼接保留的 tokens 和聚合 token
            x = torch.cat([left_x, extra_token], dim=1)
            incep_x = self.inception(self.norm2(x).permute(0, 2, 1))
            reshape_incep_x = incep_x.permute(0, 2, 1)

            # MoE 学习
            temp_moe_x, moe_loss = self.moe(self.norm2(x))

            x = x + temp_moe_x + reshape_incep_x
        else:
            # 非稀疏化模式（warm-up 或 remain_ratio=1.0）：Attention 加权 + Inception + MoE
            # MoE 在此阶段同样参与前向传播，保证其参数从训练初期就获得梯度
            x = self.norm1(x)
            attn_x_score = self.attention_head(x)
            x = x * attn_x_score
            incep_x = self.inception(self.norm2(x).permute(0, 2, 1))
            reshape_incep_x = incep_x.permute(0, 2, 1)
            temp_moe_x, moe_loss = self.moe(self.norm2(x))
            x = x + temp_moe_x + reshape_incep_x
            # 不进行稀疏化时，索引保持不变
            orig_to_new = torch.arange(num_patches_in, device=x.device, dtype=torch.long)
            orig_to_new = orig_to_new.unsqueeze(0).expand(batch_size, -1)
        
        return self.act(x), moe_loss, orig_to_new


