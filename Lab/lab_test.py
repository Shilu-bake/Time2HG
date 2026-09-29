import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.layers import trunc_normal_

# Import Shapelet learning modules
from Lab.shapelet_learning import (
    ShapeletEmbedLayer,
    MoE_Block,
    ShapeletLearningLayer,
    RMSNorm,
)

from models.Time2HGModel import InceptionModule


class SegmentEmbedLayer(nn.Module):
    """
    将时间序列切分成等长的segment，并使用1-D CNN进行嵌入
    与原本的ShapeEmbedLayer不同之处：
    - ShapeEmbedLayer: 使用滑动窗口（stride < shape_size）获取shapelet
    - SegmentEmbedLayer: 将时间序列切分成不重叠的segment（stride = segment_size）
    """
    def __init__(self, seq_len, segment_size=8, in_chans=1, embed_dim=128):
        super().__init__()
        # segment方式：stride等于segment_size，不重叠切分
        self.segment_size = segment_size
        self.stride = segment_size  # 关键：stride = segment_size 实现等长切分
        
        # 计算能切分出多少个segment
        num_segments = seq_len // segment_size
        self.num_segments = num_segments
        
        # 使用1-D CNN进行嵌入，与原本的ShapeEmbedLayer相同的方式
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=segment_size, stride=self.stride)
    
    def forward(self, x):
        """
        输入: x, shape=(batch_size, in_chans, seq_len)
        输出: x_out, shape=(batch_size, num_segments, embed_dim)
        """
        # 使用Conv1d进行嵌入
        x_out = self.proj(x)  # shape: (batch_size, embed_dim, num_segments)
        x_out = x_out.flatten(2).transpose(1, 2)  # shape: (batch_size, num_segments, embed_dim)
        return x_out


class SegmentBasedModel(nn.Module):
    """
    基于segment的嵌入模型，与原本Time2HGNet的嵌入部分对齐
    只包含 segment 嵌入 + 位置编码，返回向量化结果
    """
    def __init__(self, seq_len, segment_size=8, in_chans=1, embed_dim=128):
        super().__init__()
        
        # 使用segment方式嵌入
        self.segment_embed = SegmentEmbedLayer(
            seq_len=seq_len,
            segment_size=segment_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        
        # 位置编码（与原项目 Time2HGNet 保持一致）
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.segment_embed.num_segments, embed_dim),
            requires_grad=True
        )
        
    def forward(self, x):
        """
        输入: x, shape=(batch_size, in_chans, seq_len)
        输出: x, shape=(batch_size, num_segments, embed_dim)
        返回嵌入后的segment向量，可以用于后续的分类、聚类等任务
        """
        # Segment嵌入（1-D CNN）
        x = self.segment_embed(x)  # (batch_size, num_segments, embed_dim)
        
        # 添加位置编码
        x = x + self.pos_embed  # (batch_size, num_segments, embed_dim)
        
        return x


class ShapeletBasedModel(nn.Module):
    """
    基于shapelet的学习模型，包含 Time2HGNet 核心学习机制
    
    核心特性：
    - Shapelet 嵌入（滑动窗口）
    - MoE 学习（类别判别性特征）
    - Shapelet 稀疏化（Top-k 选择 + token 聚合）
    - 残差连接（梯度流动优化）
    - Warm-up 渐进式训练
    """
    def __init__(self, seq_len, shape_size=8, in_chans=1, embed_dim=128, 
                 stride=4, depth=2, num_experts=8, sparse_rate=0.5, num_classes=None,use_head = False,
                 moe_top_k=1):
        """
        参数:
            seq_len: 时间序列长度
            shape_size: Shapelet 大小
            in_chans: 输入通道数
            embed_dim: 嵌入维度
            stride: 滑动窗口步长
            depth: 学习层深度
            num_experts: MoE 专家数量（通常等于类别数）
            moe_top_k: 每个 token 激活的 MoE 专家数量
            sparse_rate: 最大稀疏率（最后一层的稀疏程度）
            num_classes: 类别数（如果提供，则添加分类头；否则仅做特征提取）
        """
        super().__init__()
        
        self.seq_len = seq_len
        self.shape_size = shape_size
        self.embed_dim = embed_dim
        self.sparse_rate = sparse_rate
        self.depth = depth
        self.num_classes = num_classes
        self.use_head = use_head
        self.repr_classifier = nn.Linear(embed_dim, num_classes)

        # Shapelet 嵌入层（滑动窗口）
        self.shapelet_embed = ShapeletEmbedLayer(
            seq_len=seq_len,
            shape_size=shape_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            stride=stride
        )
        
        # 位置编码
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.shapelet_embed.num_patches, embed_dim),
            requires_grad=True
        )
        self.pos_drop = nn.Dropout(p=0.15)
        
        # 渐进稀疏率（从0到sparse_rate线性增长）
        self.sparse_ratio_d = [x.item() for x in torch.linspace(0, sparse_rate, depth)]

        # 独立的输出聚合注意力头（用于 get_representation_with_attention / classify_with_attention）
        self.output_attention_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 16),
            nn.Tanh(),
            nn.Linear(embed_dim // 16, 1),
            nn.Sigmoid(),
        )

        # 每层独立的注意力头和 MoE 模块，避免参数共享导致多层退化为单层
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 16),
                nn.Tanh(),
                nn.Linear(embed_dim // 16, 1),
                nn.Sigmoid(),
            )
            for _ in range(depth)
        ])

        self.moes = nn.ModuleList([
            MoE_Block(
                input_size=embed_dim,
                output_size=embed_dim,
                num_experts=num_experts,
                hidden_size=embed_dim,
                k=moe_top_k
            )
            for _ in range(depth)
        ])

        # 多层 Shapelet 学习块（每层绑定各自独立的 MoE 和 Attention Head）
        self.learning_layers = nn.ModuleList([
            ShapeletLearningLayer(
                dim=embed_dim,
                moe_nets=self.moes[i],
                atten_head=self.attention_heads[i]
            )
            for i in range(depth)
        ])

        # 权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """权重初始化（与 Time2HGNet 保持一致）"""
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                trunc_normal_(m.weight, std=.02)
            if m.bias is not None: 
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            with torch.no_grad():
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
    
    def classify_with_attention(self, x):
        """
        使用 attention 加权的方式对 shapelet 特征进行分类
        
        参数:
            x: (batch_size, num_patches, embed_dim) - 学习后的 shapelet 特征
        
        返回:
            cls_logits: (batch_size, num_classes) - 分类 logits
        """
        # 计算最终的 attention 权重（使用独立的输出聚合注意力头）
        end_attn_score = self.output_attention_head(x)  # (batch_size, num_patches, 1)
        
        # 每个 shapelet 输出分类 logits
        instance_logits = self.repr_classifier(x)  # (batch_size, num_patches, num_classes)
        
        # Attention 加权
        weighted_instance_logits = instance_logits * end_attn_score

        # 聚合所有 shapelet 的预测
        cls_logits = torch.mean(weighted_instance_logits, dim=1)  # (batch_size, num_classes)
        
        return cls_logits

    def get_representation_with_attention(self, x):
        """
        使用注意力加权池化获取时序表示向量
        
        参数:
            x: (batch_size, num_patches, embed_dim) - shapelet 特征
        
        返回:
            ts_representation: (batch_size, embed_dim) - 时序表示向量
        """
        # 1. 计算注意力权重（使用独立的输出聚合注意力头）
        attn_weights = self.output_attention_head(x)  # (batch_size, num_patches, 1)
        
        # 2. Softmax 归一化（确保权重和为1）
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # 3. 加权聚合
        ts_representation = torch.sum(x * attn_weights, dim=1)  # (batch_size, embed_dim)
        
        return ts_representation

    def forward(self, x, num_epoch_i=100, warm_up_epoch=50, patch_keep_mask=None):
        """
        前向传播
        
        参数:
            x: (batch_size, in_chans, seq_len) - 输入时间序列
            num_epoch_i: 当前训练轮数
            warm_up_epoch: Warm-up 轮数（在此之前不进行稀疏化）
            patch_keep_mask: (batch_size, num_patches) - 可选的 shapelet patch 保留掩码
                           1 表示保留，0 表示丢失（置零）
        
        返回:
            如果有分类头 (use_head=True):
                cls_logits: (batch_size, num_classes) - 分类 logits
                moe_loss: MoE 负载均衡损失
            否则:
                x: (batch_size, num_patches_final, embed_dim) - MoE 学习后的 shapelet 特征
                index_map: (batch_size, num_patches_initial) - 原始patch索引到最终输出序列索引的映射
                moe_loss: MoE 负载均衡损失
        """
        # Shapelet 嵌入
        x = self.shapelet_embed(x)  # (batch_size, num_patches, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        if patch_keep_mask is not None:
            x = x * patch_keep_mask.unsqueeze(-1).to(dtype=x.dtype)

        # 记录原始 patch 数量，并初始化“原始索引 -> 当前索引”的总映射
        batch_size, num_patches_initial, _ = x.shape
        index_map_total = torch.arange(num_patches_initial, device=x.device, dtype=torch.long)
        index_map_total = index_map_total.unsqueeze(0).expand(batch_size, -1)

        # 累积 MoE 损失
        moe_loss = None
        
        # 通过多层学习块，同时更新“原始索引 -> 当前索引”的总映射
        for d, layer in enumerate(self.learning_layers):
            # 计算当前层的保留比例
            depth_remain_ratio = 1.0 - self.sparse_ratio_d[d]
            
            # Warm-up 期间不进行稀疏化
            if num_epoch_i < warm_up_epoch:
                depth_remain_ratio = 1.0
            
            # 通过学习层，获得当前层的局部索引映射
            x, _temp_mloss, index_map_layer = layer(x, remain_ratio=depth_remain_ratio)
            # 组合映射：原始索引 -> 进入当前层的索引 -> 当前层输出索引
            index_map_total = torch.gather(index_map_layer, 1, index_map_total)
            
            # 累积 MoE 损失
            if moe_loss is None:
                moe_loss = _temp_mloss
            else:
                moe_loss = moe_loss + _temp_mloss
        
        # 如果有分类头，进行 attention 加权分类（与 Time2HGNet 一致）
        if self.use_head:
            cls_logits = self.classify_with_attention(x)
            return cls_logits, moe_loss

        # 返回：学习后特征、原始索引到最终输出索引的总映射，以及 MoE 损失
        return x, index_map_total, moe_loss


class SegmentLearningLayer(nn.Module):
    """
    Segment学习层 - 无稀疏化版本
    保留：MoE、Attention加权、Inception、残差连接
    移除：Top-k稀疏化、token聚合
    """
    def __init__(self, dim, moe_nets=None, atten_head=None):
        super().__init__()
        
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.attention_head = atten_head
        self.moe = moe_nets
        self.act = nn.GELU()
        self.inception = InceptionModule(dim, 32)
    
    def forward(self, x):
        """
        参数:
            x: (batch_size, num_patches, dim)
        返回:
            x: (batch_size, num_patches, dim) - 保持维度不变
            moe_loss: MoE 负载均衡损失
        """
        # Attention 加权（不用于筛选）
        x = self.norm1(x)
        attn_x_score = self.attention_head(x)
        x = x * attn_x_score
        
        # Inception 特征提取
        incep_x = self.inception(self.norm2(x).permute(0, 2, 1))
        reshape_incep_x = incep_x.permute(0, 2, 1)
        
        # MoE 学习
        temp_moe_x, moe_loss = self.moe(self.norm2(x))
        
        # 残差连接
        x = x + temp_moe_x + reshape_incep_x
        
        return self.act(x), moe_loss


class SegmentModel(nn.Module):
    """
    基于segment的学习模型（无稀疏化）
    
    核心特性：
    - Segment 嵌入（滑动窗口）
    - MoE 学习（特征增强）
    - Attention 加权（重要性评分）
    - Inception 特征提取
    - 残差连接
    - 无稀疏化（保持完整性）
    """
    def __init__(self, seq_len, shape_size=8, in_chans=1, embed_dim=128, 
                 stride=4, depth=2, num_experts=8, num_classes=None, use_head=False):
        """
        参数:
            seq_len: 时间序列长度
            shape_size: Segment 大小
            in_chans: 输入通道数
            embed_dim: 嵌入维度
            stride: 滑动窗口步长
            depth: 学习层深度
            num_experts: MoE 专家数量
            num_classes: 类别数（如果提供，则添加分类头）
            use_head: 是否使用分类头
        """
        super().__init__()
        
        self.seq_len = seq_len
        self.shape_size = shape_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_classes = num_classes
        self.use_head = use_head

        # Segment 嵌入层（滑动窗口）
        self.shapelet_embed = ShapeletEmbedLayer(
            seq_len=seq_len,
            shape_size=shape_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            stride=stride
        )
        
        # 位置编码
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.shapelet_embed.num_patches, embed_dim),
            requires_grad=True
        )
        self.pos_drop = nn.Dropout(p=0.15)
        
        # 注意力头（用于重要性评分）
        self.attention_head = nn.Sequential(
            nn.Linear(embed_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        
        # MoE 模块（特征增强学习）
        self.moe = MoE_Block(
            input_size=embed_dim, 
            output_size=embed_dim, 
            num_experts=num_experts, 
            hidden_size=embed_dim
        )
        
        # 多层 Segment 学习块（无稀疏化）
        self.learning_layers = nn.ModuleList([
            SegmentLearningLayer(
                dim=embed_dim, 
                moe_nets=self.moe, 
                atten_head=self.attention_head
            )
            for i in range(depth)
        ])
        
        # 可选的分类头
        self.head = nn.Linear(embed_dim, num_classes)

        # 权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """权重初始化"""
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                trunc_normal_(m.weight, std=.02)
            if m.bias is not None: 
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            with torch.no_grad():
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        前向传播（无稀疏化）
        
        参数:
            x: (batch_size, in_chans, seq_len) - 输入时间序列
        
        返回:
            如果 use_head=True:
                cls_logits: (batch_size, num_classes) - 分类 logits
                moe_loss: MoE 负载均衡损失
            否则:
                x: (batch_size, num_patches, embed_dim) - 学习后的特征
                moe_loss: MoE 负载均衡损失
        """
        # Segment 嵌入
        x = self.shapelet_embed(x)  # (batch_size, num_patches, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 累积 MoE 损失
        moe_loss = None
        
        # 通过多层学习块（无稀疏化）
        for layer in self.learning_layers:
            x, _temp_mloss = layer(x)
            
            # 累积 MoE 损失
            if moe_loss is None:
                moe_loss = _temp_mloss
            else:
                moe_loss = moe_loss + _temp_mloss
        
        # 如果使用分类头，进行 attention 加权分类
        if self.use_head:
            cls_logits = self.classify_with_attention(x)
            return cls_logits, moe_loss

        # 否则返回学习后的特征
        return x, moe_loss


def assign_nearest_shapelets(segment_embeddings, shapelet_embeddings, k=3):
    """
    为每个segment分配k个最近的shapelet
    
    参数:
        segment_embeddings: (batch_size, num_segments, embed_dim)
        shapelet_embeddings: (batch_size, num_patches, embed_dim)
        k: 每个segment分配的shapelet数量
    
    返回:
        indices: (batch_size, num_segments, k) - 每个segment的k个最近shapelet的索引
        distances: (batch_size, num_segments, k) - 对应的欧氏距离
    """
    # 步骤1：计算距离矩阵
    # segment_embeddings: (B, S, D)
    # shapelet_embeddings: (B, P, D)
    # 扩展维度进行广播
    seg_expanded = segment_embeddings.unsqueeze(2)  # (B, S, 1, D)
    shape_expanded = shapelet_embeddings.unsqueeze(1)  # (B, 1, P, D)
    
    # 计算欧氏距离
    distances = torch.norm(seg_expanded - shape_expanded, dim=-1)  # (B, S, P)
    
    # 步骤2：动态调整k值，不超过实际shapelet数量
    num_shapelets = distances.shape[-1]  # 实际的shapelet数量
    actual_k = min(k, num_shapelets)
    
    # 找到top-k最近的shapelet
    topk_distances, topk_indices = torch.topk(
        distances, k=actual_k, dim=-1, largest=False, sorted=True
    )
    # topk_distances: (B, S, actual_k)
    # topk_indices: (B, S, actual_k)
    
    return topk_indices, topk_distances


def assign_temporal_shapelets(segment_size, shapelet_size, stride, seq_len, 
                               temporal_window=2, device='cuda'):
    """
    基于时间邻近性为每个 segment 分配 shapelet
    
    该函数根据时间位置（而非嵌入相似度）建立 segment-shapelet 连接，
    保留时间序列的因果性和时间依赖关系。
    
    参数:
        segment_size: segment 的大小（等长切分）
        shapelet_size: shapelet 的大小（滑动窗口）
        stride: shapelet 的滑动步长
        seq_len: 时间序列长度
        temporal_window: 时间邻近窗口（连接前后 N 个 segment 范围内的 shapelet）
        device: 设备（该参数保留用于兼容性，但实际不使用）
    
    返回:
        connections: List[List[int]], 每个 segment 连接的 shapelet 索引列表
        num_segments: segment 数量
        num_patches: shapelet 数量
    
    示例:
        如果 seq_len=100, segment_size=20, shapelet_size=20, stride=5, temporal_window=1.5
        - Segment 0 覆盖 [0, 20)，中心位置 10
        - Shapelet 0 覆盖 [0, 20)，中心位置 10
        - 时间距离 = |10 - 10| / 20 = 0 <= 1.5 ✓ 连接
        - Shapelet 1 覆盖 [5, 25)，中心位置 15
        - 时间距离 = |10 - 15| / 20 = 0.25 <= 1.5 ✓ 连接
    """
    # 计算 segment 位置（无重叠切分）
    num_segments = seq_len // segment_size
    segment_positions = [(i * segment_size, (i + 1) * segment_size)
                         for i in range(num_segments)]
    
    # 计算 shapelet 位置（滑动窗口）
    num_patches = (seq_len - shapelet_size) // stride + 1
    shapelet_positions = [(i * stride, i * stride + shapelet_size) 
                          for i in range(num_patches)]
    
    # 为每个 segment 找到时间邻近的 shapelet
    connections = []
    for seg_idx, (seg_start, seg_end) in enumerate(segment_positions):
        seg_center = (seg_start + seg_end) / 2
        connected_shapelets = []
        
        for shape_idx, (shape_start, shape_end) in enumerate(shapelet_positions):
            shape_center = (shape_start + shape_end) / 2
            
            # 计算时间距离（以 segment_size 为单位）
            time_distance = abs(shape_center - seg_center) / segment_size
            
            # 只连接在时间窗口内的 shapelet
            if time_distance <= temporal_window:
                connected_shapelets.append(shape_idx)
        
        connections.append(connected_shapelets)
    
    return connections, num_segments, num_patches