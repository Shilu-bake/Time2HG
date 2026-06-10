"""
HGNNP 超图卷积模块

功能：
1. 将二部图矩阵转换为 dhg.Hypergraph 对象
2. 使用 HGNNP 模型对 shapelet 节点特征进行卷积
3. 支持批量处理

版本历史：
- v1: 逐样本串行处理（原实现）
- v2: 批量化并行处理（优化版本，默认使用）
"""

import torch
import torch.nn as nn
import sys
import os
from typing import List, Optional

# 添加 dhg 库路径
current_dir = os.path.dirname(__file__)
dhg_path = os.path.join(current_dir, '..', 'DeepHypergraph-main')
if dhg_path not in sys.path:
    sys.path.insert(0, dhg_path)

import dhg
from dhg.models.hypergraphs import HGNNP

# 导入批量化实现
try:
    from .BatchHGNNPWrapper import BatchHGNNPWrapper
    BATCH_IMPL_AVAILABLE = True
except ImportError:
    BATCH_IMPL_AVAILABLE = False
    print("Warning: BatchHGNNPWrapper not available, falling back to legacy implementation")


def bipartite_matrix_to_hypergraph(bipartite_matrix: torch.Tensor, device: torch.device) -> dhg.Hypergraph:
    """
    从二部图矩阵构建超图
    
    在这个映射中：
    - shapelet 作为超图的节点（vertices）
    - segment 作为超图的超边（hyperedges）
    - 二部图矩阵中的连接关系定义了哪些 shapelet 属于哪个 segment
    
    参数:
        bipartite_matrix: 二部图邻接矩阵, shape=(num_segments, num_patches)
                         矩阵中的值为1表示该 segment 与该 shapelet 有连接
        device: torch 设备
    
    返回:
        dhg.Hypergraph: 构建的超图对象
            - 节点数 = num_patches (shapelet 数量)
            - 超边数 = num_segments (segment 数量)
    """
    # 获取维度信息
    num_segments, num_patches = bipartite_matrix.shape
    
    # 构建超边列表
    # 每个 segment 对应一个超边，包含与其连接的所有 shapelet 节点
    e_list = []
    
    for seg_idx in range(num_segments):
        # 找到与当前 segment 连接的所有 shapelet 索引
        connected_shapelets = torch.nonzero(bipartite_matrix[seg_idx], as_tuple=True)[0]
        
        # 只有当至少有一个 shapelet 连接时才添加超边
        if len(connected_shapelets) > 0:
            # 转换为 Python list
            shapelet_list = connected_shapelets.cpu().tolist()
            e_list.append(shapelet_list)
    
    # 构建超图
    # num_v: 节点数（shapelet 数量）
    # e_list: 超边列表，每个超边是一个包含节点索引的列表
    hg = dhg.Hypergraph(num_v=num_patches, e_list=e_list, device=device)
    
    return hg


class HGNNPWrapper(nn.Module):
    """
    HGNNP 批量处理包装器（优化版本）
    
    该类封装了 HGNNP 模型，使其能够处理批量的时序样本数据。
    
    实现版本：
    - use_batch_impl=True (默认): 使用批量化实现（完全并行，5-10x 加速）
    - use_batch_impl=False: 使用传统实现（逐样本循环，兼容性保证）
    
    参数:
        in_channels: 输入特征维度（embed_dim）
        hid_channels: 隐藏层维度
        out_channels: 输出特征维度（通常等于 in_channels）
        use_bn: 是否使用 Batch Normalization，默认 False
        drop_rate: Dropout 比率，默认 0.5
        use_residual: 是否使用残差连接，默认 True
        use_batch_impl: 是否使用批量化实现，默认 True
    """
    
    def __init__(
        self, 
        in_channels: int, 
        hid_channels: int, 
        out_channels: int, 
        use_bn: bool = False, 
        drop_rate: float = 0.5,
        use_residual: bool = True,
        use_batch_impl: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.use_residual = use_residual
        self.use_batch_impl = use_batch_impl and BATCH_IMPL_AVAILABLE
        
        # 根据配置选择实现
        if self.use_batch_impl:
            # 使用批量化实现（推荐）
            self.batch_model = BatchHGNNPWrapper(
                in_channels=in_channels,
                hid_channels=hid_channels,
                out_channels=out_channels,
                use_bn=use_bn,
                drop_rate=drop_rate,
                use_residual=use_residual
            )
            self.legacy_model = None
        else:
            # 使用传统逐样本实现（兼容性）
            self.legacy_model = HGNNP(
                in_channels=in_channels,
                hid_channels=hid_channels,
                num_classes=out_channels,
                use_bn=use_bn,
                drop_rate=drop_rate
            )
            self.batch_model = None
            
            # 残差连接投影层（仅传统实现需要）
            if use_residual and in_channels != out_channels:
                self.residual_proj = nn.Linear(in_channels, out_channels)
            else:
                self.residual_proj = None
    
    def forward(
        self, 
        bipartite_matrices: torch.Tensor, 
        shapelet_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        批量前向传播
        
        参数:
            bipartite_matrices: 二部图矩阵批次
                shape=(batch_size, num_segments, num_patches)
                每个矩阵定义了 segment-shapelet 的连接关系
            shapelet_embeddings: shapelet 特征向量批次
                shape=(batch_size, num_patches, embed_dim)
                每个 shapelet 的特征表示
        
        返回:
            output_embeddings: 卷积后的 shapelet 特征向量
                shape=(batch_size, num_patches, out_channels)
        """
        if self.use_batch_impl:
            # 使用批量化实现（推荐，5-10x 加速）
            return self.batch_model(bipartite_matrices, shapelet_embeddings)
        else:
            # 使用传统逐样本实现（兼容性保证）
            return self._forward_legacy(bipartite_matrices, shapelet_embeddings)
    
    def _forward_legacy(
        self,
        bipartite_matrices: torch.Tensor,
        shapelet_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        传统逐样本前向传播（保留用于兼容性测试）
        
        警告：这个实现效率较低，仅用于验证批量化实现的正确性。
        生产环境建议使用 use_batch_impl=True。
        """
        batch_size = bipartite_matrices.shape[0]
        num_patches = shapelet_embeddings.shape[1]
        device = shapelet_embeddings.device
        
        # 存储每个样本的输出
        output_list = []
        
        # 逐样本处理
        for i in range(batch_size):
            # 1. 提取当前样本的二部图矩阵和节点特征
            bipartite_matrix_i = bipartite_matrices[i]  # (num_segments, num_patches)
            node_features_i = shapelet_embeddings[i]    # (num_patches, embed_dim)
            
            # 2. 从二部图矩阵构建超图
            hg_i = bipartite_matrix_to_hypergraph(bipartite_matrix_i, device)
            
            # 3. 执行 HGNNP 卷积
            # 注意：HGNNP 期望输入 X 的形状为 (num_vertices, in_channels)
            output_i = self.legacy_model(node_features_i, hg_i)  # (num_patches, out_channels)

            # 4. 应用残差连接（如果启用）
            if self.use_residual:
                if self.residual_proj is not None:
                    # 维度不匹配，使用投影
                    residual = self.residual_proj(node_features_i)
                else:
                    # 维度匹配，直接相加
                    residual = node_features_i
                output_i = output_i + residual
            
            # 5. 收集输出
            output_list.append(output_i)
        
        # 5. 拼接所有样本的输出
        output_embeddings = torch.stack(output_list, dim=0)  # (batch_size, num_patches, out_channels)
        
        return output_embeddings
    
    def __repr__(self) -> str:
        impl_type = "batch" if self.use_batch_impl else "legacy"
        return (f"HGNNPWrapper(in_channels={self.in_channels}, "
                f"hid_channels={self.hid_channels}, "
                f"out_channels={self.out_channels}, "
                f"impl={impl_type})")


