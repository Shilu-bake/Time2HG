"""
批量化 HGNNP 卷积层

功能：
1. 实现批量化的 HGNNP 卷积操作
2. 支持 batch 输入，完全并行化处理
3. 与原始 dhg.HGNNPConv 接口兼容
"""

import torch
import torch.nn as nn
from typing import Optional

from .batch_hypergraph_ops import (
    batch_hypergraph_v2v,
    batch_compute_degrees,
    bipartite_matrix_to_incidence_matrix
)


class BatchHGNNPConv(nn.Module):
    r"""批量化 HGNNP 卷积层
    
    基于 HGNN+ 论文的超图卷积操作，但支持批量输入。
    
    数学公式（mean aggregation）：
        X' = σ(D_v^{-1} H W_e D_e^{-1} H^T X Θ)
    
    其中：
    - X: 节点特征矩阵 (batch, num_v, in_channels)
    - H: 超图关联矩阵 (batch, num_v, num_e)
    - D_v: 节点度矩阵（对角阵）
    - D_e: 超边度矩阵（对角阵）
    - W_e: 超边权重矩阵（本实现中为单位矩阵）
    - Θ: 可学习的线性变换参数
    - σ: 激活函数（ReLU）
    
    参数:
        in_channels (int): 输入特征维度
        out_channels (int): 输出特征维度
        bias (bool): 是否使用偏置，默认 True
        use_bn (bool): 是否使用 Batch Normalization，默认 False
        drop_rate (float): Dropout 概率，默认 0.5
        is_last (bool): 是否为最后一层（最后一层不使用激活和 dropout），默认 False
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        is_last: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_last = is_last
        
        # 线性变换层（对应公式中的 Θ）
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)
        
        # Batch Normalization（可选）
        # 注意：BatchNorm1d 期望输入 (N, C) 或 (N, C, L)
        # 对于 (batch, num_v, channels)，我们需要特殊处理
        if use_bn:
            self.bn = nn.BatchNorm1d(out_channels)
        else:
            self.bn = None
        
        # 激活函数
        self.act = nn.ReLU(inplace=True)
        
        # Dropout
        self.drop = nn.Dropout(drop_rate)
    
    def forward(
        self, 
        X: torch.Tensor,
        bipartite_matrix: torch.Tensor
    ) -> torch.Tensor:
        r"""前向传播（直接接受二部图矩阵）
        
        参数:
            X (torch.Tensor): 节点特征矩阵
                shape = (batch, num_v, in_channels)
            bipartite_matrix (torch.Tensor): 二部图邻接矩阵
                shape = (batch, num_segments, num_patches)
                表示 segment 和 shapelet 的连接关系
        
        返回:
            torch.Tensor: 输出节点特征矩阵
                shape = (batch, num_v, out_channels)
        """
        batch_size = X.size(0)
        num_v = X.size(1)
        
        # Step 1: 线性变换
        X = self.theta(X)  # (batch, num_v, out_channels)
        
        # Step 2: 转换为超图关联矩阵
        # bipartite_matrix: (batch, num_segments, num_patches)
        # H: (batch, num_patches, num_segments)
        H = bipartite_matrix_to_incidence_matrix(bipartite_matrix)
        
        # Step 3: 批量化超图消息传递（v2v）
        X = batch_hypergraph_v2v(X, H, aggr="mean")
        
        # Step 4: 激活、归一化、Dropout（如果不是最后一层）
        if not self.is_last:
            X = self.act(X)
            
            if self.bn is not None:
                # BatchNorm1d 期望输入 (N, C) 或 (N, C, L)
                # 我们需要将 (batch, num_v, channels) reshape 为 (batch*num_v, channels)
                X = X.reshape(batch_size * num_v, self.out_channels)
                X = self.bn(X)
                X = X.reshape(batch_size, num_v, self.out_channels)
            
            X = self.drop(X)
        
        return X
    
    def forward_with_incidence(
        self,
        X: torch.Tensor,
        H: torch.Tensor
    ) -> torch.Tensor:
        r"""前向传播（直接接受关联矩阵）
        
        这是一个替代接口，直接接受超图关联矩阵。
        
        参数:
            X (torch.Tensor): 节点特征矩阵
                shape = (batch, num_v, in_channels)
            H (torch.Tensor): 超图关联矩阵
                shape = (batch, num_v, num_e)
        
        返回:
            torch.Tensor: 输出节点特征矩阵
                shape = (batch, num_v, out_channels)
        """
        batch_size = X.size(0)
        num_v = X.size(1)
        
        # Step 1: 线性变换
        X = self.theta(X)  # (batch, num_v, out_channels)
        
        # Step 2: 批量化超图消息传递（v2v）
        X = batch_hypergraph_v2v(X, H, aggr="mean")
        
        # Step 3: 激活、归一化、Dropout（如果不是最后一层）
        if not self.is_last:
            X = self.act(X)
            
            if self.bn is not None:
                X = X.reshape(batch_size * num_v, self.out_channels)
                X = self.bn(X)
                X = X.reshape(batch_size, num_v, self.out_channels)
            
            X = self.drop(X)
        
        return X
    
    def __repr__(self) -> str:
        return (f"BatchHGNNPConv(in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"use_bn={self.bn is not None}, "
                f"drop_rate={self.drop.p if self.drop else 0.0}, "
                f"is_last={self.is_last})")


class BatchHGNNP(nn.Module):
    r"""批量化 HGNNP 模型（2层）
    
    完整的 HGNNP 模型，包含两层卷积。
    
    参数:
        in_channels (int): 输入特征维度
        hid_channels (int): 隐藏层维度
        num_classes (int): 输出类别数（或输出维度）
        use_bn (bool): 是否使用 Batch Normalization，默认 False
        drop_rate (float): Dropout 概率，默认 0.5
    """
    
    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.num_classes = num_classes
        
        # 第一层：in_channels -> hid_channels
        self.layer1 = BatchHGNNPConv(
            in_channels=in_channels,
            out_channels=hid_channels,
            use_bn=use_bn,
            drop_rate=drop_rate,
            is_last=False
        )
        
        # 第二层：hid_channels -> num_classes
        self.layer2 = BatchHGNNPConv(
            in_channels=hid_channels,
            out_channels=num_classes,
            use_bn=use_bn,
            drop_rate=drop_rate,
            is_last=True  # 最后一层不使用激活和 dropout
        )
    
    def forward(
        self, 
        X: torch.Tensor,
        bipartite_matrix: torch.Tensor
    ) -> torch.Tensor:
        r"""前向传播
        
        参数:
            X (torch.Tensor): 节点特征矩阵
                shape = (batch, num_v, in_channels)
            bipartite_matrix (torch.Tensor): 二部图邻接矩阵
                shape = (batch, num_segments, num_patches)
        
        返回:
            torch.Tensor: 输出特征矩阵
                shape = (batch, num_v, num_classes)
        """
        # 第一层卷积
        X = self.layer1(X, bipartite_matrix)
        
        # 第二层卷积
        #X = self.layer2(X, bipartite_matrix)
        
        return X
    
    def __repr__(self) -> str:
        return (f"BatchHGNNP(in_channels={self.in_channels}, "
                f"hid_channels={self.hid_channels}, "
                f"num_classes={self.num_classes})")


# 测试函数
def test_batch_hgnnp_conv():
    """
    测试批量化 HGNNP 卷积层的正确性
    """
    print("=" * 80)
    print("测试批量化 HGNNP 卷积层")
    print("=" * 80)
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 创建测试数据
    batch_size = 4
    num_patches = 10  # shapelet 数量（节点）
    num_segments = 6  # segment 数量（超边）
    in_channels = 128
    out_channels = 128
    
    # 节点特征
    X = torch.randn(batch_size, num_patches, in_channels)
    
    # 二部图矩阵（随机稀疏）
    bipartite_matrix = torch.zeros(batch_size, num_segments, num_patches)
    for b in range(batch_size):
        for s in range(num_segments):
            # 每个 segment 随机连接 2-4 个 shapelet
            num_connected = torch.randint(2, 5, (1,)).item()
            connected_patches = torch.randperm(num_patches)[:num_connected]
            bipartite_matrix[b, s, connected_patches] = 1.0
    
    print(f"\n输入形状:")
    print(f"  X: {X.shape}")
    print(f"  bipartite_matrix: {bipartite_matrix.shape}")
    
    # 测试单层卷积
    print(f"\n测试单层 BatchHGNNPConv...")
    conv = BatchHGNNPConv(
        in_channels=in_channels,
        out_channels=out_channels,
        use_bn=True,
        drop_rate=0.5,
        is_last=False
    )
    
    # 训练模式
    conv.train()
    X_out_train = conv(X, bipartite_matrix)
    print(f"  训练模式输出: {X_out_train.shape}")
    
    # 评估模式
    conv.eval()
    X_out_eval = conv(X, bipartite_matrix)
    print(f"  评估模式输出: {X_out_eval.shape}")
    
    # 测试完整模型
    print(f"\n测试完整 BatchHGNNP 模型...")
    model = BatchHGNNP(
        in_channels=in_channels,
        hid_channels=256,
        num_classes=out_channels,
        use_bn=True,
        drop_rate=0.5
    )
    
    model.train()
    X_out = model(X, bipartite_matrix)
    print(f"  输出形状: {X_out.shape}")
    assert X_out.shape == (batch_size, num_patches, out_channels)
    
    # 测试梯度回传
    print(f"\n测试梯度回传...")
    loss = X_out.sum()
    loss.backward()
    
    # 检查参数是否有梯度
    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"  所有参数是否有梯度: {has_grad}")
    assert has_grad, "某些参数没有梯度！"
    
    print("\n" + "=" * 80)
    print("✓ 所有测试通过！")
    print("=" * 80)


if __name__ == "__main__":
    test_batch_hgnnp_conv()

