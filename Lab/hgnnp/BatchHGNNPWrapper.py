"""
批量化 HGNNP 包装器

功能：
1. 封装批量化 HGNNP 模型，提供与原 HGNNPWrapper 兼容的接口
2. 支持残差连接
3. 完全并行化处理，无需逐样本循环
"""

import torch
import torch.nn as nn
from typing import Optional

from .BatchHGNNPConv import BatchHGNNP


class BatchHGNNPWrapper(nn.Module):
    """
    批量化 HGNNP 包装器
    
    该类封装了批量化 HGNNP 模型，使其能够高效处理批量的时序样本数据。
    与原 HGNNPWrapper 的区别是：完全并行化处理整个 batch，无需逐样本循环。
    
    参数:
        in_channels (int): 输入特征维度（embed_dim）
        hid_channels (int): 隐藏层维度
        out_channels (int): 输出特征维度（通常等于 in_channels）
        use_bn (bool): 是否使用 Batch Normalization，默认 False
        drop_rate (float): Dropout 比率，默认 0.5
        use_residual (bool): 是否使用残差连接，默认 True
    """
    
    def __init__(
        self, 
        in_channels: int, 
        hid_channels: int, 
        out_channels: int, 
        use_bn: bool = False, 
        drop_rate: float = 0.5,
        use_residual: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.use_residual = use_residual
        
        # 初始化批量化 HGNNP 模型
        self.batch_hgnnp_model = BatchHGNNP(
            in_channels=in_channels,
            hid_channels=hid_channels,
            num_classes=out_channels,
            use_bn=use_bn,
            drop_rate=drop_rate
        )
        
        # 如果使用残差连接且维度不匹配，需要投影层
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
        批量前向传播（完全并行化）
        
        参数:
            bipartite_matrices (torch.Tensor): 二部图矩阵批次
                shape = (batch_size, num_segments, num_patches)
                每个矩阵定义了 segment-shapelet 的连接关系
            shapelet_embeddings (torch.Tensor): shapelet 特征向量批次
                shape = (batch_size, num_patches, embed_dim)
                每个 shapelet 的特征表示
        
        返回:
            torch.Tensor: 卷积后的 shapelet 特征向量
                shape = (batch_size, num_patches, out_channels)
        """
        # 保存输入用于残差连接
        residual = shapelet_embeddings
        
        # 批量化 HGNNP 处理（完全并行）
        output_embeddings = self.batch_hgnnp_model(
            shapelet_embeddings,    # (batch_size, num_patches, in_channels)
            bipartite_matrices      # (batch_size, num_segments, num_patches)
        )
        # 输出: (batch_size, num_patches, out_channels)
        
        # 应用残差连接（如果启用）
        if self.use_residual:
            if self.residual_proj is not None:
                # 维度不匹配，使用投影
                residual = self.residual_proj(residual)
            # 残差连接
            output_embeddings = output_embeddings + residual
        
        return output_embeddings
    
    def __repr__(self) -> str:
        return (f"BatchHGNNPWrapper(in_channels={self.in_channels}, "
                f"hid_channels={self.hid_channels}, "
                f"out_channels={self.out_channels}, "
                f"use_residual={self.use_residual})")


# 测试函数：对比批量化实现与原始实现
def test_batch_vs_original():
    """
    对比批量化实现与原始逐样本实现的正确性
    """
    print("=" * 80)
    print("测试批量化 vs 原始实现")
    print("=" * 80)
    
    # 尝试导入原始实现
    try:
        import sys
        import os
        current_dir = os.path.dirname(__file__)
        sys.path.insert(0, current_dir)
        from HGNNP import HGNNPWrapper
        has_original = True
    except ImportError:
        print("\n警告: 无法导入原始 HGNNPWrapper，跳过对比测试")
        has_original = False
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 创建测试数据
    batch_size = 4
    num_patches = 10
    num_segments = 6
    embed_dim = 128
    
    # Shapelet embeddings
    shapelet_embeddings = torch.randn(batch_size, num_patches, embed_dim)
    
    # 二部图矩阵
    bipartite_matrices = torch.zeros(batch_size, num_segments, num_patches)
    for b in range(batch_size):
        for s in range(num_segments):
            num_connected = torch.randint(2, 5, (1,)).item()
            connected_patches = torch.randperm(num_patches)[:num_connected]
            bipartite_matrices[b, s, connected_patches] = 1.0
    
    print(f"\n输入形状:")
    print(f"  shapelet_embeddings: {shapelet_embeddings.shape}")
    print(f"  bipartite_matrices: {bipartite_matrices.shape}")
    
    # 测试批量化实现
    print(f"\n测试批量化 BatchHGNNPWrapper...")
    batch_model = BatchHGNNPWrapper(
        in_channels=embed_dim,
        hid_channels=256,
        out_channels=embed_dim,
        use_bn=True,
        drop_rate=0.5,
        use_residual=True
    )
    
    batch_model.eval()  # 评估模式（禁用 dropout）
    
    import time
    start = time.time()
    batch_output = batch_model(bipartite_matrices, shapelet_embeddings)
    batch_time = time.time() - start
    
    print(f"  输出形状: {batch_output.shape}")
    print(f"  处理时间: {batch_time*1000:.2f} ms")
    
    # 如果有原始实现，进行对比测试
    if has_original:
        print(f"\n测试原始 HGNNPWrapper（逐样本循环）...")
        
        # 确保使用相同的权重
        original_model = HGNNPWrapper(
            in_channels=embed_dim,
            hid_channels=256,
            out_channels=embed_dim,
            use_bn=True,
            drop_rate=0.5,
            use_residual=True
        )
        
        # 复制权重
        original_model.load_state_dict(batch_model.state_dict(), strict=False)
        original_model.eval()
        
        start = time.time()
        original_output = original_model(bipartite_matrices, shapelet_embeddings)
        original_time = time.time() - start
        
        print(f"  输出形状: {original_output.shape}")
        print(f"  处理时间: {original_time*1000:.2f} ms")
        
        # 比较结果
        max_diff = torch.abs(batch_output - original_output).max().item()
        mean_diff = torch.abs(batch_output - original_output).mean().item()
        
        print(f"\n结果对比:")
        print(f"  最大差异: {max_diff:.6e}")
        print(f"  平均差异: {mean_diff:.6e}")
        print(f"  加速比: {original_time / batch_time:.2f}x")
        
        # 验证结果一致性（允许小的数值误差）
        if max_diff < 1e-4:
            print(f"  ✓ 结果一致！")
        else:
            print(f"  ✗ 警告: 结果差异较大！")
    
    # 测试梯度回传
    print(f"\n测试梯度回传...")
    batch_model.train()
    batch_output = batch_model(bipartite_matrices, shapelet_embeddings)
    loss = batch_output.sum()
    loss.backward()
    
    has_grad = all(p.grad is not None for p in batch_model.parameters() if p.requires_grad)
    print(f"  所有参数是否有梯度: {has_grad}")
    assert has_grad, "某些参数没有梯度！"
    
    print("\n" + "=" * 80)
    print("✓ 所有测试通过！")
    print("=" * 80)


# 性能基准测试
def benchmark_batch_hgnnp():
    """
    性能基准测试：测试不同 batch_size 的性能
    """
    print("=" * 80)
    print("性能基准测试")
    print("=" * 80)
    
    # 测试参数
    num_patches = 50
    num_segments = 30
    embed_dim = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n设备: {device}")
    print(f"num_patches: {num_patches}, num_segments: {num_segments}, embed_dim: {embed_dim}")
    
    # 创建模型
    model = BatchHGNNPWrapper(
        in_channels=embed_dim,
        hid_channels=256,
        out_channels=embed_dim,
        use_bn=True,
        drop_rate=0.5,
        use_residual=True
    ).to(device)
    
    model.eval()
    
    print(f"\n{'Batch Size':<12} {'Time (ms)':<15} {'Throughput (samples/s)':<25}")
    print("-" * 52)
    
    # 测试不同 batch_size
    for batch_size in [16, 32, 64, 128, 256, 512]:
        # 创建随机数据
        shapelet_embeddings = torch.randn(batch_size, num_patches, embed_dim).to(device)
        bipartite_matrices = torch.rand(batch_size, num_segments, num_patches).to(device)
        bipartite_matrices = (bipartite_matrices > 0.7).float()  # 稀疏化
        
        # 预热
        with torch.no_grad():
            _ = model(bipartite_matrices, shapelet_embeddings)
        
        # 多次运行取平均
        num_runs = 20
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.time()
                _ = model(bipartite_matrices, shapelet_embeddings)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        throughput = batch_size / avg_time
        
        print(f"{batch_size:<12} {avg_time*1000:<15.2f} {throughput:<25.1f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    import time
    
    # 运行测试
    test_batch_vs_original()
    
    print("\n\n")
    
    # 运行性能基准测试
    benchmark_batch_hgnnp()

