"""
批量化超图操作

功能：
1. 批量计算超图的度矩阵
2. 批量执行超图消息传递（v2v, v2e, e2v）
3. 完全基于矩阵运算，无需构建 dhg.Hypergraph 对象
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def batch_compute_degrees(H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    批量计算超图的度矩阵
    
    参数:
        H: 关联矩阵, shape=(batch, num_v, num_e)
           H[b, i, j] = 1 表示节点 i 属于超边 j
    
    返回:
        D_v: 节点度（每个节点连接的超边数）, shape=(batch, num_v)
        D_e: 超边度（每个超边包含的节点数）, shape=(batch, num_e)
    """
    D_v = H.sum(dim=2)  # (batch, num_v) 对每个节点，统计连接的超边数
    D_e = H.sum(dim=1)  # (batch, num_e) 对每个超边，统计包含的节点数
    
    return D_v, D_e


def batch_hypergraph_v2e(
    X: torch.Tensor,
    H: torch.Tensor,
    D_e: Optional[torch.Tensor] = None,
    aggr: str = "mean",
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    批量化超图消息传递：从节点到超边（v2e）
    
    数学公式：
        X_e = H^T @ X  (sum aggregation)
        X_e = H^T @ X / D_e  (mean aggregation)
    
    参数:
        X: 节点特征矩阵, shape=(batch, num_v, in_channels)
        H: 关联矩阵, shape=(batch, num_v, num_e)
        D_e: 超边度矩阵, shape=(batch, num_e)，如果为 None 则自动计算
        aggr: 聚合方式，可选 "mean", "sum"
        epsilon: 数值稳定性常数
    
    返回:
        X_e: 超边特征矩阵, shape=(batch, num_e, in_channels)
    """
    # v2e: 节点特征聚合到超边
    # X_e[b, j, :] = sum_i H[b, i, j] * X[b, i, :]
    X_e = torch.bmm(H.transpose(1, 2), X)  # (batch, num_e, in_channels)
    
    if aggr == "mean":
        # 计算超边度（如果未提供）
        if D_e is None:
            D_e = H.sum(dim=1)  # (batch, num_e)
        
        # 除以超边度进行平均聚合
        X_e = X_e / (D_e.unsqueeze(-1) + epsilon)
    
    return X_e


def batch_hypergraph_e2v(
    X_e: torch.Tensor,
    H: torch.Tensor,
    D_v: Optional[torch.Tensor] = None,
    aggr: str = "mean",
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    批量化超图消息传递：从超边到节点（e2v）
    
    数学公式：
        X_v = H @ X_e  (sum aggregation)
        X_v = H @ X_e / D_v  (mean aggregation)
    
    参数:
        X_e: 超边特征矩阵, shape=(batch, num_e, in_channels)
        H: 关联矩阵, shape=(batch, num_v, num_e)
        D_v: 节点度矩阵, shape=(batch, num_v)，如果为 None 则自动计算
        aggr: 聚合方式，可选 "mean", "sum"
        epsilon: 数值稳定性常数
    
    返回:
        X_v: 节点特征矩阵, shape=(batch, num_v, in_channels)
    """
    # e2v: 超边特征聚合到节点
    # X_v[b, i, :] = sum_j H[b, i, j] * X_e[b, j, :]
    X_v = torch.bmm(H, X_e)  # (batch, num_v, in_channels)
    
    if aggr == "mean":
        # 计算节点度（如果未提供）
        if D_v is None:
            D_v = H.sum(dim=2)  # (batch, num_v)
        
        # 除以节点度进行平均聚合
        X_v = X_v / (D_v.unsqueeze(-1) + epsilon)
    
    return X_v


def batch_hypergraph_v2v(
    X: torch.Tensor,
    H: torch.Tensor,
    D_v: Optional[torch.Tensor] = None,
    D_e: Optional[torch.Tensor] = None,
    aggr: str = "mean",
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    批量化超图消息传递：从节点到节点（v2v）
    
    这是 v2e 和 e2v 的组合，实现超图卷积的核心操作。
    
    数学公式（mean aggregation）：
        X' = D_v^{-1} @ H @ D_e^{-1} @ H^T @ X
    
    参数:
        X: 节点特征矩阵, shape=(batch, num_v, in_channels)
        H: 关联矩阵, shape=(batch, num_v, num_e)
        D_v: 节点度矩阵, shape=(batch, num_v)，如果为 None 则自动计算
        D_e: 超边度矩阵, shape=(batch, num_e)，如果为 None 则自动计算
        aggr: 聚合方式，可选 "mean", "sum"
        epsilon: 数值稳定性常数
    
    返回:
        X_out: 输出节点特征矩阵, shape=(batch, num_v, in_channels)
    """
    # 预计算度矩阵（避免重复计算）
    if D_v is None or D_e is None:
        _D_v, _D_e = batch_compute_degrees(H)
        if D_v is None:
            D_v = _D_v
        if D_e is None:
            D_e = _D_e
    
    # Step 1: v2e（节点 -> 超边）
    X_e = batch_hypergraph_v2e(X, H, D_e, aggr=aggr, epsilon=epsilon)
    
    # Step 2: e2v（超边 -> 节点）
    X_out = batch_hypergraph_e2v(X_e, H, D_v, aggr=aggr, epsilon=epsilon)
    
    return X_out


def bipartite_matrix_to_incidence_matrix(bipartite_matrix: torch.Tensor) -> torch.Tensor:
    """
    将二部图矩阵转换为超图关联矩阵
    
    在我们的场景中：
    - 二部图矩阵: (batch, num_segments, num_patches)
      表示 segment 和 shapelet 的连接关系
    - 超图关联矩阵: (batch, num_patches, num_segments)
      表示 shapelet（节点）和 segment（超边）的关联关系
    
    参数:
        bipartite_matrix: 二部图邻接矩阵, shape=(batch, num_segments, num_patches)
    
    返回:
        H: 超图关联矩阵, shape=(batch, num_patches, num_segments)
    """
    # 简单转置即可
    H = bipartite_matrix.transpose(1, 2)
    return H


# 测试函数（可选）
def test_batch_hypergraph_ops():
    """
    测试批量化超图操作的正确性
    """
    print("=" * 80)
    print("测试批量化超图操作")
    print("=" * 80)
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 创建测试数据
    batch_size = 4
    num_v = 10  # 节点数（shapelet）
    num_e = 6   # 超边数（segment）
    in_channels = 8
    
    # 节点特征
    X = torch.randn(batch_size, num_v, in_channels)
    
    # 关联矩阵（随机稀疏）
    H = torch.zeros(batch_size, num_v, num_e)
    for b in range(batch_size):
        for e in range(num_e):
            # 每个超边随机连接 2-4 个节点
            num_connected = torch.randint(2, 5, (1,)).item()
            connected_nodes = torch.randperm(num_v)[:num_connected]
            H[b, connected_nodes, e] = 1.0
    
    print(f"\n输入形状:")
    print(f"  X: {X.shape}")
    print(f"  H: {H.shape}")
    
    # 测试度计算
    D_v, D_e = batch_compute_degrees(H)
    print(f"\n度矩阵:")
    print(f"  D_v: {D_v.shape}, 示例: {D_v[0]}")
    print(f"  D_e: {D_e.shape}, 示例: {D_e[0]}")
    
    # 测试 v2e
    X_e = batch_hypergraph_v2e(X, H, D_e, aggr="mean")
    print(f"\nv2e 输出:")
    print(f"  X_e: {X_e.shape}")
    
    # 测试 e2v
    X_v = batch_hypergraph_e2v(X_e, H, D_v, aggr="mean")
    print(f"\ne2v 输出:")
    print(f"  X_v: {X_v.shape}")
    
    # 测试 v2v（端到端）
    X_out = batch_hypergraph_v2v(X, H, D_v, D_e, aggr="mean")
    print(f"\nv2v 输出:")
    print(f"  X_out: {X_out.shape}")
    
    # 验证 v2v = e2v(v2e(X))
    diff = torch.abs(X_v - X_out).max().item()
    print(f"\nv2v 与 e2v(v2e(X)) 差异: {diff:.6e}")
    assert diff < 1e-6, "v2v 和 e2v(v2e(X)) 结果不一致！"
    
    # 测试二部图转换
    bipartite_matrix = torch.zeros(batch_size, num_e, num_v)
    for b in range(batch_size):
        bipartite_matrix[b] = H[b].t()
    
    H_converted = bipartite_matrix_to_incidence_matrix(bipartite_matrix)
    diff2 = torch.abs(H - H_converted).max().item()
    print(f"\n二部图转换差异: {diff2:.6e}")
    assert diff2 < 1e-6, "二部图转换结果不一致！"
    
    print("\n" + "=" * 80)
    print("✓ 所有测试通过！")
    print("=" * 80)


if __name__ == "__main__":
    test_batch_hypergraph_ops()

