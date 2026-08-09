"""
UCR数据集的Shapelet Embedding提取和HGNNP训练脚本（无Segment向量化）

功能：
1. 加载指定的UCR数据集训练集和测试集
2. 对每个时序样本提取shapelet embeddings（使用学习模型）
3. 基于时间邻近性构建二部图连接模式
4. 使用HGNNP超图卷积增强shapelet特征
5. 进行端到端训练和5折交叉验证
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# 添加项目根目录到路径
rootPath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, rootPath)

# 导入数据处理函数
from ts_utils import set_seed, build_loss, evaluate_model, build_dataset, get_all_datasets
from data.preprocessing import normalize_per_series, load_data, load_data_split, transfer_labels, fill_nan_value
from data.shape_size_hyp import ucr_hyp_dict_shape_size

# 导入模型类
from Lab.lab_test import (
    ShapeletBasedModel,
    assign_temporal_shapelets
)

# 导入 HGNNP 超图卷积模块
from Lab.hgnnp.HGNNP import HGNNPWrapper


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='UCR数据集的Embedding提取（无Segment）')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, default='Car',
                        help='UCR数据集名称，如 CBF, GunPoint')

    default_dataroot = os.path.join(rootPath, 'data', 'UCRArchive_2018')
    parser.add_argument('--dataroot', type=str, 
                        default=default_dataroot,
                        help='UCR数据集根目录路径')
    parser.add_argument('--random_seed', type=int, default=42, 
                        help='随机种子')
    
    # 模型参数
    parser.add_argument('--shape_size', type=int, default=16, 
                        help='shapelet的大小')
    parser.add_argument('--embed_dim', type=int, default=128, 
                        help='嵌入维度')
    parser.add_argument('--stride', type=int, default=4,
                        help='shapelet的滑动步长')
    parser.add_argument('--depth', type=int, default=2, 
                        help='shapelet学习的深度')
    parser.add_argument('--num_experts', type=int, default=8, 
                        help='MoE专家数量（通常等于类别数）')
    parser.add_argument('--sparse_rate', type=float, default=0.5, 
                        help='最大稀疏率')
    parser.add_argument('--moe_loss_rate', type=float, default=0.003,
                        help='MoE损失权重')
    parser.add_argument('--warm_up_epoch', type=int, default=50,
                        help='Warm-up轮数')
    parser.add_argument('--drop_out', type=float, default=0.1,
                        help='HGNNP_dropout')
    
    # 处理参数
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='批处理大小')
    parser.add_argument('--use_large_batch', type=int, default=1, 
                        help='1 是 True，0 是 False，使用大批次加速训练')
    parser.add_argument('--cuda', type=str, default='cuda:0',
                        help='使用的GPU设备')
    parser.add_argument('--temporal_window', type=float, default=2.0,
                        help='时间邻近连接的窗口大小（将被转换为整数 segment 范围）')
    
    return parser.parse_args()


def build_bipartite_graph_batch_temporal(shapelet_embeddings, 
                                        temporal_connections, device):
    """
    基于时间邻近性构建二部图矩阵（用于端到端训练，无segment embeddings）
    
    参数:
        shapelet_embeddings: tensor, shape=(batch_size, num_patches, embed_dim)
        temporal_connections: List[List[int]], 预计算的时间邻近连接
                             每个元素是一个 segment 连接的 shapelet 索引列表
        device: torch 设备
    
    返回:
        bipartite_matrix: tensor, shape=(batch_size, num_segments, num_patches)
                         矩阵中的值为1表示该时间位置与该 shapelet 有连接（时间邻近）
    """
    batch_size, num_patches, _ = shapelet_embeddings.shape
    num_segments = len(temporal_connections)
    
    # 初始化全零矩阵
    bipartite_matrix = torch.zeros(batch_size, num_segments, num_patches, device=device)
    
    # 根据预计算的时间连接填充（对整个 batch 使用相同的连接模式）
    for seg_idx, connected_shapes in enumerate(temporal_connections):
        for shape_idx in connected_shapes:
            if shape_idx < num_patches:  # 边界检查
                # 对 batch 中的所有样本使用相同的时间连接模式
                bipartite_matrix[:, seg_idx, shape_idx] = 1.0
    
    return bipartite_matrix


def forward_pass(batch_data, shapelet_model, hgnnp_model,
                 temporal_connections=None, epoch=None, warm_up_epoch=None):
    """
    统一的前向传播函数（训练、验证、测试共用）
    
    参数:
        batch_data: (batch_size, in_chans, seq_len) - 输入时间序列
        shapelet_model: ShapeletBasedModel 实例
        hgnnp_model: HGNNPWrapper 实例
        temporal_connections: List[List[int]], 预计算的时间邻近连接
        epoch: 当前训练轮数（训练和测试都应显式传入）
        warm_up_epoch: Warm-up 轮数
    
    返回:
        logits: (batch_size, num_classes) - 分类 logits
        moe_loss: MoE 负载均衡损失（shapelet）
    """
    # 始终显式传入 epoch 和 warm_up_epoch，避免使用默认值导致稀疏化行为不一致
    shapelet_emb, index_map, moe_loss = shapelet_model(
        batch_data, num_epoch_i=epoch, warm_up_epoch=warm_up_epoch
    )

    # 基于原始 temporal_connections 和 index_map，为每个样本构建与稀疏化后序列对齐的二部图
    batch_size = batch_data.shape[0]
    num_segments = len(temporal_connections)
    num_patches_final = shapelet_emb.shape[1]

    bipartite_matrix = torch.zeros(
        batch_size, num_segments, num_patches_final, device=batch_data.device
    )

    # temporal_connections[seg] 中存的是原始 patch 索引（0..num_patches_initial-1）
    # index_map[b, orig_idx] 给出该原始 patch 在最终序列中的位置（聚合到同一 token 的会映射到相同位置）
    # 这里用向量化的方式，一次性为整个 batch 写入，避免 Python 的双重 for 循环
    for seg_idx, connected_shapes in enumerate(temporal_connections):
        if not connected_shapes:
            continue

        # (L,)
        orig_idx_tensor = torch.as_tensor(
            connected_shapes, device=batch_data.device, dtype=torch.long
        )

        # 对 batch 内每个样本，根据 index_map 计算映射后的索引 (B, L)
        mapped_indices = index_map[:, orig_idx_tensor]  # (B, L)

        # 构造 batch 维度索引 (B, L)
        b_idx = torch.arange(batch_size, device=batch_data.device).unsqueeze(1).expand_as(mapped_indices)

        # 构造 segment 维度索引 (B, L)，每个位置都是 seg_idx
        seg_idx_tensor = torch.full_like(mapped_indices, seg_idx)

        # 利用高级索引一次性写入 (将对应位置置为 1)
        bipartite_matrix[b_idx, seg_idx_tensor, mapped_indices] = 1.0
    
    # HGNNP 使用 MoE 学习后的 shapelet_emb 作为节点特征，且图结构与其索引严格对齐
    enhanced_emb = hgnnp_model(bipartite_matrix, shapelet_emb)

    # 使用 attention 加权分类 - 对增强后的特征进行分类
    representation = shapelet_model.get_representation_with_attention(enhanced_emb)

    logits = shapelet_model.repr_classifier(representation)

    
    return logits, moe_loss



def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args)
    
    # 设置设备
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 使用 build_dataset 分别加载训练集和测试集
    print(f"\n正在加载数据集 {args.dataset} 的训练集和测试集...")
    train_dataset, train_target, test_dataset, test_target, num_classes = build_dataset(args)
    args.num_experts = num_classes

    # 获取序列长度（以训练集为准）
    seq_len = train_dataset.shape[1]
    
    print(f"数据信息:")
    print(f"  - 训练集样本数: {train_dataset.shape[0]}")
    print(f"  - 测试集样本数: {test_dataset.shape[0]}")
    print(f"  - 序列长度: {seq_len}")
    print(f"  - 类别数: {num_classes}")
    
    # 从超参数字典读取数据集特定的配置
    print(f"\n从超参数字典读取数据集特定配置...")
    args.shape_size = ucr_hyp_dict_shape_size[args.dataset]['shape_size']
    args.shape_use_ratio = ucr_hyp_dict_shape_size[args.dataset]['shape_use_ratio']
    args.shape_ratio = ucr_hyp_dict_shape_size[args.dataset]['shape_ratio']
    args.drop_out = ucr_hyp_dict_shape_size[args.dataset]['drop_out']
    args.warm_up_epoch = ucr_hyp_dict_shape_size[args.dataset]['warm_up_epoch']
    
    print(f"数据集 {args.dataset} 的超参数配置:")
    print(f"  - shape_size: {args.shape_size}")
    print(f"  - shape_use_ratio: {args.shape_use_ratio}")
    print(f"  - shape_ratio: {args.shape_ratio}")
    print(f"  - drop_out: {args.drop_out}")
    print(f"  - warm_up_epoch: {args.warm_up_epoch}")
    
    # 如果使用比例模式，根据序列长度计算 shape_size
    if args.shape_use_ratio == 1:
        args.shape_size = int(seq_len * args.shape_ratio)
        if args.shape_size <= args.stride:
            args.stride = min(2, args.shape_size)
        print(f"  - 使用比例模式，计算后的 shape_size: {args.shape_size}")
    
    # 确保 stride 不大于 shape_size
    if args.stride > args.shape_size:
        args.stride = args.shape_size
        print(f"  - 调整 stride 为: {args.stride}")
    
    # ========== 批次大小配置（基于训练集）==========
    print(f"\n基于训练集配置批次大小...")
    args.batch_size = int(min(train_dataset.shape[0] * 0.6 / 10, 16))

    # 使用大批次加速训练（后续各折会复用该batch_size）
    if args.use_large_batch == 1:
        args.batch_size = min(512, train_dataset.shape[0])

    print(f"\n批次大小配置:")
    print(f"  - use_large_batch: {args.use_large_batch}")
    print(f"  - batch_size: {args.batch_size}")

    # ========== 创建模型（所有折共享）==========
    print("\n" + "=" * 80)
    print("创建模型（所有折共享）")
    print("=" * 80)

    shapelet_model = ShapeletBasedModel(
        seq_len=seq_len,
        shape_size=args.shape_size,
        in_chans=1,
        embed_dim=args.embed_dim,
        stride=args.stride,
        depth=args.depth,
        num_experts=args.num_experts,
        sparse_rate=args.sparse_rate,
        num_classes=num_classes  # 添加分类头
    ).to(device)

    hgnnp_model = HGNNPWrapper(
        in_channels=args.embed_dim,
        hid_channels=args.embed_dim,
        out_channels=args.embed_dim,
        use_bn=True,
        drop_rate=args.drop_out,
        use_residual=True
    ).to(device)
    
    print(f"Shapelet 模型: {shapelet_model.__class__.__name__}")
    print(f"  - 输入维度: {args.embed_dim}")
    print(f"  - Shapelet 大小: {args.shape_size}")
    print(f"  - 类别数: {num_classes}")
    
    # ========== 预计算时间邻近连接（所有折共享）==========
    print(f"\n预计算时间邻近连接...")
    # 将传入的 temporal_window 浮点数四舍五入为整数，用于控制连接前后 N 个 segment 范围
    temporal_window = int(round(args.temporal_window))
    if temporal_window < 1:
        temporal_window = 1
    temporal_connections, num_segments, num_patches = assign_temporal_shapelets(
        segment_size=args.shape_size,
        shapelet_size=args.shape_size,
        stride=args.stride,
        seq_len=seq_len,
        temporal_window=temporal_window,
        device=device
    )
    
    print(f"  - Segment 数量（时间位置数）: {num_segments}")
    print(f"  - Shapelet 数量: {num_patches}")
    print(f"  - 平均每个时间位置连接 {sum(len(c) for c in temporal_connections) / len(temporal_connections):.1f} 个 shapelet")
    
    # 打印连接统计
    min_connections = min(len(c) for c in temporal_connections)
    max_connections = max(len(c) for c in temporal_connections)
    print(f"  - 最少连接数: {min_connections}, 最多连接数: {max_connections}")
    print(f"  - 时间窗口原始输入: {args.temporal_window}, 实际使用整数: {temporal_window} 个 segment 范围")

    # 保存初始权重，供每折复用
    shapelet_init_state = shapelet_model.state_dict()
    hgnnp_init_state = hgnnp_model.state_dict()

    # ========== 构建k折数据集 ==========
    print("\n构建5折 train/val/test 划分...")
    train_datasets, train_targets, val_datasets, val_targets, test_datasets, test_targets = get_all_datasets(
        train_dataset, train_target, test_dataset, test_target
    )

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    test_accuracies = []
    total_train_time = 0.0

    num_epochs = 500

    for fold_idx, train_dataset in enumerate(train_datasets):
        print("\n" + "=" * 80)
        print(f"Fold {fold_idx} 开始训练与评估")
        print("=" * 80)

        # 每折恢复初始权重
        shapelet_model.load_state_dict(shapelet_init_state)
        hgnnp_model.load_state_dict(hgnnp_init_state)

        # 为每折重新创建优化器和调度器
        optimizer = torch.optim.Adam(
            list(shapelet_model.parameters()) +
            list(hgnnp_model.parameters()),
            lr=0.001,
            weight_decay=0.0
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=400, eta_min=1e-5
        )

        train_target = train_targets[fold_idx]
        val_dataset = val_datasets[fold_idx]
        val_target = val_targets[fold_idx]
        test_dataset = test_datasets[fold_idx]
        test_target = test_targets[fold_idx]

        # 添加 channel 维度
        train_dataset = train_dataset[:, :, np.newaxis]
        val_dataset = val_dataset[:, :, np.newaxis]
        test_dataset = test_dataset[:, :, np.newaxis]

        # 数据预处理：填充NaN并归一化
        train_dataset = train_dataset.astype(np.float32)
        val_dataset = val_dataset.astype(np.float32)
        test_dataset = test_dataset.astype(np.float32)

        train_dataset, val_dataset, test_dataset = fill_nan_value(
            train_dataset, val_dataset, test_dataset
        )

        train_dataset = normalize_per_series(train_dataset)
        val_dataset = normalize_per_series(val_dataset)
        test_dataset = normalize_per_series(test_dataset)

        # 转为Tensor
        train_tensor = torch.from_numpy(train_dataset).float().to(device)
        train_labels = torch.from_numpy(train_target).long().to(device)
        val_tensor = torch.from_numpy(val_dataset).float().to(device)
        val_labels = torch.from_numpy(val_target).long().to(device)
        test_tensor = torch.from_numpy(test_dataset).float().to(device)
        test_labels = torch.from_numpy(test_target).long().to(device)

        # DataLoader
        train_loader = DataLoader(
            TensorDataset(train_tensor, train_labels),
            batch_size=args.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(val_tensor, val_labels),
            batch_size=args.batch_size,
            shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(test_tensor, test_labels),
            batch_size=args.batch_size,
            shuffle=False
        )

        last_loss = float('inf')
        stop_count = 0
        increase_count = 0
        min_val_loss = float('inf')
        best_test_acc = 0.0

        for epoch in range(num_epochs):
            if stop_count == 80 or increase_count == 80:
                print(f'Fold {fold_idx} 在第 {epoch} 轮早停')
                break

            shapelet_model.train()
            hgnnp_model.train()

            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.transpose(1, 2)  # (B, 1, L)

                optimizer.zero_grad()

                logits, moe_loss = forward_pass(
                    batch_data, shapelet_model, hgnnp_model,
                    temporal_connections=temporal_connections,
                    epoch=epoch, warm_up_epoch=args.warm_up_epoch
                )

                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = logits.max(1)
                train_total += batch_labels.size(0)
                train_correct += predicted.eq(batch_labels).sum().item()

            scheduler.step()

            train_loss /= len(train_loader)
            train_acc = 100.0 * train_correct / train_total

            # 验证集评估
            shapelet_model.eval()
            hgnnp_model.eval()

            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    batch_data = batch_data.transpose(1, 2)

                    logits, moe_loss = forward_pass(
                        batch_data, shapelet_model, hgnnp_model,
                        temporal_connections=temporal_connections,
                        epoch=epoch, warm_up_epoch=args.warm_up_epoch
                    )

                    loss = criterion(logits, batch_labels)
                    val_loss += loss.item()
                    _, predicted = logits.max(1)
                    val_total += batch_labels.size(0)
                    val_correct += predicted.eq(batch_labels).sum().item()

            val_loss /= len(val_loader)
            val_acc = 100.0 * val_correct / val_total

            # 根据验证集loss更新早停与最佳测试表现
            if min_val_loss > val_loss:
                min_val_loss = val_loss

                # 在验证最优时，在测试集上评估一次
                test_correct = 0
                test_total = 0

                with torch.no_grad():
                    for batch_data, batch_labels in test_loader:
                        batch_data = batch_data.transpose(1, 2)

                        logits, moe_loss = forward_pass(
                            batch_data, shapelet_model, hgnnp_model,
                            temporal_connections=temporal_connections,
                            epoch=epoch, warm_up_epoch=args.warm_up_epoch
                        )

                        _, predicted = logits.max(1)
                        test_total += batch_labels.size(0)
                        test_correct += predicted.eq(batch_labels).sum().item()

                if test_total > 0:
                    best_test_acc = 100.0 * test_correct / test_total
                else:
                    best_test_acc = 0.0

            if (epoch > args.warm_up_epoch) and (abs(last_loss - val_loss) <= 1e-4):
                stop_count += 1
            else:
                stop_count = 0

            if (epoch > args.warm_up_epoch) and (val_loss > last_loss):
                increase_count += 1
            else:
                increase_count = 0

            last_loss = val_loss

            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(
                    f"Fold {fold_idx} | Epoch [{epoch+1}/{num_epochs}] "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
                    f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% "
                    f"| Best Test Acc: {best_test_acc:.2f}%"
                )

        print(f"Fold {fold_idx} 结束训练，Best Test Acc: {best_test_acc:.2f}%")
        test_accuracies.append(best_test_acc)

    # 跨折统计
    test_accuracies = np.array(test_accuracies)
    mean_test_acc = np.mean(test_accuracies) if len(test_accuracies) > 0 else 0.0

    print("\n" + "=" * 60)
    print("所有折训练完成！")
    print(f"  - 每折测试Accuracy: {test_accuracies}")
    print(f"  - 5折平均测试Accuracy (mean_test_acc): {mean_test_acc:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
