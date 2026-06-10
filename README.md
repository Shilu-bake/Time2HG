# Time2HG

基于 Shapelet 学习与 HGNNP 超图卷积的 UCR 时间序列分类项目。对时序样本提取 shapelet embedding，按时间邻近性构建二部图，经 HGNNP 增强特征后端到端训练，并支持 5 折交叉验证。

## 功能概览

1. 加载 UCR 数据集（训练集 + 测试集合并后做 k 折划分）
2. Shapelet embedding 提取（`ShapeletBasedModel`）
3. 基于时间邻近性的二部图构建
4. HGNNP 超图卷积增强 shapelet 特征
5. 5 折交叉验证训练与评估

## 项目结构

```
Time2HG/
├── run_k.py                        # 主入口脚本
├── requirements.txt                # Python 依赖
├── ts_utils.py                     # 数据集构建、随机种子等工具
├── data/
│   ├── preprocessing.py            # 数据加载、归一化、k 折划分
│   ├── shape_size_hyp.py           # 各数据集超参数配置
│   └── UCRArchive_2018/            # UCR 数据集目录（需自行放置）
├── models/
│   ├── Time2HGModel.py             # 核心网络模块（含 InceptionModule）
│   └── loss.py                     # 损失函数
└── Lab/
    ├── lab_test.py                 # ShapeletBasedModel 等模型定义
    ├── shapelet_learning.py        # Shapelet 学习层
    ├── hgnnp/                      # HGNNP 批量化实现
    └── DeepHypergraph-main/        # vendored dhg 库（无需 pip 安装）
```

## 环境要求

- Python >= 3.9
- PyTorch >= 1.12.1
- 建议使用 GPU（无 GPU 时自动回退到 CPU）

### 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：`torch`、`numpy`、`pandas`、`scikit-learn`、`scipy`、`timm`、`fastai`。

`dhg`（DeepHypergraph）已内置于 `Lab/DeepHypergraph-main/`，无需单独安装。

## 数据准备

将 [UCR Archive 2018](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/) 数据集按以下结构放入 `data/UCRArchive_2018/`：

```
data/UCRArchive_2018/
└── Car/
    ├── Car_TRAIN.tsv
    └── Car_TEST.tsv
```

每个数据集一个子目录，内含 `{数据集名}_TRAIN.tsv` 与 `{数据集名}_TEST.tsv` 两个文件。

## 快速开始

在 **Time2HG 根目录** 下运行：

```bash
python run_k.py --dataset Car --cuda cuda:0
```

指定数据路径：

```bash
python run_k.py --dataset Car --dataroot ./data/UCRArchive_2018 --cuda cuda:0
```

CPU 运行：

```bash
python run_k.py --dataset Car --cuda cpu
```

## 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | `Car` | UCR 数据集名称 |
| `--dataroot` | `./data/UCRArchive_2018` | 数据集根目录 |
| `--cuda` | `cuda:0` | 计算设备（如 `cuda:0`、`cpu`） |
| `--embed_dim` | `128` | 嵌入维度 |
| `--shape_size` | `16` | shapelet 大小（部分数据集会被超参字典覆盖） |
| `--stride` | `4` | shapelet 滑动步长 |
| `--depth` | `2` | shapelet 学习深度 |
| `--sparse_rate` | `0.5` | 最大稀疏率 |
| `--temporal_window` | `2.0` | 时间邻近连接窗口（取整后使用） |
| `--use_large_batch` | `1` | 是否使用大批次加速（1=是，0=否） |
| `--random_seed` | `42` | 随机种子 |

各数据集的 `shape_size`、`drop_out`、`warm_up_epoch` 等会自动从 `data/shape_size_hyp.py` 中的 `ucr_hyp_dict_shape_size` 读取。

## 输出说明

脚本会对每个 fold 进行最多 500 轮训练（含早停），最终在控制台打印：

- 每折最佳测试准确率
- 5 折平均测试准确率（`mean_test_acc`）

## 验证安装

```bash
python -c "from models.Time2HGModel import InceptionModule; from Lab.hgnnp.HGNNP import HGNNPWrapper; print('OK')"
```

## 许可证说明

- 本项目代码遵循原研究代码的使用方式
- `Lab/DeepHypergraph-main/` 为第三方库 [DHG](https://github.com/iMoonLab/DeepHypergraph)，遵循其 Apache-2.0 许可证
