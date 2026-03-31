# GNN可解释性研究需求文档 v2.0
## 方向一（节点归因）+ 方向二（边归因）
## 基于 MolCLR-Urban 项目扩展

> **适用模型：** Claude Code + GLM-4.5
> **基础项目：** MolCLR-Urban（分子图对比学习迁移城市空间形态分类）
> **工作目录：** `/media/gy/study2/vibecoding/work2/code`
> **文档版本：** v2.0

---

## 0. 项目总览

### 0.1 背景与复用原则

本需求在已有 MolCLR-Urban 项目基础上**纯增量扩展**，不修改、不重写任何现有文件。
现有项目结构如下，所有标注 `【复用】` 的模块直接调用，不得重新实现：

```
/media/gy/study2/vibecoding/work2/code/   ← 工作目录
├── models/
│   ├── gcn_molclr2.py      【复用】预训练模型
│   └── gcn_finetune.py     【复用】微调分类模型 ← 可解释性的核心依赖
├── dataset/
│   ├── dataset_subgraph.py 【复用】预训练数据集
│   └── dataset_finetune.py 【复用】微调数据集 ← 数据加载直接调用此处
├── train_molclr.py         【不动】
├── finetune_urban_pretrain5fold.py  【不动】
├── mapping1_predict.py     【不动】
└── mapping2_generate_raster_class.py【不动】
```

**新增模块（本文档需要实现的部分）：**

```
/media/gy/study2/vibecoding/work2/code/
├── xai_config.py               ← 新增：可解释性专用配置
├── analysis/
│   ├── __init__.py
│   ├── node_attribution.py     ← 新增：方向一节点归因
│   └── edge_attribution.py     ← 新增：方向二边归因
├── visualization/
│   ├── __init__.py
│   ├── node_viz.py             ← 新增：节点重要性可视化
│   └── edge_viz.py             ← 新增：边重要性可视化
├── run_dir1.py                 ← 新增：方向一入口
├── run_dir2.py                 ← 新增：方向二入口
└── outputs/                    ← 新增：所有输出
    ├── figures/
    │   ├── dir1_node_importance_maps/
    │   ├── dir1_node_score_distribution/
    │   ├── dir2_edge_type_heatmap.png
    │   └── dir2_edge_importance_maps/
    └── results/
        ├── node_importance_scores.pkl
        ├── edge_importance_scores.pkl
        ├── dir1_summary_stats.csv
        └── dir2_edge_type_matrix.csv
```

### 0.2 研究目标

- **方向一（节点级归因）：** 对于每个城市局部图，哪些地物节点（建筑、道路、植被等）对GNN判定UST类别贡献最大？
- **方向二（边级归因）：** 哪些空间邻接关系类型（建筑-道路、建筑-植被等）对UST分类决策影响最大？

---

## 0.3 依赖说明

### 核心依赖

| 包名 | 用途 | 使用位置 |
|------|------|----------|
| `torch` | 深度学习框架 | 所有模块 |
| `torch_geometric` | 图神经网络 | node_attribution.py, edge_attribution.py |
| `numpy` | 数值计算 | 所有模块 |
| `pandas` | 数据处理 | 所有模块 |
| `matplotlib` | 可视化 | visualization/*.py |
| `seaborn` | 统计可视化 | visualization/*.py |
| `networkx` | 网络图绘制 | visualization/edge_viz.py |
| `pickle` | 序列化 | run_dir1.py, run_dir2.py |

### 可选依赖

| 包名 | 用途 | 使用位置 |
|------|------|----------|
| `rasterio` | 读取TIF文件 | visualization/node_viz.py（可选，缺失时跳过背景图） |
| `scipy` | 稀疏矩阵处理 | dataset/dataset_finetune.py（复用） |

### 标准库依赖

```python
import os, sys, argparse, logging, time, random
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union
```

### 安装命令

```bash
# 核心依赖
pip install torch torchvision
pip install torch-geometric
pip install numpy pandas matplotlib seaborn networkx

# 可选依赖（用于TIF背景图）
pip install rasterio
```

---

## 1. 路径与配置（`xai_config.py`）

**创建新文件 `xai_config.py`，不修改项目原有任何配置文件。**

```python
# xai_config.py
# GNN可解释性分析专用配置，在MolCLR-Urban项目根目录下创建

import os

# ── 工作根目录 ──────────────────────────────────────────
PROJECT_ROOT = "/media/gy/study2/vibecoding/work2/code"

# ── 原始数据路径（只读，不修改）────────────────────────
DATA_CONFIG = {
    "npz_dir":    "/media/gy/ssd/shanghai_exp/data_prepare/graph_dataset_0225_32_512",
    "tif_dir":    "/media/gy/ssd/shanghai_exp/data_prepare/0225_512/shanghai",
    "label_csv":  "/media/gy/DCA5-16F8/聚类结果/image_labels.csv",
    "lst_tif":    "/media/gy/ssd/UST与物理功能的相关性探索/src_data/lst.tif",
}

# ── 模型权重路径 ─────────────────────────────────────────
# 使用 fold3 微调权重（Mar27_16-48-21 实验，当前最优）
MODEL_CONFIG = {
    "finetune_weights": (
        "/media/gy/ssd/shanghai_exp/MolCLR-Urban_alldata1127_512_cls"
        "/finetune/Mar27_16-48-21_urban_pretrain_fold3"
        "/checkpoints/model.pth"
    ),
    "num_classes": 17,
    "device": "cuda",  # 若无GPU改为 "cpu"
}

# ── 输出目录 ─────────────────────────────────────────────
OUTPUT_CONFIG = {
    "root":      os.path.join(PROJECT_ROOT, "outputs"),
    "figures":   os.path.join(PROJECT_ROOT, "outputs/figures"),
    "results":   os.path.join(PROJECT_ROOT, "outputs/results"),
    "node_maps": os.path.join(PROJECT_ROOT, "outputs/figures/dir1_node_importance_maps"),
    "node_dist": os.path.join(PROJECT_ROOT, "outputs/figures/dir1_node_score_distribution"),
    "edge_maps": os.path.join(PROJECT_ROOT, "outputs/figures/dir2_edge_importance_maps"),
}

# ── 可解释性超参数 ────────────────────────────────────────
XAI_CONFIG = {
    "gradcam_layer":       "last",  # 使用最后一层GNN嵌入做GradCAM
    "batch_size":          32,      # GradCAM批量推断，OOM时自动降为1
    "gnnexplainer_epochs": 200,
    "gnnexplainer_lr":     0.01,
    "sample_per_ust":      5,       # GNNExplainer每类UST抽样数
    "viz_sample_per_ust":  3,       # 可视化每类UST抽样数（方向一）
    "random_seed":         42,
}

# ── UST类别名称（17类）────────────────────────────────────
UST_NAMES = {
    0: "不透水面",
    1: "高密度中层水平",
    2: "高密度中层左倾斜",
    3: "高密度中层右倾斜",
    4: "低密度高层",
    5: "低密度中层",
    6: "大型中低层",
    7: "高密度低层",
    8: "中密度高层",
    9: "高密度中层",
    10: "城中村（高密度低层）",
    11: "中密度低层",
    12: "农村",
    13: "运动场",
    14: "低密度低层",
    15: "绿地",
    16: "水体",
}

# ── 节点语义类别（col3 值对应，原始数据0-9）────────────────────────────
NODE_CATEGORY_NAMES = {
    0: "不透水面",
    1: "草地",
    2: "运动场",
    3: "树木",
    4: "水体",
    5: "道路",
    6: "低层建筑",
    7: "中层建筑",
    8: "高层建筑",
    9: "其他",
}

# 节点类别颜色（用于可视化）
NODE_CATEGORY_COLORS = {
    0: "#808080",  # 不透水面 - 灰色
    1: "#90EE90",  # 草地 - 浅绿
    2: "#FFD700",  # 运动场 - 金色
    3: "#228B22",  # 树木 - 深绿
    4: "#4169E1",  # 水体 - 蓝色
    5: "#A9A9A9",  # 道路 - 深灰
    6: "#FFA07A",  # 低层建筑 - 浅橙
    7: "#FF6347",  # 中层建筑 - 番茄红
    8: "#DC143C",  # 高层建筑 - 深红
    9: "#9370DB",  # 其他 - 紫色
}
```

---

## 2. 数据加载（直接复用现有代码）

### 2.1 复用策略

**直接导入并使用 `dataset/dataset_finetune.py` 中的数据集类，不新建 dataset.py。**

### 2.2 节点特征说明（19维）

| 列索引 | 含义 | 备注 |
|--------|------|------|
| col0, col1 | 归一化质心坐标 (x, y) | 范围 [0,1]，×512 得像素坐标 |
| col2 | 节点面积（像素数） | 用于可视化圆圈半径 |
| col3 | 节点类别（0-9） | **建筑节点为6/7/8** |
| col4–col10 | 非建筑节点各类别像素占比 | 建筑节点此处无意义 |
| col11–col18 | 建筑节点几何特征 | 非建筑节点此处无意义 |

---

## 3. 模型加载（复用 `models/gcn_finetune.py`）

### 3.1 复用策略

**直接导入并使用 `models/gcn_finetune.py` 中的模型类，不新建 gnn.py。**

---

## 4. 方向一：节点级归因（`analysis/node_attribution.py`）

### 4.1 核心功能

- `load_finetune_model()`: 加载微调模型权重
- `compute_node_scores_batch()`: 批量计算节点重要性分数（GradCAM）
- `NodeAttributionAnalyzer`: 统计分析类

### 4.2 GradCAM 实现要点

使用输入特征梯度的 L2 范数作为节点重要性，兼容旧版 PyTorch。

---

## 5. 方向二：边级归因（`analysis/edge_attribution.py`）

### 5.1 核心功能

- `get_edge_type()`: 生成边类型字符串（字母序拼接）
- `annotate_edge_types()`: 为所有边标注类型
- `compute_edge_importance()`: 计算边重要性（聚合两端节点分数）
- `EdgeAttributionAnalyzer`: 统计分析类

### 5.2 边类型命名规则

按字母序拼接两端节点类别名，避免重复：
- (7, 6) → "低层建筑-道路"
- (4, 7) → "树木-中层建筑"

---

## 6. 可视化

### 6.1 节点重要性可视化（`visualization/node_viz.py`）

- `plot_node_importance_map()`: 在语义分割底图上叠加节点重要性热力圆
- `plot_node_category_importance_heatmap()`: 节点类别×UST热力矩阵
- `plot_score_boxplot_per_ust()`: 分面箱线图

### 6.2 边重要性可视化（`visualization/edge_viz.py`）

- `plot_edge_type_heatmap()`: UST×边类型热力矩阵（带数值标注）
- `plot_edge_importance_network()`: 网络图（节点按坐标布局，边按重要性着色）

---

## 7. 入口脚本

### 7.1 `run_dir1.py`

```bash
用法：
  python run_dir1.py --method gradcam --batch_size 32
  python run_dir1.py --method gnnexplainer --sample_per_ust 5
  python run_dir1.py --viz_only --scores_path outputs/results/node_importance_scores.pkl
```

### 7.2 `run_dir2.py`

```bash
用法：
  python run_dir2.py --node_scores outputs/results/node_importance_scores.pkl
  python run_dir2.py --node_scores outputs/results/node_importance_scores.pkl --aggregation product
```

---

## 8. 运行结果

### 8.1 性能统计

| 指标 | 数值 |
|------|------|
| 样本总数 | 10,095 |
| 方向一耗时 | 约 7 分钟 |
| 方向二耗时 | 约 11 分钟 |
| 总耗时 | 约 18 分钟 |

### 8.2 输出文件

```
outputs/
├── results/
│   ├── node_importance_scores.pkl    # 节点分数（方向一核心输出）
│   ├── dir1_summary_stats.csv        # 节点统计汇总
│   ├── dir2_edge_type_matrix.csv     # 边类型矩阵（方向二核心输出，49种边类型）
│   └── dir2_top_edge_types.csv       # Top-K边类型
└── figures/
    ├── node_category_ust_heatmap.png           # 节点类别×UST热力矩阵
    ├── dir1_node_importance_maps/              # 节点重要性地图（51张）
    ├── dir1_node_score_distribution/           # 箱线图
    ├── dir2_edge_type_heatmap.png              # 边类型×UST热力矩阵
    └── dir2_edge_importance_maps/              # 边重要性网络图（17张）
```

---

## 9. 已知问题与解决方案

| 问题 | 解决方案 |
|------|----------|
| 数据集one_hot编码有bug（广播错误） | 直接从原始npz文件读取col3作为节点类别 |
| 类别9未定义 | 在xai_config.py中补充定义为"其他" |
| 权重state_dict含`module.`前缀 | 加载前统一去除 |
| CUDA OOM | 自动降batch_size为1重试 |
| TIF文件缺失 | 可视化时跳过并记录日志 |

---

## 10. 项目文档

- **需求文档**: `requirements.md`（本文档）
- **项目总结**: `docs/project_summary.md`
- **配置文件**: `xai_config.py`

---

*文档结束 | v2.0 | 已完成方向一、方向二实现*