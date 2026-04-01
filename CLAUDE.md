# GNN 可解释性分析项目

## 项目概述

本项目在 MolCLR-Urban 项目基础上扩展 GNN 可解释性分析能力，实现两个研究方向：

- **方向一（节点归因）**：识别对 UST（城市空间类型）分类贡献最大的地物节点
- **方向二（边归因）**：识别对 UST 分类影响最大的空间邻接关系类型

核心原则：**不修改任何现有文件**，只新增模块。

## 目录结构

```
/media/gy/study2/vibecoding/work2/code/
├── xai_config.py              # 配置文件（路径、超参数、类别映射）
├── run_dir1.py                 # 方向一入口脚本（节点归因）
├── run_dir2.py                 # 方向二入口脚本（边归因）
├── analysis/
│   ├── node_attribution.py     # GradCAM 节点归因
│   ├── gnnexplainer_attribution.py  # GNNExplainer 节点归因
│   ├── pgexplainer_attribution.py   # PGExplainer 节点归因
│   ├── graphmask_attribution.py     # GraphMASK 节点归因
│   ├── graphlime_attribution.py     # GraphLIME 节点归因
│   └── edge_attribution.py     # 边级归因分析
├── visualization/
│   ├── node_viz.py             # 节点重要性可视化
│   └ edge_viz.py               # 边重要性可视化
└── outputs/
    ├── results/                # 分析结果（PKL/CSV）
    └── figures/                # 可视化图表
```

## 数据路径配置

原始数据路径（只读，不修改）：
```python
DATA_CONFIG = {
    "npz_dir": "/media/gy/ssd/shanghai_exp/data_prepare/graph_dataset_0225_32_512",
    "tif_dir": "/media/gy/ssd/shanghai_exp/data_prepare/0225_512/shanghai",  # 语义分割图
    "jpg_dir": "/media/gy/DCA5-16F8/聚类结果/处理好的-二次筛选",  # 原始遥感图
    "label_csv": "/media/gy/DCA5-16F8/聚类结果/image_labels.csv",
}
```

模型权重路径：
```python
MODEL_CONFIG = {
    "finetune_weights": "/media/gy/ssd/shanghai_exp/MolCLR-Urban_alldata1127_512_cls/finetune/Mar27_16-48-21_urban_pretrain_fold3/checkpoints/model.pth",
    "num_classes": 17,
    "num_layer": 5,
    "emb_dim": 18,
    "feat_dim": 128,
}
```

## 使用方法

### 方向一：节点归因分析

支持 5 种可解释性方法：

```bash
# GradCAM（最快，约2分钟）
python run_dir1.py --method gradcam

# GNNExplainer（中等速度，约30分钟）
python run_dir1.py --method gnnexplainer --epochs 200

# PGExplainer（已优化，约5-10分钟）
python run_dir1.py --method pgexplainer --epochs 30

# GraphMASK（中等速度）
python run_dir1.py --method graphmask --epochs 50

# GraphLIME（已优化，约30分钟）
python run_dir1.py --method graphlime --samples 1000
```

### 方向二：边归因分析

```bash
python run_dir2.py --node_scores outputs/results/node_importance_scores.pkl
python run_dir2.py --node_scores outputs/results/node_importance_scores.pkl --aggregation product
```

## UST 类别（17类）

| ID | 名称 |
|---|---|
| 0 | 不透水面 |
| 1 | 高密度中层水平 |
| 2 | 高密度中层左倾斜 |
| 3 | 高密度中层右倾斜 |
| 4 | 低密度高层 |
| 5 | 低密度中层 |
| 6 | 大型中低层 |
| 7 | 高密度低层 |
| 8 | 中密度高层 |
| 9 | 高密度中层 |
| 10 | 城中村（高密度低层） |
| 11 | 中密度低层 |
| 12 | 农村 |
| 13 | 运动场 |
| 14 | 低密度低层 |
| 15 | 绿地 |
| 16 | 水体 |

## 节点语义类别（10类）

节点类别来自 NPZ 文件的 col3 字段：

| ID | 名称 | 颜色 |
|---|---|---|
| 0 | 不透水面 | 灰色 |
| 1 | 草地 | 浅绿 |
| 2 | 运动场 | 金色 |
| 3 | 树木 | 深绿 |
| 4 | 水体 | 蓝色 |
| 5 | 道路 | 深灰 |
| 6 | 低层建筑 | 浅橙 |
| 7 | 中层建筑 | 番茄红 |
| 8 | 高层建筑 | 深红 |
| 9 | 其他 | 紫色 |

## 关键接口

### dataset/dataset_finetune.py (MolCLR-Urban)

```python
from dataset.dataset_finetune import MolTestDataset

dataset = MolTestDataset(
    data_path=config.DATA_CONFIG["npz_dir"],
    csv_file=config.DATA_CONFIG["label_csv"],
    target=None,
    task='classification'
)

# 返回：(Data对象, 文件名) 元组
data, filename = dataset[idx]

# Data 字段：
# - x: [N, 20] 节点特征
# - edge_index: [2, E] 边索引
# - edge_attr: [E] 边属性
# - y: [1, 1] 标签
```

### models/gcn_finetune.py (MolCLR-Urban)

```python
from models.gcn_finetune import GCN

model = GCN(
    task="classification",
    num_layer=5,
    emb_dim=18,
    feat_dim=128,
    drop_ratio=0,
    pool="mean"
)

# forward 输入：data (含 x, edge_index, batch)
# forward 输出：(h, logits) 元组
h, logits = model(data)
# h: [batch_size, emb_dim] 图嵌入
# logits: [batch_size, num_classes] 分类输出
```

## 已知坑点处理

| 坑点 | 处理方案 |
|------|----------|
| forward 返回元组 | 取 `[1]` 获取 logits |
| DataParallel 权重前缀 | 加载时去除 `module.` |
| GradCAM 需要梯度 | 用 `model.train()` 但不 step |
| 变长图批处理 | 用 PyG DataLoader/Batch |
| CUDA OOM | 捕获后降 batch_size 为 1 |
| TIF 文件缺失 | 捕获 FileNotFoundError 跳过 |
| dataset one-hot bug | 从原始 NPZ 文件读取 col3 |

## 可视化配置

```python
VIZ_CONFIG = {
    "node_score_threshold": 0.5,  # 节点重要性阈值
    "edge_score_threshold": 0.3,  # 边重要性阈值
}

# 双底图模式：
# - 语义分割图（TIF）
# - 原始遥感图（JPG）
```

JPG 文件映射：
```python
UST_TO_JPG_FOLDER = {
    0: "不透水面",
    1: "高密度中层水平more_0.3",
    7: "高密度低层more_0.3",
    11: "中密度低层别墅区",
    ...
}
```

## 性能优化记录

### GraphLIME 优化（220x提速）

原问题：10000样本预计111小时

解决方案：
- 批量推理：使用 PyG Batch 对象合并多个扰动样本
- 批量大小：100（可调）
- 扰动样本数：从5000降到1000

效果：从111小时降到30分钟

### PGExplainer 优化（10-20x提速）

原问题：ETA约93小时

解决方案：
- 训练轮数：100→30，配合早停机制
- 抽样训练：只用20%数据训练解释器
- 批量解释：batch_size=50
- 向量化聚合：scatter_add/np.add.at 替代 for 循环

## 输出文件

### 方向一输出

```
outputs/
├── results/
│   ├── node_importance_scores_gradcam.pkl
│   ├── node_importance_scores_gnnexplainer.pkl
│   ├── node_importance_scores_pgexplainer.pkl
│   ├── node_importance_scores_graphmask.pkl
│   ├── node_importance_scores_graphlime.pkl
│   └── dir1_summary_stats_<method>.csv
├── figures/
│   ├── node_category_ust_heatmap_<method>.png
│   ├── dir1_node_importance_maps/<method>/UST*_<graph_key>.png
│   └── dir1_node_score_distribution/<method>/score_boxplot_per_ust.png
```

### 方向二输出

```
outputs/
├── results/
│   ├── dir2_edge_type_matrix.csv
│   └── dir2_top_edge_types.csv
├── figures/
│   ├── dir2_edge_type_heatmap.png
│   └ edge_importance_maps/UST*_<graph_key>_network.png
```

## 环境依赖

- Python 3.x
- PyTorch
- PyTorch Geometric (PyG)
- scikit-learn (用于 GraphLIME Ridge 回归)
- rasterio (用于 TIF 文件读取)
- matplotlib
- networkx

## 原始项目路径

```python
MOLCLR_ROOT = "/media/gy/ssd/shanghai_exp/MolCLR-Urban_alldata1127_512_cls"
```

这个路径包含 `models/` 和 `dataset/` 目录，需要添加到 sys.path。