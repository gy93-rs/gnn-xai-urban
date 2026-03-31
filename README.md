# GNN 可解释性分析项目

基于 MolCLR-Urban 项目，对城市结构类型（UST）分类模型进行可解释性分析。支持 5 种主流 GNN 解释方法，从图神经网络中提取对分类决策贡献最大的**节点**和**边**。

## 功能特性

- **5 种可解释性方法**：GradCAM、GNNExplainer、PGExplainer、GraphMASK、GraphLIME
- **节点级归因**：识别对分类贡献最大的地物节点
- **边级归因**：分析空间邻接关系对分类的影响
- **丰富可视化**：热力矩阵、重要性地图、网络图、箱线图

## 方法对比

| 方法 | 速度 | 边归因 | 特征归因 | 需要训练 | 适用场景 |
|------|------|--------|----------|----------|----------|
| GradCAM | ⚡ 最快 | ❌ | ❌ | ❌ | 快速分析 |
| GNNExplainer | 🐢 慢 | ✅ | ✅ | 每图优化 | 精细解释 |
| PGExplainer | 🚀 中等 | ✅ | ❌ | 一次性 | 大规模分析 |
| GraphMASK | 🚀 中等 | ✅ | ❌ | 一次性 | 稀疏解释 |
| GraphLIME | 🐢 慢 | ❌ | ✅ | ❌ | 特征分析 |

## 安装

### 环境要求

- Python 3.7+
- PyTorch 1.7+
- PyTorch Geometric 1.6+

### 依赖安装

```bash
# 核心依赖
pip install torch torchvision
pip install torch-geometric
pip install numpy pandas matplotlib seaborn networkx

# 可选依赖
pip install sklearn  # GraphLIME 需要
pip install rasterio  # TIF 背景图可视化
```

## 快速开始

### 1. 配置路径

编辑 `xai_config.py`，设置数据路径和模型权重路径：

```python
DATA_CONFIG = {
    "npz_dir": "path/to/graph_dataset",
    "tif_dir": "path/to/tif_files",
    "label_csv": "path/to/labels.csv",
}

MODEL_CONFIG = {
    "finetune_weights": "path/to/model.pth",
}
```

### 2. 运行节点归因分析

```bash
# GradCAM（默认，最快）
python run_dir1.py --method gradcam

# GNNExplainer（精细解释，同时输出边归因）
python run_dir1.py --method gnnexplainer --epochs 200

# PGExplainer（批量处理，效率高）
python run_dir1.py --method pgexplainer --epochs 100

# GraphMASK（稀疏解释）
python run_dir1.py --method graphmask --epochs 50

# GraphLIME（特征分析）
python run_dir1.py --method graphlime --samples 5000
```

### 3. 运行边归因分析

```bash
# 基于节点分数计算边重要性
python run_dir2.py --node_scores outputs/results/node_importance_scores_gradcam.pkl

# 可选聚合方式
python run_dir2.py --node_scores outputs/results/node_importance_scores_gradcam.pkl --aggregation max
```

## 项目结构

```
code/
├── xai_config.py                    # 配置文件
├── run_dir1.py                      # 节点归因入口
├── run_dir2.py                      # 边归因入口
├── analysis/
│   ├── node_attribution.py          # GradCAM 实现
│   ├── gnnexplainer_attribution.py  # GNNExplainer 实现
│   ├── pgexplainer_attribution.py   # PGExplainer 实现
│   ├── graphmask_attribution.py     # GraphMASK 实现
│   ├── graphlime_attribution.py     # GraphLIME 实现
│   └── edge_attribution.py          # 边归因分析
├── visualization/
│   ├── node_viz.py                  # 节点可视化
│   └── edge_viz.py                  # 边可视化
├── docs/
│   └── project_summary.md           # 项目总结
└── outputs/
    ├── results/                     # 结果文件
    └── figures/                     # 可视化图片
```

## 输出文件

```
outputs/
├── results/
│   ├── node_importance_scores_<method>.pkl  # 节点重要性分数
│   └── dir1_summary_stats_<method>.csv      # 统计汇总
└── figures/
    ├── node_category_ust_heatmap_<method>.png  # 热力矩阵
    ├── dir1_node_importance_maps/              # 节点重要性地图
    └── dir1_node_score_distribution/           # 箱线图
```

## 数据说明

### 节点特征（18 维）

| 列索引 | 含义 |
|--------|------|
| col0-1 | 归一化质心坐标 |
| col2 | 节点面积 |
| col3 | 节点类别 (0-9) |
| col4-10 | 非建筑节点类别占比 |
| col11-18 | 建筑节点几何特征 |

### 节点类别（10 类）

| ID | 名称 | ID | 名称 |
|----|------|----|------|
| 0 | 不透水面 | 5 | 道路 |
| 1 | 草地 | 6 | 低层建筑 |
| 2 | 运动场 | 7 | 中层建筑 |
| 3 | 树木 | 8 | 高层建筑 |
| 4 | 水体 | 9 | 其他 |

### UST 类别（17 类）

城市结构类型分类目标，包括高密度中层、低密度高层、城中村、农村、绿地、水体等。

## 参数说明

### 通用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--method` | gradcam | 解释方法 |
| `--device` | cuda | 计算设备 |
| `--viz_only` | False | 仅生成可视化 |
| `--viz_per_ust` | 3 | 每类 UST 可视化样本数 |
| `--seed` | 42 | 随机种子 |

### 方法特定参数

| 参数 | 适用方法 | 默认值 | 说明 |
|------|----------|--------|------|
| `--epochs` | GNNExplainer/PGExplainer/GraphMASK | 方法默认 | 训练轮数 |
| `--lr` | GNNExplainer/PGExplainer/GraphMASK | 方法默认 | 学习率 |
| `--samples` | GraphLIME | 5000 | 扰动样本数 |
| `--alpha` | GraphLIME | 1.0 | Ridge 正则化 |
| `--lambda_sparsity` | GraphMASK | 0.1 | 稀疏性惩罚 |

## 可视化示例

### 节点重要性热力矩阵

展示不同 UST 类别对各节点类型的依赖程度。

### 节点重要性地图

在遥感影像上叠加节点重要性热力圆，直观展示重要区域。

### 边类型重要性矩阵

分析不同空间邻接关系对分类的贡献。

## 依赖项目

本项目基于 [MolCLR](https://github.com/yuyangw/MolCLR) 进行迁移学习，使用其预训练模型进行城市结构分类。

## 引用

如果本项目对你的研究有帮助，请引用：

```bibtex
@misc{gnn-xai-urban,
  author = {gy93-rs},
  title = {GNN Explainability Analysis for Urban Structure Classification},
  year = {2026},
  url = {https://github.com/gy93-rs/gnn-xai-urban}
}
```

## License

MIT License