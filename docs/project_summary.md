# GNN可解释性分析项目总结

## 项目背景

基于 MolCLR-Urban 项目，对城市结构类型（UST）分类模型进行可解释性分析。目标是从图神经网络中提取对分类决策贡献最大的**节点**和**边**。

---

## 已完成工作

### 方向一：节点级归因分析（GradCAM）

| 内容 | 说明 |
|------|------|
| 方法 | GradCAM（梯度加权类激活映射） |
| 算法原理 | 输入特征梯度 L2 范数：`importance_v = ||∂score_y/∂x_v||₂`，归一化到 [0,1] |
| 输入 | 10,095个图样本，17类UST |
| 输出 | 每个节点的重要性分数（归一化到[0,1]） |
| 可视化 | 热力矩阵图 + 51张节点重要性地图 + 箱线图 |

**核心发现**：不同UST类别对各节点类型的依赖程度不同，建筑类节点在高密度城区样本中重要性更高。

---

### 方向二：边级归因分析

| 内容 | 说明 |
|------|------|
| 方法 | 基于节点分数聚合（边重要性 = 两端节点分数均值） |
| 算法原理 | 边分数 = 聚合(源节点分数, 目标节点分数)，聚合方式可选：mean/max/product |
| 边类型 | 49种（按两端节点类别组合，字母序命名） |
| 输出 | UST×边类型重要性矩阵（17行×49列） |
| 可视化 | 热力矩阵图 + 17张网络图（每类UST一张） |

**核心发现**：建筑相关边类型（如"低层建筑-高层建筑"、"不透水面-高层建筑"）在多个UST类别中具有较高重要性。

---

## 项目结构

```
code/
├── xai_config.py              # 配置文件（路径、类别名称、颜色）
├── run_dir1.py                # 方向一入口脚本
├── run_dir2.py                # 方向二入口脚本
├── analysis/
│   ├── node_attribution.py    # GradCAM节点归因
│   └── edge_attribution.py    # 边归因分析
├── visualization/
│   ├── node_viz.py            # 节点可视化（热力图、地图）
│   └── edge_viz.py            # 边可视化（热力矩阵、网络图）
└── outputs/
    ├── results/               # CSV和PKL结果文件
    └── figures/               # 可视化图片
```

---

## 类别映射

### 节点类别（10类）

| 类别ID | 名称 | 颜色 | 出现次数 |
|--------|------|------|----------|
| 0 | 不透水面 | 灰色 #808080 | 106,134 |
| 1 | 草地 | 浅绿 #90EE90 | 19,497 |
| 2 | 运动场 | 金色 #FFD700 | 9,755 |
| 3 | 树木 | 深绿 #228B22 | 1,215 |
| 4 | 水体 | 蓝色 #4169E1 | 50,956 |
| 5 | 道路 | 深灰 #A9A9A9 | 8,214 |
| 6 | 低层建筑 | 浅橙 #FFA07A | 40,863 |
| 7 | 中层建筑 | 番茄红 #FF6347 | 8,333 |
| 8 | 高层建筑 | 深红 #DC143C | 15,448 |
| 9 | 其他 | 紫色 #9370DB | 3,996 |

### UST类别（17类）

| ID | 名称 | ID | 名称 |
|----|------|----| -----|
| 0 | 不透水面 | 9 | 高密度中层 |
| 1 | 高密度中层水平 | 10 | 城中村（高密度低层） |
| 2 | 高密度中层左倾斜 | 11 | 中密度低层 |
| 3 | 高密度中层右倾斜 | 12 | 农村 |
| 4 | 低密度高层 | 13 | 运动场 |
| 5 | 低密度中层 | 14 | 低密度低层 |
| 6 | 大型中低层 | 15 | 绿地 |
| 7 | 高密度低层 | 16 | 水体 |
| 8 | 中密度高层 | | |

---

## 关键技术修复

### 1. 节点类别提取问题

**问题描述**：数据集加载器的one_hot编码有bug，导致无法正确恢复节点类别。

**原因分析**：
```python
# 原代码（有bug）
one_hot[:, x1[:, 3].astype(int) - 1] = 1
# 这行代码把样本中所有出现的类别都设置到了每个节点上（广播错误）
```

**解决方案**：直接从原始npz文件读取col3列作为节点类别：
```python
npz_path = os.path.join(config.DATA_CONFIG["npz_dir"], filename)
raw_data = np.load(npz_path)
raw_col3 = raw_data['array1'][:, 3].astype(np.int64)
data.node_cat = torch.tensor(raw_col3, dtype=torch.long)
```

### 2. 类别9处理

**问题描述**：原始数据中存在类别9，但配置文件中未定义。

**解决方案**：在`xai_config.py`中补充类别9定义为"其他"。

### 3. 边类型命名

**规则**：按字母序拼接两端节点类别名，避免(A-B)和(B-A)重复。

**示例**：
- 源节点类别=6（低层建筑），目标节点类别=5（道路）
- 边类型名称 = "低层建筑-道路"（字母序）

### 4. 可视化优化

- 热力矩阵：添加数值标注（保留2位小数，字体8pt）
- 网络图：使用固定调色板（NODE_CATEGORY_COLORS）
- 节点大小：缩放到[50, 500]范围

---

## 运行性能

| 指标 | 数值 |
|------|------|
| 样本总数 | 10,095 |
| 方向一耗时 | 7分24秒 |
| 方向二耗时 | 11分01秒 |
| 总耗时 | 18分25秒 |

---

## 输出文件清单

```
outputs/
├── results/
│   ├── node_importance_scores.pkl    # 节点分数（方向一核心输出）
│   ├── dir1_summary_stats.csv        # 节点统计汇总
│   ├── dir2_edge_type_matrix.csv     # 边类型矩阵（方向二核心输出）
│   └── dir2_top_edge_types.csv       # Top-K边类型
└── figures/
    ├── node_category_ust_heatmap.png           # 节点类别×UST热力矩阵
    ├── dir1_node_importance_maps/              # 节点重要性地图（51张）
    ├── dir1_node_score_distribution/           # 箱线图
    ├── dir2_edge_type_heatmap.png              # 边类型×UST热力矩阵
    └── dir2_edge_importance_maps/              # 边重要性网络图（17张）
```

---

## 使用方法

### 运行方向一（节点归因）

```bash
python run_dir1.py
```

### 运行方向二（边归因）

```bash
python run_dir2.py --node_scores outputs/results/node_importance_scores.pkl
```

### 可选参数

方向二支持以下参数：
- `--aggregation`: 边重要性聚合方式（默认`mean`，可选`max`、`product`）
- `--seed`: 随机种子（默认42）

---

## 依赖说明

- PyTorch
- PyTorch Geometric
- NumPy
- Pandas
- Matplotlib
- Seaborn
- NetworkX

---

## 文档信息

- **创建日期**: 2026-03-28
- **项目路径**: `/media/gy/study2/vibecoding/work2/code`
- **数据路径**: `/media/gy/ssd/shanghai_exp/data_prepare/graph_dataset_0225_32_512`