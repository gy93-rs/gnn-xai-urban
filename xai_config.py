# xai_config.py
# GNN 可解释性分析专用配置
# 在 MolCLR-Urban 项目根目录下创建

import os
import sys

# ── 工作根目录 ──────────────────────────────────────────
PROJECT_ROOT = "/media/gy/study2/vibecoding/work2/code"

# ── 原始项目根目录（包含 dataset/ 和 models/）──────────────────────────────
MOLCLR_ROOT = "/media/gy/ssd/shanghai_exp/MolCLR-Urban_alldata1127_512_cls"

# 添加原始项目路径到 sys.path（用于导入 dataset 和 models）
if MOLCLR_ROOT not in sys.path:
    sys.path.insert(0, MOLCLR_ROOT)

# ── 原始项目根目录（包含 dataset/ 和 models/）────────────
MOLCLR_ROOT = "/media/gy/ssd/shanghai_exp/MolCLR-Urban_alldata1127_512_cls"

# ── 原始数据路径（只读，不修改）────────────────────────
DATA_CONFIG = {
    "npz_dir": "/media/gy/ssd/shanghai_exp/data_prepare/graph_dataset_0225_32_512",
    "tif_dir": "/media/gy/ssd/shanghai_exp/data_prepare/0225_512/shanghai",
    "label_csv": "/media/gy/DCA5-16F8/聚类结果/image_labels.csv",
    "lst_tif": "/media/gy/ssd/UST与物理功能的相关性探索/src_data/lst.tif",
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
    "num_layer": 5,
    "emb_dim": 18,      # 从权重文件推断
    "feat_dim": 128,    # 从权重文件推断
    "drop_ratio": 0,
    "pool": "mean",
    "task": "classification",
}

# ── 输出目录 ─────────────────────────────────────────────
OUTPUT_CONFIG = {
    "root": os.path.join(PROJECT_ROOT, "outputs"),
    "figures": os.path.join(PROJECT_ROOT, "outputs/figures"),
    "results": os.path.join(PROJECT_ROOT, "outputs/results"),
    "node_maps": os.path.join(PROJECT_ROOT, "outputs/figures/dir1_node_importance_maps"),
    "node_dist": os.path.join(PROJECT_ROOT, "outputs/figures/dir1_node_score_distribution"),
    "edge_maps": os.path.join(PROJECT_ROOT, "outputs/figures/dir2_edge_importance_maps"),
}

# ── 可解释性超参数 ────────────────────────────────────────
XAI_CONFIG = {
    "gradcam_layer": "last",  # 使用最后一层GNN嵌入做GradCAM
    "batch_size": 32,         # GradCAM批量推断，OOM时自动降为1
    "gnnexplainer_epochs": 200,
    "gnnexplainer_lr": 0.01,
    "sample_per_ust": 5,      # GNNExplainer每类UST抽样数
    "viz_sample_per_ust": 3,  # 可视化每类UST抽样数（方向一）
    "random_seed": 42,
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
# 注意：数据集加载器one_hot编码有bug，直接读取原始col3
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
    9: "其他",  # 数据中存在但未定义的类别
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


def create_output_dirs():
    """创建所有输出目录"""
    for key, path in OUTPUT_CONFIG.items():
        os.makedirs(path, exist_ok=True)
        print(f"[xai_config] 创建目录: {path}")


def setup_random_seed(seed: int = None):
    """设置随机种子"""
    import random
    import numpy as np
    import torch

    if seed is None:
        seed = XAI_CONFIG["random_seed"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[xai_config] 随机种子设置为: {seed}")