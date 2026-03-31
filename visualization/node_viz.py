"""
节点重要性可视化模块
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import seaborn as sns

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

import xai_config as config

logger = logging.getLogger(__name__)


# 节点类别颜色映射
NODE_CAT_COLORS = {
    0: "#808080",   # background - gray
    1: "#A0A0A0",   # impervious - light gray
    2: "#90EE90",   # grass - light green
    3: "#32CD32",   # sports_field - lime green
    4: "#228B22",   # tree - forest green
    5: "#4169E1",   # water - royal blue
    6: "#696969",   # road - dim gray
    7: "#FFD700",   # low_bld - gold
    8: "#FFA500",   # mid_bld - orange
    9: "#FF4500",   # high_bld - orange red
}


# 语义分割配色方案
SEMANTIC_COLORS = {
    # 地物类别 (ch0)
    0: (240, 240, 240),  # 背景 - 浅灰
    1: (180, 180, 180),  # 不透水面 - 中灰
    2: (34, 139, 34),    # 草地 - 森林绿
    3: (144, 238, 144),  # 运动场 - 浅绿
    4: (0, 100, 0),      # 乔木 - 深绿
    5: (30, 144, 255),   # 水体 - 道奇蓝
}

ROAD_COLOR = (255, 165, 0)      # 道路 - 橙色
ROAD_ALPHA = 0.7

BUILDING_COLORS = {
    1: (255, 200, 100),  # 低层建筑 - 浅黄
    2: (220, 100, 50),   # 中层建筑 - 橙红
    3: (160, 32, 32),    # 高层建筑 - 深红
}

# 语义类别名称
SEMANTIC_NAMES = {
    0: "背景",
    1: "不透水面",
    2: "草地",
    3: "运动场",
    4: "乔木",
    5: "水体",
    "road": "道路",
    "bld_low": "低层建筑",
    "bld_mid": "中层建筑",
    "bld_high": "高层建筑",
}


def _render_semantic_map(ch0: np.ndarray, ch1: np.ndarray, ch2: np.ndarray) -> np.ndarray:
    """
    根据语义通道渲染 RGB 图像。

    合成逻辑（按优先级从低到高叠加）：
    1. 先用 ch0 的类别配色填充底图
    2. ch1==1 的像素用道路橙色覆盖（alpha混合）
    3. ch2>0 的像素用对应建筑颜色覆盖（完全覆盖，不透明）

    Args:
        ch0: 地物类别通道 (H, W)
        ch1: 道路掩码通道 (H, W)
        ch2: 建筑高度类型通道 (H, W)

    Returns:
        rgb: (H, W, 3) RGB 图像，值范围 [0, 255]
    """
    h, w = ch0.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    # Step 1: 用 ch0 的类别配色填充底图
    for cat_id, color in SEMANTIC_COLORS.items():
        mask = (ch0 == cat_id)
        rgb[mask] = color

    # Step 2: ch1==1 的像素用道路橙色覆盖（alpha混合）
    road_mask = (ch1 == 1)
    if road_mask.any():
        # Alpha 混合: result = alpha * road_color + (1-alpha) * base_color
        base_color = rgb[road_mask]
        road_color_arr = np.array(ROAD_COLOR, dtype=np.float32)
        rgb[road_mask] = ROAD_ALPHA * road_color_arr + (1 - ROAD_ALPHA) * base_color

    # Step 3: ch2>0 的像素用对应建筑颜色覆盖（完全覆盖，不透明）
    for bld_type, color in BUILDING_COLORS.items():
        bld_mask = (ch2 == bld_type)
        rgb[bld_mask] = color

    return rgb.astype(np.uint8)


def plot_node_importance_map(
    graph_key: str,
    node_scores: np.ndarray,
    data,
    tif_dir: str,
    output_path: str,
    ust_label: int = None,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    在语义分割底图上叠加节点重要性热力圆。

    Args:
        graph_key: 图标识符
        node_scores: [N] 节点重要性分数
        data: Data 对象（含 x, edge_index）
        tif_dir: TIF 文件目录
        output_path: 输出图片路径
        ust_label: UST 类别标签
        figsize: 图片尺寸
    """
    try:
        import rasterio
    except ImportError:
        logger.warning("rasterio 未安装，跳过 TIF 背景图")
        # 不使用背景图，直接绘制节点
        _plot_node_importance_simple(graph_key, node_scores, data, output_path, ust_label, figsize)
        return

    tif_path = os.path.join(tif_dir, f"{graph_key}.tif")

    # 读取 TIF 文件
    try:
        with rasterio.open(tif_path) as src:
            # 读取三通道
            ch0 = src.read(1)  # 地物类别
            ch1 = src.read(2)  # 道路掩码
            ch2 = src.read(3)  # 建筑高度类型
    except FileNotFoundError:
        logger.warning(f"TIF 文件不存在: {tif_path}")
        _plot_node_importance_simple(graph_key, node_scores, data, output_path, ust_label, figsize)
        return

    # 渲染语义分割图
    rgb = _render_semantic_map(ch0, ch1, ch2)

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb)

    # 获取节点坐标和属性
    x_coords = data.x[:, 0].detach().cpu().numpy()  # 归一化坐标
    y_coords = data.x[:, 1].detach().cpu().numpy()
    areas = data.x[:, 2].detach().cpu().numpy()  # 面积

    # 转换为像素坐标
    img_size = rgb.shape[0]  # 假设正方形
    px = (x_coords + 0.5) * img_size
    py = (y_coords + 0.5) * img_size

    # 绘制节点圆圈
    cmap = plt.cm.hot_r
    norm = Normalize(vmin=0, vmax=1)

    for i in range(len(node_scores)):
        score = node_scores[i]
        radius = max(3, np.sqrt(areas[i] / np.pi) * 10)  # 缩放半径
        color = cmap(score)
        alpha = 0.5 + score * 0.4

        circle = plt.Circle((px[i], py[i]), radius,
                            facecolor=color, alpha=alpha,
                            linewidth=0.5, edgecolor='white')
        ax.add_patch(circle)

    # 添加节点重要性 colorbar（右侧）
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('节点重要性', fontsize=11)

    # 添加语义类别图例（左下角）
    legend_elements = []

    # 地物类别
    for cat_id in sorted(SEMANTIC_COLORS.keys()):
        color = np.array(SEMANTIC_COLORS[cat_id]) / 255.0
        legend_elements.append(
            mpatches.Patch(facecolor=color, label=SEMANTIC_NAMES[cat_id])
        )

    # 道路
    road_color = np.array(ROAD_COLOR) / 255.0
    legend_elements.append(
        mpatches.Patch(facecolor=road_color, alpha=ROAD_ALPHA, label=SEMANTIC_NAMES["road"])
    )

    # 建筑
    for bld_type in sorted(BUILDING_COLORS.keys()):
        color = np.array(BUILDING_COLORS[bld_type]) / 255.0
        legend_elements.append(
            mpatches.Patch(facecolor=color, label=SEMANTIC_NAMES[f"bld_{['low', 'mid', 'high'][bld_type-1]}"])
        )

    ax.legend(handles=legend_elements, loc='lower left', fontsize=7,
              title='语义类别', framealpha=0.85, ncol=2)

    # 标题
    title = f"UST-{ust_label}: {config.UST_NAMES.get(ust_label, 'Unknown')}" if ust_label is not None else graph_key
    ax.set_title(f"{title} | {graph_key}", fontsize=14)
    ax.axis('off')

    # 保存
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"节点重要性图已保存: {output_path}")


def _plot_node_importance_simple(
    graph_key: str,
    node_scores: np.ndarray,
    data,
    output_path: str,
    ust_label: int = None,
    figsize: Tuple[int, int] = (10, 10)
):
    """简化版节点重要性图（无背景）"""
    fig, ax = plt.subplots(figsize=figsize)

    # 获取节点坐标
    x_coords = data.x[:, 0].cpu().numpy()
    y_coords = data.x[:, 1].cpu().numpy()
    areas = data.x[:, 2].cpu().numpy()

    # 绘制节点
    cmap = plt.cm.hot_r
    scatter = ax.scatter(x_coords, y_coords,
                        c=node_scores, cmap=cmap,
                        s=areas * 100, alpha=0.7,
                        edgecolors='white', linewidth=0.5)

    plt.colorbar(scatter, ax=ax, label='Node importance')

    title = f"UST-{ust_label}" if ust_label is not None else graph_key
    ax.set_title(f"{title} | {graph_key}", fontsize=14)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_node_category_importance_heatmap(
    matrix_df: pd.DataFrame,
    output_path: str,
    figsize: Tuple[int, int] = (20, 8)
):
    """
    节点类别 × UST 类别 热力矩阵图。

    Args:
        matrix_df: cross_ust_node_importance_matrix() 输出
        output_path: 输出路径
        figsize: 图片尺寸
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制热力图
    sns.heatmap(matrix_df, annot=True, fmt='.2f', cmap='YlOrRd',
                ax=ax, linewidths=0.5, cbar_kws={'label': 'Mean importance'})

    ax.set_xlabel('UST Category', fontsize=12)
    ax.set_ylabel('Node Category', fontsize=12)
    ax.set_title('Node Category × UST Importance Matrix', fontsize=14)

    # 旋转 x 轴标签
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"节点类别热力矩阵已保存: {output_path}")


def plot_score_boxplot_per_ust(
    summary_df: pd.DataFrame,
    output_path: str,
    n_cols: int = 6
):
    """
    分面箱线图：每类 UST 一张子图。

    Args:
        summary_df: 汇总 DataFrame
        output_path: 输出路径
        n_cols: 列数
    """
    num_ust = summary_df['ust_label'].nunique()
    n_rows = int(np.ceil(num_ust / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()

    for idx, ust_label in enumerate(sorted(summary_df['ust_label'].unique())):
        ax = axes[idx]

        subset = summary_df[summary_df['ust_label'] == ust_label]

        # 箱线图
        subset.boxplot(column='score', by='node_cat', ax=ax)

        ax.set_title(f"UST-{ust_label}: {config.UST_NAMES.get(ust_label, '')}", fontsize=10)
        ax.set_xlabel('Node category')
        ax.set_ylabel('Importance')
        ax.set_xticklabels([config.NODE_CATEGORY_NAMES.get(int(t.get_text()), t.get_text())
                           for t in ax.get_xticklabels()], rotation=45, ha='right', fontsize=7)

    # 隐藏多余子图
    for idx in range(num_ust, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Node Importance Distribution by UST', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"箱线图已保存: {output_path}")


def plot_score_boxplot_per_ust(
    summary_df: pd.DataFrame,
    output_path: str,
    n_cols: int = 6
):
    """
    分面箱线图：每类 UST 一张子图。

    Args:
        summary_df: 汇总 DataFrame
        output_path: 输出路径
        n_cols: 列数
    """
    num_ust = summary_df['ust_label'].nunique()
    n_rows = int(np.ceil(num_ust / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()

    for idx, ust_label in enumerate(sorted(summary_df['ust_label'].unique())):
        ax = axes[idx]

        subset = summary_df[summary_df['ust_label'] == ust_label].copy()
        subset['node_cat_name'] = subset['node_cat'].map(
            lambda x: config.NODE_CATEGORY_NAMES.get(x, f"cat_{x}")
        )

        # 箱线图
        sns.boxplot(data=subset, x='node_cat_name', y='score', ax=ax, palette='Set3')

        ax.set_title(f"UST-{ust_label}: {config.UST_NAMES.get(ust_label, '')}", fontsize=10)
        ax.set_xlabel('Node category')
        ax.set_ylabel('Importance')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)

    # 隐藏多余子图
    for idx in range(num_ust, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Node Importance Distribution by UST', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"箱线图已保存: {output_path}")


if __name__ == "__main__":
    print("节点可视化模块加载成功")