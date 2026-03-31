"""
边级归因可视化模块
生成 UST×边类型热力矩阵、边重要性网络图等
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import xai_config as config

logger = logging.getLogger(__name__)


def plot_edge_type_heatmap(
    matrix_df: pd.DataFrame,
    output_path: str,
    figsize: Tuple[int, int] = None
):
    """
    UST × 边类型 热力矩阵图（方向二核心图）。

    Args:
        matrix_df: compute_edge_type_matrix() 输出
        output_path: 输出路径
        figsize: 图片尺寸
    """
    if figsize is None:
        n_cols = len(matrix_df.columns)
        width = max(16, n_cols * 0.9)
        figsize = (width, 10)

    fig, ax = plt.subplots(figsize=figsize)

    # 创建掩码（NaN 显示为灰色）
    mask = matrix_df.isna()

    # 热力图（每格标注数值，保留2位小数，字体8pt）
    sns.heatmap(
        matrix_df,
        ax=ax,
        cmap='YlOrRd',
        mask=mask,
        cbar_kws={'label': 'Edge Importance'},
        linewidths=0.5,
        linecolor='white',
        annot=True,
        fmt='.2f',
        annot_kws={'fontsize': 8}
    )

    # NaN 显示浅灰色背景
    ax.set_facecolor('#f0f0f0')

    # 标签
    ax.set_xlabel('Edge Type', fontsize=12)
    ax.set_ylabel('UST Class', fontsize=12)

    # x 轴标签旋转
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"边类型热力矩阵已保存: {output_path}")


def plot_edge_importance_network(
    graph_key: str,
    data,
    edge_scores: np.ndarray,
    edge_types: List[str],
    output_path: str,
    ust_label: int = None,
    top_k_edges: int = None,
    figsize: Tuple[int, int] = (12, 12)
):
    """
    networkx 空间图可视化，显示边重要性。

    Args:
        graph_key: 图标识
        data: PyG Data 对象
        edge_scores: [E] 边重要性分数
        edge_types: [E] 边类型字符串列表
        output_path: 输出路径
        ust_label: UST 类别标签
        top_k_edges: 仅显示重要性最高的 K 条边（None 则显示全部）
        figsize: 图片尺寸
    """
    # 构建 networkx 图
    G = nx.Graph()

    # 节点位置和属性
    num_nodes = data.x.shape[0]
    node_positions = {}

    # 从节点特征提取位置和属性
    if hasattr(data.x, 'cpu'):
        x_coords = data.x[:, 0].cpu().numpy()
        y_coords = data.x[:, 1].cpu().numpy()
        node_areas = data.x[:, 2].cpu().numpy()
        node_cats = data.node_cat.cpu().numpy() if hasattr(data, 'node_cat') else np.zeros(num_nodes)
    else:
        x_coords = data.x[:, 0].numpy()
        y_coords = data.x[:, 1].numpy()
        node_areas = data.x[:, 2].numpy()
        node_cats = data.node_cat.numpy() if hasattr(data, 'node_cat') else np.zeros(num_nodes)

    # 坐标转换：x 直接使用，y 翻转（1 - y_coords）
    for i in range(num_nodes):
        px = x_coords[i]
        py = 1 - y_coords[i]  # y 轴翻转
        node_positions[i] = (px, py)

        G.add_node(i,
                   area=node_areas[i],
                   category=node_cats[i])

    # 边
    edge_index = data.edge_index
    if hasattr(edge_index, 'cpu'):
        edge_index = edge_index.cpu().numpy()
    else:
        edge_index = edge_index.numpy()

    # 选择要显示的边
    if top_k_edges is not None and top_k_edges < len(edge_scores):
        top_indices = np.argsort(edge_scores)[-top_k_edges:]
    else:
        top_indices = range(len(edge_scores))

    # 添加边
    for idx in top_indices:
        src, dst = edge_index[0, idx], edge_index[1, idx]
        G.add_edge(src, dst,
                   weight=edge_scores[idx],
                   edge_type=edge_types[idx])

    # 绘图
    fig, ax = plt.subplots(figsize=figsize)

    # 节点颜色（按类别，使用固定调色板）
    node_colors = []
    for node in G.nodes():
        cat = G.nodes[node]['category']
        # 使用与方向一相同的固定调色板
        color = config.NODE_CATEGORY_COLORS.get(cat, '#808080')
        node_colors.append(color)

    # 节点大小（按面积，范围 [50, 500]）
    node_sizes = []
    areas = [G.nodes[node]['area'] for node in G.nodes()]
    if len(areas) > 0:
        area_min, area_max = np.min(areas), np.max(areas)
        for node in G.nodes():
            area = G.nodes[node]['area']
            # 缩放到 [50, 500]
            if area_max - area_min > 1e-8:
                size = 50 + (area - area_min) / (area_max - area_min) * 450
            else:
                size = 275  # 默认中间值
            node_sizes.append(size)
    else:
        node_sizes = [100] * num_nodes

    # 边宽度（按重要性）
    edge_widths = []
    edge_colors = []
    for u, v in G.edges():
        score = G.edges[u, v]['weight']
        edge_widths.append(score * 5 + 0.3)
        edge_colors.append(score)

    # 绘制边
    edges = G.edges()
    if len(edges) > 0:
        # 边颜色映射
        edge_norm = Normalize(vmin=0, vmax=1)
        edge_cmap = plt.cm.Reds

        nx.draw_networkx_edges(
            G, node_positions, ax=ax,
            edgelist=edges,
            width=edge_widths,
            edge_color=edge_colors,
            edge_cmap=edge_cmap,
            edge_vmin=0, edge_vmax=1
        )

    # 绘制节点
    nx.draw_networkx_nodes(
        G, node_positions, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.8
    )

    # 图例：节点类别（使用固定调色板）
    legend_patches = []
    unique_cats = set(node_cats)
    for cat in sorted(unique_cats):
        cat_name = config.NODE_CATEGORY_NAMES.get(cat, f"cat_{cat}")
        color = config.NODE_CATEGORY_COLORS.get(cat, '#808080')
        patch = mpatches.Patch(
            color=color,
            label=cat_name
        )
        legend_patches.append(patch)

    ax.legend(handles=legend_patches, loc='lower left',
              title='Node Category', fontsize=8)

    # 颜色条：边重要性
    if len(edges) > 0:
        sm = ScalarMappable(cmap=plt.cm.Reds, norm=edge_norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Edge Importance', fontsize=10)

    # 标题
    title = f"UST-{ust_label} | {graph_key}" if ust_label is not None else graph_key
    ax.set_title(title, fontsize=14)

    ax.axis('off')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"边重要性网络图已保存: {output_path}")


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    print("边可视化模块加载成功！")