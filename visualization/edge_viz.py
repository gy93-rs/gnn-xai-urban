"""
边级归因可视化模块
生成 UST×边类型热力矩阵、边重要性网络图等
支持阈值过滤和双底图模式
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
    edge_threshold: float = None,
    node_threshold: float = None,
    figsize: Tuple[int, int] = (12, 12)
):
    """
    networkx 空间图可视化，显示边重要性（支持阈值过滤）。

    Args:
        graph_key: 图标识
        data: PyG Data 对象
        edge_scores: [E] 边重要性分数
        edge_types: [E] 边类型字符串列表
        output_path: 输出路径
        ust_label: UST 类别标签
        edge_threshold: 边重要性阈值，低于此值不显示（默认从config读取）
        node_threshold: 节点重要性阈值，如果提供则只显示高重要性节点
        figsize: 图片尺寸

    Returns:
        生成的图片路径列表（语义分割底图 + 原始遥感图）
    """
    # 获取阈值配置
    if edge_threshold is None:
        edge_threshold = config.VIZ_CONFIG.get("edge_score_threshold", 0.3)

    generated_paths = []

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

    # 阈值过滤边
    high_edge_mask = edge_scores >= edge_threshold
    n_high_edges = high_edge_mask.sum()
    n_total_edges = len(edge_scores)

    logger.info(f"边过滤: {n_high_edges}/{n_total_edges} 条边重要性 >= {edge_threshold:.2f}")

    # 添加高重要性边
    for idx in range(len(edge_scores)):
        if not high_edge_mask[idx]:
            continue

        src, dst = edge_index[0, idx], edge_index[1, idx]
        G.add_edge(src, dst,
                   weight=edge_scores[idx],
                   edge_type=edge_types[idx])

    # 找出参与边的节点（只显示有边的节点）
    nodes_with_edges = set()
    for u, v in G.edges():
        nodes_with_edges.add(u)
        nodes_with_edges.add(v)

    # 绘图
    fig, ax = plt.subplots(figsize=figsize)

    # 节点颜色（按类别）
    node_colors = []
    for node in nodes_with_edges:
        cat = G.nodes[node]['category']
        color = config.NODE_CATEGORY_COLORS.get(cat, '#808080')
        node_colors.append(color)

    # 节点大小（按面积，范围 [100, 600]）
    node_sizes = []
    areas = [G.nodes[node]['area'] for node in nodes_with_edges]
    if len(areas) > 0:
        area_min, area_max = np.min(areas), np.max(areas)
        for node in nodes_with_edges:
            area = G.nodes[node]['area']
            if area_max - area_min > 1e-8:
                size = 100 + (area - area_min) / (area_max - area_min) * 500
            else:
                size = 350
            node_sizes.append(size)
    else:
        node_sizes = [200] * len(nodes_with_edges)

    # 边宽度（按重要性）
    edge_widths = []
    edge_colors = []
    for u, v in G.edges():
        score = G.edges[u, v]['weight']
        # 更宽的边表示更高重要性
        edge_widths.append(2 + score * 8)
        edge_colors.append(score)

    # 绘制边
    edges = G.edges()
    if len(edges) > 0:
        edge_norm = Normalize(vmin=edge_threshold, vmax=1)
        edge_cmap = plt.cm.YlOrRd

        nx.draw_networkx_edges(
            G, node_positions, ax=ax,
            edgelist=edges,
            width=edge_widths,
            edge_color=edge_colors,
            edge_cmap=edge_cmap,
            edge_vmin=edge_threshold, edge_vmax=1,
            alpha=0.8
        )

    # 绘制节点
    if len(nodes_with_edges) > 0:
        node_positions_filtered = {n: node_positions[n] for n in nodes_with_edges}
        nx.draw_networkx_nodes(
            G, node_positions_filtered, ax=ax,
            nodelist=list(nodes_with_edges),
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.85,
            edgecolors='black',
            linewidths=1.5
        )

    # 图例：节点类别
    legend_patches = []
    unique_cats = set([G.nodes[n]['category'] for n in nodes_with_edges])
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
        sm = ScalarMappable(cmap=plt.cm.YlOrRd, norm=edge_norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label(f'Edge Importance (threshold={edge_threshold:.2f})', fontsize=10)

    # 标题
    ust_name = config.UST_NAMES.get(ust_label, 'Unknown') if ust_label is not None else ''
    title = f"UST-{ust_label}: {ust_name} | {graph_key}" if ust_label is not None else graph_key
    ax.set_title(f"{title}\n高重要性边: {n_high_edges}/{n_total_edges}", fontsize=14)

    ax.axis('off')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    generated_paths.append(output_path)
    logger.info(f"边重要性网络图已保存: {output_path}")

    return generated_paths


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    print("边可视化模块加载成功！")