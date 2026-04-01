"""
节点重要性可视化模块
支持阈值过滤和双底图模式（语义分割图 + 原始遥感图）
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


def find_jpg_path(graph_key: str, ust_label: int, jpg_base_dir: str) -> Optional[str]:
    """
    根据graph_key和ust_label查找对应的jpg原始遥感图路径。

    文件名转换：shanghai_XX_YY -> ShanghaiSrc_XX_YY.jpg

    Args:
        graph_key: 图标识符，如 "shanghai_17_68"
        ust_label: UST类别标签
        jpg_base_dir: jpg文件夹根目录

    Returns:
        jpg文件路径，若找不到则返回None
    """
    # 转换文件名：shanghai_XX_YY -> XX_YY
    parts = graph_key.replace("shanghai_", "").replace("shanghai", "")
    jpg_filename = f"{config.JPG_FILENAME_PREFIX}{parts}.jpg"

    # 根据UST标签找到对应文件夹
    folder_name = config.UST_TO_JPG_FOLDER.get(ust_label)
    if folder_name:
        jpg_path = os.path.join(jpg_base_dir, folder_name, jpg_filename)
        if os.path.exists(jpg_path):
            return jpg_path

    # 降级：遍历所有子文件夹查找
    for subdir in os.listdir(jpg_base_dir):
        jpg_path = os.path.join(jpg_base_dir, subdir, jpg_filename)
        if os.path.exists(jpg_path):
            return jpg_path

    return None


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
    figsize: Tuple[int, int] = (12, 10),
    score_threshold: float = None,
    jpg_dir: str = None,
    dual_mode: bool = True
):
    """
    在底图上叠加节点重要性热力圆（支持阈值过滤和双底图模式）。

    Args:
        graph_key: 图标识符，如 "shanghai_17_68"
        node_scores: [N] 节点重要性分数
        data: Data 对象（含 x, edge_index）
        tif_dir: TIF 文件目录（语义分割图）
        output_path: 输出图片路径（语义分割底图版本）
        ust_label: UST 类别标签
        figsize: 图片尺寸
        score_threshold: 节点重要性阈值，低于此值的节点不显示（默认从config读取）
        jpg_dir: JPG 原始遥感图根目录
        dual_mode: 是否生成双底图版本（语义分割 + 原始遥感图）

    Returns:
        生成的图片路径列表
    """
    # 获取阈值配置
    if score_threshold is None:
        score_threshold = config.VIZ_CONFIG.get("node_score_threshold", 0.5)

    if jpg_dir is None:
        jpg_dir = config.DATA_CONFIG.get("jpg_dir", "")

    generated_paths = []

    try:
        import rasterio
    except ImportError:
        logger.warning("rasterio 未安装，跳过 TIF 背景图")
        _plot_node_importance_simple(
            graph_key, node_scores, data, output_path, ust_label, figsize, score_threshold
        )
        return [output_path]

    tif_path = os.path.join(tif_dir, f"{graph_key}.tif")

    # 读取 TIF 文件
    try:
        with rasterio.open(tif_path) as src:
            ch0 = src.read(1)  # 地物类别
            ch1 = src.read(2)  # 道路掩码
            ch2 = src.read(3)  # 建筑高度类型
    except FileNotFoundError:
        logger.warning(f"TIF 文件不存在: {tif_path}")
        _plot_node_importance_simple(
            graph_key, node_scores, data, output_path, ust_label, figsize, score_threshold
        )
        return [output_path]

    # 渲染语义分割图
    semantic_rgb = _render_semantic_map(ch0, ch1, ch2)
    img_size = semantic_rgb.shape[0]

    # 获取节点坐标和属性
    x_coords = data.x[:, 0].detach().cpu().numpy()
    y_coords = data.x[:, 1].detach().cpu().numpy()
    areas = data.x[:, 2].detach().cpu().numpy()

    # 转换为像素坐标（y轴翻转）
    px = (x_coords + 0.5) * img_size
    py = img_size - (y_coords + 0.5) * img_size  # y轴翻转

    # === 模式1：语义分割底图 ===
    fig1, ax1 = plt.subplots(figsize=figsize)
    ax1.imshow(semantic_rgb)

    # 过滤低重要性节点
    high_importance_mask = node_scores >= score_threshold
    n_high = high_importance_mask.sum()
    n_total = len(node_scores)

    logger.info(f"节点过滤: {n_high}/{n_total} 个节点重要性 >= {score_threshold:.2f}")

    cmap = plt.cm.YlOrRd  # 黄-橙-红渐变，更适合热力图
    norm = Normalize(vmin=score_threshold, vmax=1)

    # 绘制高重要性节点
    for i in range(len(node_scores)):
        if not high_importance_mask[i]:
            continue

        score = node_scores[i]
        radius = max(5, np.sqrt(areas[i] / np.pi) * 15)
        color = cmap(norm(score))
        alpha = 0.6 + (score - score_threshold) / (1 - score_threshold) * 0.3

        circle = plt.Circle((px[i], py[i]), radius,
                            facecolor=color, alpha=alpha,
                            linewidth=1.5, edgecolor='black')
        ax1.add_patch(circle)

    # 添加 colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar1 = plt.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label(f'节点重要性 (阈值={score_threshold:.2f})', fontsize=11)

    # 添加图例
    legend_elements = _build_legend_elements()
    ax1.legend(handles=legend_elements, loc='lower left', fontsize=7,
              title='语义类别', framealpha=0.85, ncol=2)

    title1 = f"UST-{ust_label}: {config.UST_NAMES.get(ust_label, 'Unknown')} [语义分割]" if ust_label is not None else f"{graph_key} [语义分割]"
    ax1.set_title(f"{title1}\n高重要性节点: {n_high}/{n_total}", fontsize=14)
    ax1.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    generated_paths.append(output_path)
    logger.info(f"语义分割底图版已保存: {output_path}")

    # === 模式2：原始遥感图底图 ===
    if dual_mode and jpg_dir:
        jpg_path = find_jpg_path(graph_key, ust_label, jpg_dir)

        if jpg_path:
            # 读取jpg图像
            from PIL import Image
            jpg_img = Image.open(jpg_path)
            jpg_rgb = np.array(jpg_img)

            # 确保图像尺寸匹配（jpg可能需要resize）
            if jpg_rgb.shape[0] != img_size or jpg_rgb.shape[1] != img_size:
                jpg_img = jpg_img.resize((img_size, img_size), Image.LANCZOS)
                jpg_rgb = np.array(jpg_img)

            fig2, ax2 = plt.subplots(figsize=figsize)
            ax2.imshow(jpg_rgb)

            # 绘制高重要性节点（在遥感图上更显眼）
            for i in range(len(node_scores)):
                if not high_importance_mask[i]:
                    continue

                score = node_scores[i]
                radius = max(8, np.sqrt(areas[i] / np.pi) * 20)
                color = cmap(norm(score))
                alpha = 0.7 + (score - score_threshold) / (1 - score_threshold) * 0.25

                # 添加发光效果（多层圆圈）
                # 外层：发光效果
                glow_circle = plt.Circle((px[i], py[i]), radius * 1.5,
                                         facecolor=color, alpha=alpha * 0.3,
                                         linewidth=0)
                ax2.add_patch(glow_circle)

                # 内层：核心节点
                circle = plt.Circle((px[i], py[i]), radius,
                                    facecolor=color, alpha=alpha,
                                    linewidth=2, edgecolor='white')
                ax2.add_patch(circle)

            # 添加 colorbar
            cbar2 = plt.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label(f'节点重要性 (阈值={score_threshold:.2f})', fontsize=11)

            title2 = f"UST-{ust_label}: {config.UST_NAMES.get(ust_label, 'Unknown')} [遥感影像]" if ust_label is not None else f"{graph_key} [遥感影像]"
            ax2.set_title(f"{title2}\n高重要性节点: {n_high}/{n_total}", fontsize=14)
            ax2.axis('off')

            # 生成第二个输出路径
            output_path_jpg = output_path.replace(".png", "_remote_sensing.png")
            plt.tight_layout()
            plt.savefig(output_path_jpg, dpi=150, bbox_inches='tight')
            plt.close()
            generated_paths.append(output_path_jpg)
            logger.info(f"原始遥感图底图版已保存: {output_path_jpg}")
        else:
            logger.warning(f"找不到JPG遥感图: {graph_key}")

    return generated_paths


def _build_legend_elements() -> List[mpatches.Patch]:
    """构建语义类别图例元素"""
    legend_elements = []

    for cat_id in sorted(SEMANTIC_COLORS.keys()):
        color = np.array(SEMANTIC_COLORS[cat_id]) / 255.0
        legend_elements.append(
            mpatches.Patch(facecolor=color, label=SEMANTIC_NAMES[cat_id])
        )

    road_color = np.array(ROAD_COLOR) / 255.0
    legend_elements.append(
        mpatches.Patch(facecolor=road_color, alpha=ROAD_ALPHA, label=SEMANTIC_NAMES["road"])
    )

    for bld_type in sorted(BUILDING_COLORS.keys()):
        color = np.array(BUILDING_COLORS[bld_type]) / 255.0
        legend_elements.append(
            mpatches.Patch(facecolor=color, label=SEMANTIC_NAMES[f"bld_{['low', 'mid', 'high'][bld_type-1]}"])
        )

    return legend_elements


def _plot_node_importance_simple(
    graph_key: str,
    node_scores: np.ndarray,
    data,
    output_path: str,
    ust_label: int = None,
    figsize: Tuple[int, int] = (10, 10),
    score_threshold: float = 0.5
):
    """简化版节点重要性图（无背景，支持阈值过滤）"""
    fig, ax = plt.subplots(figsize=figsize)

    # 获取节点坐标
    x_coords = data.x[:, 0].cpu().numpy()
    y_coords = data.x[:, 1].cpu().numpy()
    areas = data.x[:, 2].cpu().numpy()

    # 过滤低重要性节点
    high_mask = node_scores >= score_threshold
    n_high = high_mask.sum()
    n_total = len(node_scores)

    # 绘制高重要性节点
    cmap = plt.cm.YlOrRd
    norm = Normalize(vmin=score_threshold, vmax=1)

    scatter = ax.scatter(
        x_coords[high_mask], y_coords[high_mask],
        c=node_scores[high_mask], cmap=cmap, norm=norm,
        s=areas[high_mask] * 150, alpha=0.7,
        edgecolors='black', linewidth=1
    )

    plt.colorbar(scatter, ax=ax, label=f'Node importance (threshold={score_threshold:.2f})')

    title = f"UST-{ust_label}" if ust_label is not None else graph_key
    ax.set_title(f"{title} | {graph_key}\nHigh importance nodes: {n_high}/{n_total}", fontsize=14)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"简化版节点图已保存: {output_path}")


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