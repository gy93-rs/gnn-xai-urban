"""
边级归因分析模块
识别对 UST 分类影响最大的空间邻接关系类型
"""

import os
import sys
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import xai_config as config

logger = logging.getLogger(__name__)


def get_edge_type(src_cat: int, dst_cat: int) -> str:
    """
    生成无向边类型字符串，按字母序排序避免 (A-B)/(B-A) 重复。

    Args:
        src_cat: 源节点类别
        dst_cat: 目标节点类别

    Returns:
        边类型字符串，如 "low_bld-road"
    """
    a = config.NODE_CATEGORY_NAMES.get(src_cat, f"cat_{src_cat}")
    b = config.NODE_CATEGORY_NAMES.get(dst_cat, f"cat_{dst_cat}")
    return "-".join(sorted([a, b]))


def annotate_edge_types(data: Data) -> List[str]:
    """
    为 data 中所有边标注类型。

    Args:
        data: PyG Data 对象，需要有 node_cat 和 edge_index

    Returns:
        边类型列表，长度为 E
    """
    edge_index = data.edge_index
    node_cat = data.node_cat

    edge_types = []
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        src_cat = node_cat[src].item()
        dst_cat = node_cat[dst].item()
        edge_types.append(get_edge_type(src_cat, dst_cat))

    return edge_types


def compute_edge_importance(
    edge_index: torch.Tensor,
    node_scores: np.ndarray,
    aggregation: str = "mean"
) -> np.ndarray:
    """
    边重要性 = aggregation(src节点分数, dst节点分数)

    Args:
        edge_index: [2, E] 边索引
        node_scores: [N] 节点重要性分数
        aggregation: 聚合方式 "mean" | "max" | "product"

    Returns:
        [E] 边重要性分数，归一化到 [0, 1]
    """
    src_scores = node_scores[edge_index[0].cpu().numpy()]
    dst_scores = node_scores[edge_index[1].cpu().numpy()]

    if aggregation == "mean":
        edge_scores = (src_scores + dst_scores) / 2
    elif aggregation == "max":
        edge_scores = np.maximum(src_scores, dst_scores)
    elif aggregation == "product":
        edge_scores = src_scores * dst_scores
    else:
        raise ValueError(f"未知的聚合方式: {aggregation}")

    # 归一化到 [0, 1]
    if len(edge_scores) == 0:
        return edge_scores  # 空数组直接返回

    min_val = edge_scores.min()
    max_val = edge_scores.max()

    if max_val - min_val > 1e-8:
        edge_scores = (edge_scores - min_val) / (max_val - min_val)
    else:
        edge_scores = np.zeros_like(edge_scores)

    return edge_scores


class EdgeAttributionAnalyzer:
    """边归因统计分析类"""

    def __init__(self, edge_df: pd.DataFrame = None, results: Dict[str, dict] = None, num_classes: int = 17):
        """
        Args:
            edge_df: 边分数统计 DataFrame（包含 ust_label, edge_type, score 列）
            results: 节点分数结果（来自方向一）- 可选，用于 analyze_edges
            num_classes: UST 类别数
        """
        self.results = results
        self.edge_df = edge_df
        self.num_classes = num_classes

        # 存储边分析结果
        self.edge_results = {}
        self.edge_type_matrix = None

    def analyze_edges(self, dataset) -> Dict[str, dict]:
        """
        对所有样本进行边归因分析。

        Args:
            dataset: MolTestDataset 实例

        Returns:
            边归因结果字典
        """
        from analysis.node_attribution import enrich_data_object

        for idx in range(len(dataset)):
            try:
                # 获取数据
                sample = dataset[idx]
                if isinstance(sample, tuple):
                    data, filename = sample
                else:
                    data = sample
                    filename = None

                graph_key = filename.replace('.npz', '') if filename else f"sample_{idx}"

                # 跳过没有节点分数的样本
                if graph_key not in self.results:
                    continue

                node_result = self.results[graph_key]
                if node_result is None:
                    continue

                # 富化数据
                data = enrich_data_object(data, filename)

                # 标注边类型
                edge_types = annotate_edge_types(data)

                # 计算边重要性
                edge_scores = compute_edge_importance(
                    data.edge_index,
                    node_result['node_scores'],
                    aggregation="mean"
                )

                # 存储
                self.edge_results[graph_key] = {
                    'edge_types': edge_types,
                    'edge_scores': edge_scores,
                    'ust_label': node_result['y'],
                    'edge_index': data.edge_index.cpu().numpy()
                }

            except Exception as e:
                logger.error(f"处理样本 {idx} 边归因失败: {e}")

        logger.info(f"边归因分析完成: {len(self.edge_results)} 个样本")
        return self.edge_results

    def compute_edge_type_matrix(self) -> pd.DataFrame:
        """
        构建 UST × 边类型 重要性矩阵。

        Returns:
            DataFrame: index=UST类别, columns=边类型
        """
        # 如果有 edge_df，直接从中提取
        if self.edge_df is not None and len(self.edge_df) > 0:
            # 收集所有边类型
            all_edge_types = sorted(self.edge_df['edge_type'].unique())

            # 构建矩阵
            ust_edge_scores = defaultdict(lambda: defaultdict(list))

            for _, row in self.edge_df.iterrows():
                ust_label = row['ust_label']
                etype = row['edge_type']
                score = row['score']
                ust_edge_scores[ust_label][etype].append(score)

        else:
            # 从 edge_results 提取
            all_edge_types = set()
            for result in self.edge_results.values():
                if result is not None:
                    all_edge_types.update(result['edge_types'])

            all_edge_types = sorted(all_edge_types)

            ust_edge_scores = defaultdict(lambda: defaultdict(list))

            for graph_key, result in self.edge_results.items():
                if result is None:
                    continue

                ust_label = result['ust_label']
                edge_types = result['edge_types']
                edge_scores = result['edge_scores']

                for etype, score in zip(edge_types, edge_scores):
                    ust_edge_scores[ust_label][etype].append(score)

        # 计算平均分数
        matrix_data = []
        for ust_label in range(self.num_classes):
            row = {}
            for etype in all_edge_types:
                scores = ust_edge_scores[ust_label].get(etype, [])
                if scores:
                    row[etype] = np.mean(scores)
                else:
                    row[etype] = np.nan
            matrix_data.append(row)

        # 创建 DataFrame
        df = pd.DataFrame(matrix_data)

        # 设置索引和列名（格式：UST-0 到 UST-16）
        df.index = [f"UST-{i}" for i in range(self.num_classes)]

        # 添加全局平均值行
        global_means = {}
        for etype in all_edge_types:
            all_scores = []
            for ust_label in range(self.num_classes):
                scores = ust_edge_scores[ust_label].get(etype, [])
                all_scores.extend(scores)
            global_means[etype] = np.mean(all_scores) if all_scores else np.nan

        df.loc['global_mean'] = global_means

        self.edge_type_matrix = df
        return df

    def top_edge_types_per_ust(self, top_k: int = 5) -> pd.DataFrame:
        """
        每类 UST 中重要性最高的 top_k 边类型。

        Args:
            top_k: 返回的边类型数量

        Returns:
            排序后的 DataFrame
        """
        if self.edge_type_matrix is None:
            self.compute_edge_type_matrix()

        records = []
        for ust_label in range(self.num_classes):
            row = self.edge_type_matrix.iloc[ust_label]
            # 排除 NaN，取 top_k
            valid = row.dropna().sort_values(ascending=False).head(top_k)

            for rank, (etype, score) in enumerate(valid.items(), 1):
                records.append({
                    'ust_label': ust_label,
                    'ust_name': config.UST_NAMES.get(ust_label, f"UST-{ust_label}"),
                    'rank': rank,
                    'edge_type': etype,
                    'mean_score': score
                })

        return pd.DataFrame(records)

    def export_edge_matrix_csv(self, output_path: str):
        """
        保存边类型矩阵为 CSV。

        Args:
            output_path: 输出路径
        """
        if self.edge_type_matrix is None:
            self.compute_edge_type_matrix()

        self.edge_type_matrix.to_csv(output_path)
        logger.info(f"边类型矩阵已保存: {output_path}")


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    # 测试边类型生成
    print(get_edge_type(7, 6))  # 应输出 "low_bld-road"
    print(get_edge_type(4, 7))  # 应输出 "low_bld-tree"