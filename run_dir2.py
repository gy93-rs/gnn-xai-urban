"""
方向二入口脚本：边级归因分析
运行命令：
  python run_dir2.py --node_scores outputs/results/node_importance_scores.pkl
  python run_dir2.py --node_scores outputs/results/node_importance_scores.pkl --aggregation product
"""

import os
import sys
import argparse
import logging
import time
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import pickle

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import xai_config as config
from analysis.edge_attribution import (
    EdgeAttributionAnalyzer,
    annotate_edge_types,
    compute_edge_importance
)
from visualization.edge_viz import (
    plot_edge_type_heatmap,
    plot_edge_importance_network
)

# 数据集
from dataset.dataset_finetune import MolTestDataset

# 设置随机种子
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 日志配置
def setup_logging(log_path: str):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path)
        ]
    )

logger = logging.getLogger(__name__)


def run_edge_analysis(args):
    """运行边级归因分析"""
    logger.info("="*60)
    logger.info("方向二：边级归因分析")
    logger.info("="*60)

    start_time = time.time()

    # 创建输出目录
    os.makedirs(config.OUTPUT_CONFIG["results"], exist_ok=True)
    os.makedirs(config.OUTPUT_CONFIG["edge_maps"], exist_ok=True)

    # 加载节点分数结果
    logger.info(f"加载节点分数: {args.node_scores}")
    with open(args.node_scores, 'rb') as f:
        node_results = pickle.load(f)
    logger.info(f"加载了 {len(node_results)} 个样本的节点分数")

    # 加载数据集
    logger.info("加载数据集...")
    dataset = MolTestDataset(
        data_path=config.DATA_CONFIG["npz_dir"],
        csv_file=config.DATA_CONFIG["label_csv"],
        target=None,
        task='classification'
    )
    logger.info(f"数据集大小: {len(dataset)}")

    # 计算边重要性
    logger.info("计算边重要性...")
    edge_results = {}
    all_edge_type_scores = []

    for idx in range(len(dataset)):
        try:
            data, filename = dataset[idx]
            graph_key = filename.replace('.npz', '')

            if graph_key not in node_results:
                continue

            node_scores = node_results[graph_key]['node_scores']
            ust_label = node_results[graph_key]['y']

            # 标注边类型（需要先添加 node_cat 属性）
            # 数据集加载器的one_hot编码有bug，直接从原始npz读取col3作为节点类别
            if not hasattr(data, 'node_cat'):
                npz_path = os.path.join(config.DATA_CONFIG["npz_dir"], filename)
                raw_data = np.load(npz_path)
                raw_col3 = raw_data['array1'][:, 3].astype(np.int64)
                data.node_cat = torch.tensor(raw_col3, dtype=torch.long)
            edge_types = annotate_edge_types(data)

            # 计算边重要性
            edge_importance = compute_edge_importance(
                data.edge_index,
                node_scores,
                aggregation=args.aggregation
            )

            edge_results[graph_key] = {
                'edge_scores': edge_importance,
                'edge_types': edge_types,
                'ust_label': ust_label
            }

            # 收集边类型统计
            for i, etype in enumerate(edge_types):
                all_edge_type_scores.append({
                    'ust_label': ust_label,
                    'edge_type': etype,
                    'score': edge_importance[i]
                })

        except Exception as e:
            logger.error(f"处理样本 {idx} 失败: {e}")

    logger.info(f"成功计算 {len(edge_results)} 个样本的边分数")

    # 构建统计 DataFrame
    edge_df = pd.DataFrame(all_edge_type_scores)

    # 分析
    logger.info("运行统计分析...")
    analyzer = EdgeAttributionAnalyzer(edge_df, num_classes=config.MODEL_CONFIG["num_classes"])

    # 导出边类型矩阵
    matrix_df = analyzer.compute_edge_type_matrix()
    matrix_csv_path = os.path.join(config.OUTPUT_CONFIG["results"], "dir2_edge_type_matrix.csv")
    matrix_df.to_csv(matrix_csv_path)
    logger.info(f"边类型矩阵已保存: {matrix_csv_path}")

    # 导出 top-k 边类型
    top_k_df = analyzer.top_edge_types_per_ust(top_k=5)
    top_k_csv_path = os.path.join(config.OUTPUT_CONFIG["results"], "dir2_top_edge_types.csv")
    top_k_df.to_csv(top_k_csv_path, index=False)
    logger.info(f"Top-K 边类型已保存: {top_k_csv_path}")

    # 可视化1：热力矩阵
    logger.info("生成边类型热力矩阵...")
    heatmap_path = os.path.join(config.OUTPUT_CONFIG["figures"], "dir2_edge_type_heatmap.png")
    plot_edge_type_heatmap(matrix_df, heatmap_path)

    # 可视化2：网络图（每类 UST 抽 1 个样本）
    logger.info("生成边重要性网络图...")

    # 按 UST 分组抽样
    ust_samples = {}
    for graph_key, data in edge_results.items():
        ust_label = data['ust_label']
        if ust_label not in ust_samples:
            ust_samples[ust_label] = []
        ust_samples[ust_label].append(graph_key)

    generated_networks = 0
    for ust_label, graph_keys in ust_samples.items():
        # 随机抽 1 个
        sample_key = random.choice(graph_keys)

        try:
            # 获取原始数据
            idx = None
            for i, (data, fname) in enumerate(dataset):
                if fname.replace('.npz', '') == sample_key:
                    idx = i
                    break

            if idx is None:
                continue

            data, filename = dataset[idx]

            # 添加 node_cat 属性
            if not hasattr(data, 'node_cat'):
                data.node_cat = (data.x[:, 3] - 1).long()

            output_path = os.path.join(
                config.OUTPUT_CONFIG["edge_maps"],
                f"UST{ust_label}_{sample_key}_network.png"
            )

            plot_edge_importance_network(
                graph_key=sample_key,
                data=data,
                edge_scores=edge_results[sample_key]['edge_scores'],
                edge_types=edge_results[sample_key]['edge_types'],
                output_path=output_path,
                ust_label=ust_label,
                top_k_edges=50  # 仅显示最重要的 50 条边
            )
            generated_networks += 1

        except Exception as e:
            logger.error(f"生成网络图失败 {sample_key}: {e}")

    logger.info(f"生成了 {generated_networks} 张边重要性网络图")

    # 执行摘要
    elapsed = time.time() - start_time
    logger.info("="*60)
    logger.info("执行摘要")
    logger.info("="*60)
    logger.info(f"总耗时: {elapsed/60:.1f} 分钟")
    logger.info(f"样本数: {len(edge_results)}")
    logger.info(f"边类型数: {len(matrix_df.columns)}")
    logger.info(f"输出目录: {config.OUTPUT_CONFIG['root']}")
    logger.info(f"输出文件:")
    logger.info(f"  - {matrix_csv_path}")
    logger.info(f"  - {top_k_csv_path}")
    logger.info(f"  - {heatmap_path}")
    logger.info(f"  - {generated_networks} 张网络图")

    return edge_results


def main():
    parser = argparse.ArgumentParser(description="方向二：边级归因分析")
    parser.add_argument('--node_scores', type=str, required=True,
                        help='节点分数结果路径 (node_importance_scores.pkl)')
    parser.add_argument('--aggregation', type=str, default='mean',
                        choices=['mean', 'max', 'product'],
                        help='边重要性聚合方式')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 创建日志目录
    os.makedirs(config.OUTPUT_CONFIG["root"], exist_ok=True)
    log_path = os.path.join(config.OUTPUT_CONFIG["root"], "run_dir2.log")
    setup_logging(log_path)

    run_edge_analysis(args)


if __name__ == "__main__":
    main()