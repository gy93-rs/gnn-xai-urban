"""
方向一入口脚本：节点级归因分析
运行命令：
  python run_dir1.py --method gradcam --batch_size 32
  python run_dir1.py --method gnnexplainer --sample_per_ust 5
  python run_dir1.py --viz_only --scores_path outputs/results/node_importance_scores.pkl
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
import torch

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 添加原始 MolCLR-Urban 项目路径（包含 models/ 和 dataset/）
MOLCLR_ROOT = "/media/gy/ssd/shanghai_exp/MolCLR-Urban_alldata1127_512_cls"
if MOLCLR_ROOT not in sys.path:
    sys.path.insert(0, MOLCLR_ROOT)

import xai_config as config
from analysis.node_attribution import (
    load_finetune_model,
    compute_node_scores_batch,
    NodeAttributionAnalyzer,
    enrich_data_object
)
from visualization.node_viz import (
    plot_node_importance_map,
    plot_node_category_importance_heatmap,
    plot_score_boxplot_per_ust
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


def run_gradcam_analysis(args):
    """运行 GradCAM 节点归因分析"""
    logger.info("="*60)
    logger.info("方向一：节点级归因分析 (GradCAM)")
    logger.info("="*60)

    start_time = time.time()

    # 创建输出目录
    os.makedirs(config.OUTPUT_CONFIG["results"], exist_ok=True)
    os.makedirs(config.OUTPUT_CONFIG["node_maps"], exist_ok=True)
    os.makedirs(config.OUTPUT_CONFIG["node_dist"], exist_ok=True)

    # 加载数据集
    logger.info("加载数据集...")
    dataset = MolTestDataset(
        data_path=config.DATA_CONFIG["npz_dir"],
        csv_file=config.DATA_CONFIG["label_csv"],
        target=None,
        task='classification'
    )
    logger.info(f"数据集大小: {len(dataset)}")

    # 加载模型
    logger.info("加载模型...")
    device = args.device if torch.cuda.is_available() else "cpu"
    model = load_finetune_model(device)

    # 计算节点重要性分数
    scores_path = os.path.join(config.OUTPUT_CONFIG["results"], "node_importance_scores.pkl")

    if args.viz_only and os.path.exists(scores_path):
        logger.info("跳过计算，加载已有结果...")
        import pickle
        with open(scores_path, 'rb') as f:
            results = pickle.load(f)
    else:
        logger.info("计算节点重要性分数...")
        results = compute_node_scores_batch(
            dataset, model,
            batch_size=args.batch_size,
            device=device,
            save_path=scores_path,
            npz_dir=config.DATA_CONFIG["npz_dir"]
        )

    logger.info(f"成功计算 {len(results)} 个样本的节点分数")

    # 统计分析
    logger.info("运行统计分析...")
    analyzer = NodeAttributionAnalyzer(results, num_classes=config.MODEL_CONFIG["num_classes"])

    # 导出汇总 CSV
    summary_csv_path = os.path.join(config.OUTPUT_CONFIG["results"], "dir1_summary_stats.csv")
    analyzer.export_summary_csv(summary_csv_path)

    # 可视化1：节点类别 × UST 热力矩阵
    logger.info("生成热力矩阵...")
    matrix_df = analyzer.cross_ust_node_importance_matrix()
    heatmap_path = os.path.join(config.OUTPUT_CONFIG["figures"], "node_category_ust_heatmap.png")
    plot_node_category_importance_heatmap(matrix_df, heatmap_path)

    # 可视化2：节点重要性地图（每类 UST 抽 3 个样本）
    logger.info("生成节点重要性地图...")
    viz_per_ust = args.viz_per_ust

    # 按 UST 分组
    ust_samples = {}
    for graph_key, data in results.items():
        if data is None:
            continue
        ust_label = data['y']
        if ust_label not in ust_samples:
            ust_samples[ust_label] = []
        ust_samples[ust_label].append(graph_key)

    # 抽样并生成地图
    generated_maps = 0
    for ust_label, graph_keys in ust_samples.items():
        # 随机抽样
        sample_keys = random.sample(graph_keys, min(viz_per_ust, len(graph_keys)))

        for graph_key in sample_keys:
            try:
                # 获取原始数据
                idx = None
                for i, (data, fname) in enumerate(dataset):
                    if fname.replace('.npz', '') == graph_key:
                        idx = i
                        break

                if idx is None:
                    continue

                data, filename = dataset[idx]
                data = enrich_data_object(data, filename, npz_dir=config.DATA_CONFIG["npz_dir"])

                # 生成地图
                output_path = os.path.join(
                    config.OUTPUT_CONFIG["node_maps"],
                    f"UST{ust_label}_{graph_key}.png"
                )

                plot_node_importance_map(
                    graph_key=graph_key,
                    node_scores=results[graph_key]['node_scores'],
                    data=data,
                    tif_dir=config.DATA_CONFIG["tif_dir"],
                    output_path=output_path,
                    ust_label=ust_label
                )
                generated_maps += 1

            except FileNotFoundError as e:
                logger.warning(f"TIF 文件缺失: {graph_key}")
            except Exception as e:
                logger.error(f"生成地图失败 {graph_key}: {e}")

    logger.info(f"生成了 {generated_maps} 张节点重要性地图")

    # 可视化3：箱线图
    logger.info("生成箱线图...")
    boxplot_path = os.path.join(config.OUTPUT_CONFIG["node_dist"], "score_boxplot_per_ust.png")
    plot_score_boxplot_per_ust(analyzer.summary_df, boxplot_path)

    # 执行摘要
    elapsed = time.time() - start_time
    logger.info("="*60)
    logger.info("执行摘要")
    logger.info("="*60)
    logger.info(f"总耗时: {elapsed/60:.1f} 分钟")
    logger.info(f"样本数: {len(results)}")
    logger.info(f"输出目录: {config.OUTPUT_CONFIG['root']}")
    logger.info(f"输出文件:")
    logger.info(f"  - {scores_path}")
    logger.info(f"  - {summary_csv_path}")
    logger.info(f"  - {heatmap_path}")
    logger.info(f"  - {boxplot_path}")
    logger.info(f"  - {generated_maps} 张节点地图")

    return results


def main():
    parser = argparse.ArgumentParser(description="方向一：节点级归因分析")
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gnnexplainer'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--viz_only', action='store_true',
                        help='仅生成可视化，跳过计算')
    parser.add_argument('--scores_path', type=str, default=None,
                        help='已有节点分数路径')
    parser.add_argument('--viz_per_ust', type=int, default=3,
                        help='每类 UST 可视化样本数')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 创建日志目录
    os.makedirs(config.OUTPUT_CONFIG["root"], exist_ok=True)
    log_path = os.path.join(config.OUTPUT_CONFIG["root"], "run_dir1.log")
    setup_logging(log_path)

    run_gradcam_analysis(args)


if __name__ == "__main__":
    main()