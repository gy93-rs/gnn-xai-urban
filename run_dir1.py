"""
方向一入口脚本：节点级归因分析
支持 5 种可解释性方法：GradCAM, GNNExplainer, PGExplainer, GraphMASK, GraphLIME

运行命令：
  python run_dir1.py --method gradcam
  python run_dir1.py --method gnnexplainer --epochs 200
  python run_dir1.py --method pgexplainer --epochs 100
  python run_dir1.py --method graphmask --epochs 50
  python run_dir1.py --method graphlime --samples 5000
  python run_dir1.py --viz_only --scores_path outputs/results/node_importance_scores.pkl
"""

import os
import sys
import argparse
import logging
import time
import random
import pickle
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


def get_scores_filename(method: str) -> str:
    """根据方法名返回分数文件名"""
    return f"node_importance_scores_{method}.pkl"


def run_gradcam_analysis(args, dataset, model, device):
    """运行 GradCAM 节点归因分析"""
    from analysis.node_attribution import compute_node_scores_batch

    scores_path = os.path.join(
        config.OUTPUT_CONFIG["results"],
        get_scores_filename("gradcam")
    )

    if args.viz_only and os.path.exists(scores_path):
        logger.info("加载已有 GradCAM 结果...")
        with open(scores_path, 'rb') as f:
            results = pickle.load(f)
    else:
        logger.info("计算 GradCAM 节点重要性分数...")
        results = compute_node_scores_batch(
            dataset, model,
            batch_size=args.batch_size,
            device=device,
            save_path=scores_path,
            npz_dir=config.DATA_CONFIG["npz_dir"]
        )

    return results, scores_path


def run_gnnexplainer_analysis(args, dataset, model, device):
    """运行 GNNExplainer 节点归因分析"""
    from analysis.gnnexplainer_attribution import compute_gnnexplainer_batch

    scores_path = os.path.join(
        config.OUTPUT_CONFIG["results"],
        get_scores_filename("gnnexplainer")
    )

    if args.viz_only and os.path.exists(scores_path):
        logger.info("加载已有 GNNExplainer 结果...")
        with open(scores_path, 'rb') as f:
            results = pickle.load(f)
    else:
        logger.info("计算 GNNExplainer 节点重要性分数...")
        results = compute_gnnexplainer_batch(
            dataset, model,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            save_path=scores_path,
            npz_dir=config.DATA_CONFIG["npz_dir"]
        )

    return results, scores_path


def run_pgexplainer_analysis(args, dataset, model, device):
    """运行 PGExplainer 节点归因分析"""
    from analysis.pgexplainer_attribution import compute_pgexplainer_batch

    scores_path = os.path.join(
        config.OUTPUT_CONFIG["results"],
        get_scores_filename("pgexplainer")
    )

    if args.viz_only and os.path.exists(scores_path):
        logger.info("加载已有 PGExplainer 结果...")
        with open(scores_path, 'rb') as f:
            results = pickle.load(f)
    else:
        logger.info("计算 PGExplainer 节点重要性分数...")
        results = compute_pgexplainer_batch(
            dataset, model,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            device=device,
            save_path=scores_path,
            npz_dir=config.DATA_CONFIG["npz_dir"]
        )

    return results, scores_path


def run_graphmask_analysis(args, dataset, model, device):
    """运行 GraphMASK 节点归因分析"""
    from analysis.graphmask_attribution import compute_graphmask_batch

    scores_path = os.path.join(
        config.OUTPUT_CONFIG["results"],
        get_scores_filename("graphmask")
    )

    if args.viz_only and os.path.exists(scores_path):
        logger.info("加载已有 GraphMASK 结果...")
        with open(scores_path, 'rb') as f:
            results = pickle.load(f)
    else:
        logger.info("计算 GraphMASK 节点重要性分数...")
        results = compute_graphmask_batch(
            dataset, model,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            lambda_sparsity=args.lambda_sparsity,
            device=device,
            save_path=scores_path,
            npz_dir=config.DATA_CONFIG["npz_dir"]
        )

    return results, scores_path


def run_graphlime_analysis(args, dataset, model, device):
    """运行 GraphLIME 节点归因分析"""
    from analysis.graphlime_attribution import compute_graphlime_batch

    scores_path = os.path.join(
        config.OUTPUT_CONFIG["results"],
        get_scores_filename("graphlime")
    )

    if args.viz_only and os.path.exists(scores_path):
        logger.info("加载已有 GraphLIME 结果...")
        with open(scores_path, 'rb') as f:
            results = pickle.load(f)
    else:
        logger.info("计算 GraphLIME 节点重要性分数...")
        results = compute_graphlime_batch(
            dataset, model,
            num_samples=args.samples,
            alpha=args.alpha,
            device=device,
            save_path=scores_path,
            npz_dir=config.DATA_CONFIG["npz_dir"]
        )

    return results, scores_path


def run_analysis(args):
    """运行节点归因分析（统一入口）"""
    logger.info("="*60)
    logger.info(f"方向一：节点级归因分析 ({args.method.upper()})")
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

    # 根据方法选择分析器
    method_runners = {
        'gradcam': run_gradcam_analysis,
        'gnnexplainer': run_gnnexplainer_analysis,
        'pgexplainer': run_pgexplainer_analysis,
        'graphmask': run_graphmask_analysis,
        'graphlime': run_graphlime_analysis,
    }

    if args.method not in method_runners:
        raise ValueError(f"未知方法: {args.method}")

    # 运行分析
    results, scores_path = method_runners[args.method](
        args, dataset, model, device
    )

    logger.info(f"成功计算 {len(results)} 个样本的节点分数")

    # 统计分析
    logger.info("运行统计分析...")
    analyzer = NodeAttributionAnalyzer(results, num_classes=config.MODEL_CONFIG["num_classes"])

    # 导出汇总 CSV
    summary_csv_path = os.path.join(
        config.OUTPUT_CONFIG["results"],
        f"dir1_summary_stats_{args.method}.csv"
    )
    analyzer.export_summary_csv(summary_csv_path)

    # 可视化1：节点类别 × UST 热力矩阵
    logger.info("生成热力矩阵...")
    matrix_df = analyzer.cross_ust_node_importance_matrix()
    heatmap_path = os.path.join(
        config.OUTPUT_CONFIG["figures"],
        f"node_category_ust_heatmap_{args.method}.png"
    )
    plot_node_category_importance_heatmap(matrix_df, heatmap_path)

    # 可视化2：节点重要性地图（每类 UST 抽样）
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

    # 创建方法专属输出目录
    method_map_dir = os.path.join(
        config.OUTPUT_CONFIG["node_maps"],
        args.method
    )
    os.makedirs(method_map_dir, exist_ok=True)

    # 抽样并生成地图
    generated_maps = 0
    for ust_label, graph_keys in ust_samples.items():
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
                    method_map_dir,
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
    method_dist_dir = os.path.join(
        config.OUTPUT_CONFIG["node_dist"],
        args.method
    )
    os.makedirs(method_dist_dir, exist_ok=True)
    boxplot_path = os.path.join(method_dist_dir, "score_boxplot_per_ust.png")
    plot_score_boxplot_per_ust(analyzer.summary_df, boxplot_path)

    # 执行摘要
    elapsed = time.time() - start_time
    logger.info("="*60)
    logger.info("执行摘要")
    logger.info("="*60)
    logger.info(f"方法: {args.method}")
    logger.info(f"总耗时: {elapsed/60:.1f} 分钟")
    logger.info(f"样本数: {len(results)}")
    logger.info(f"输出目录: {config.OUTPUT_CONFIG['root']}")
    logger.info(f"输出文件:")
    logger.info(f"  - {scores_path}")
    logger.info(f"  - {summary_csv_path}")
    logger.info(f"  - {heatmap_path}")
    logger.info(f"  - {boxplot_path}")
    logger.info(f"  - {generated_maps} 张节点地图 ({method_map_dir})")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="方向一：节点级归因分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例命令:
  python run_dir1.py --method gradcam
  python run_dir1.py --method gnnexplainer --epochs 200
  python run_dir1.py --method pgexplainer --epochs 100
  python run_dir1.py --method graphmask --epochs 50
  python run_dir1.py --method graphlime --samples 5000
        """
    )

    # 方法选择
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gnnexplainer', 'pgexplainer',
                                 'graphmask', 'graphlime'],
                        help='可解释性方法 (default: gradcam)')

    # 通用参数
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备 (default: cuda)')
    parser.add_argument('--viz_only', action='store_true',
                        help='仅生成可视化，跳过计算')
    parser.add_argument('--viz_per_ust', type=int, default=3,
                        help='每类 UST 可视化样本数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    # GradCAM 参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='GradCAM 批量大小')

    # GNNExplainer / PGExplainer / GraphMASK 参数
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数 (方法默认值)')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率 (方法默认值)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='隐藏层维度 (PGExplainer/GraphMASK)')

    # GraphMASK 特有参数
    parser.add_argument('--lambda_sparsity', type=float, default=0.1,
                        help='GraphMASK 稀疏性惩罚系数')

    # GraphLIME 特有参数
    parser.add_argument('--samples', type=int, default=5000,
                        help='GraphLIME 扰动样本数')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='GraphLIME Ridge 正则化系数')

    args = parser.parse_args()

    # 设置方法默认参数
    if args.epochs is None:
        default_epochs = {
            'gradcam': 0,
            'gnnexplainer': config.XAI_CONFIG['gnnexplainer_epochs'],
            'pgexplainer': config.XAI_CONFIG['pgexplainer_epochs'],
            'graphmask': config.XAI_CONFIG['graphmask_epochs'],
            'graphlime': 0,
        }
        args.epochs = default_epochs[args.method]

    if args.lr is None:
        default_lr = {
            'gradcam': 0.0,
            'gnnexplainer': config.XAI_CONFIG['gnnexplainer_lr'],
            'pgexplainer': config.XAI_CONFIG['pgexplainer_lr'],
            'graphmask': config.XAI_CONFIG['graphmask_lr'],
            'graphlime': 0.0,
        }
        args.lr = default_lr[args.method]

    # 设置随机种子
    set_seed(args.seed)

    # 创建日志目录
    os.makedirs(config.OUTPUT_CONFIG["root"], exist_ok=True)
    log_path = os.path.join(config.OUTPUT_CONFIG["root"], f"run_dir1_{args.method}.log")
    setup_logging(log_path)

    run_analysis(args)


if __name__ == "__main__":
    main()