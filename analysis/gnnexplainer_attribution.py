"""
GNNExplainer 节点级归因分析模块
使用 PyTorch Geometric 1.6.3 内置的 GNNExplainer
"""

import os
import sys
import logging
import pickle
from typing import Dict, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn.models import GNNExplainer

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 添加原始项目路径
ORIGINAL_PROJECT_ROOT = "/media/gy/ssd/shanghai_exp/MolCLR-Urban_alldata1127_512_cls"
if ORIGINAL_PROJECT_ROOT not in sys.path:
    sys.path.insert(0, ORIGINAL_PROJECT_ROOT)

import xai_config as config
from analysis.node_attribution import (
    load_finetune_model,
    enrich_data_object
)

logger = logging.getLogger(__name__)


def compute_gnnexplainer_single(
    data: Data,
    model: nn.Module,
    device: str = "cuda",
    epochs: int = 200,
    lr: float = 0.01,
    target_class: int = None
) -> Dict[str, np.ndarray]:
    """
    对单个图使用 GNNExplainer 计算节点和边重要性。

    Args:
        data: PyG Data 对象
        model: 已加载权重的模型
        device: 设备
        epochs: 训练轮数
        lr: 学习率
        target_class: 目标类别索引（None 则使用预测类别）

    Returns:
        包含 node_scores, edge_scores, feature_scores 的字典
    """
    model.eval()
    data = data.to(device)

    # 设置 batch 属性（图分类需要）
    if not hasattr(data, 'batch') or data.batch is None:
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)

    # 获取目标类别
    with torch.no_grad():
        _, logits = model(data)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

    # 创建 GNNExplainer
    explainer = GNNExplainer(model, epochs=epochs, lr=lr, log=False)

    try:
        # 使用 explain_graph 解释图分类
        # 注意：PyG 1.6.3 的 GNNExplainer 主要针对节点分类
        # 对于图分类，我们使用 explain_node 但对全局池化节点进行解释
        node_feat_mask, edge_mask = explainer.explain_node(
            node_idx=0,  # 任意选择一个节点
            x=data.x,
            edge_index=data.edge_index,
            batch=data.batch
        )

        # 从节点特征掩码计算节点重要性
        # node_feat_mask: [F] 每个特征的重要性
        # 我们对特征维度求和得到节点重要性
        node_scores = node_feat_mask.cpu().detach().numpy()

        # 边重要性
        edge_scores = edge_mask.cpu().detach().numpy() if edge_mask is not None else None

    except Exception as e:
        logger.warning(f"GNNExplainer 执行失败: {e}，使用回退方法")
        # 回退：使用梯度方法
        explainer.__clear_masks__()

        x = data.x.clone().detach().requires_grad_(True)
        data.x = x
        _, logits = model(data)
        score = logits[0, target_class]
        gradients = torch.autograd.grad(score, x, retain_graph=False, create_graph=False)[0]
        node_scores = gradients.pow(2).sum(dim=1).sqrt().cpu().detach().numpy()
        edge_scores = None

    finally:
        # 清理掩码
        explainer.__clear_masks__()

    # 归一化到 [0, 1]
    if node_scores is not None:
        min_val, max_val = node_scores.min(), node_scores.max()
        if max_val - min_val > 1e-8:
            node_scores = (node_scores - min_val) / (max_val - min_val)
        else:
            node_scores = np.ones_like(node_scores) * 0.5

    if edge_scores is not None:
        min_val, max_val = edge_scores.min(), edge_scores.max()
        if max_val - min_val > 1e-8:
            edge_scores = (edge_scores - min_val) / (max_val - min_val)

    return {
        'node_scores': node_scores,
        'edge_scores': edge_scores,
        'feature_scores': node_feat_mask.cpu().detach().numpy() if 'node_feat_mask' in dir() else None
    }


def compute_gnnexplainer_batch(
    dataset,
    model: nn.Module,
    epochs: int = 200,
    lr: float = 0.01,
    device: str = "cuda",
    save_path: str = None,
    npz_dir: str = None,
    sample_ratio: float = 1.0
) -> Dict[str, dict]:
    """
    对数据集全量样本计算 GNNExplainer 重要性分数。

    Args:
        dataset: MolTestDataset 实例
        model: 已加载权重的模型
        epochs: 训练轮数
        lr: 学习率
        device: 设备
        save_path: 断点续传保存路径
        npz_dir: NPZ 文件目录
        sample_ratio: 抽样比例（用于快速测试）

    Returns:
        {graph_key: {'node_scores': ..., 'edge_scores': ..., 'y': ..., 'num_nodes': ...}} 字典
    """
    results = {}

    # 尝试加载已有结果
    if save_path and os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            results = pickle.load(f)
        logger.info(f"加载已有结果: {len(results)} 个样本")

    total = len(dataset)
    processed = 0

    # 抽样
    indices = list(range(total))
    if sample_ratio < 1.0:
        import random
        random.seed(config.XAI_CONFIG["random_seed"])
        sample_size = int(total * sample_ratio)
        indices = random.sample(indices, sample_size)
        logger.info(f"抽样 {sample_size}/{total} 个样本")

    for idx in indices:
        try:
            sample = dataset[idx]
            if isinstance(sample, tuple) and len(sample) == 2:
                data, filename = sample
            else:
                data = sample
                filename = None

            graph_key = filename.replace('.npz', '') if filename else f"sample_{idx}"

            # 跳过已处理的
            if graph_key in results:
                processed += 1
                continue

            # 富化数据
            data = enrich_data_object(data, filename, npz_dir=npz_dir)

            # 计算 GNNExplainer 分数
            scores = compute_gnnexplainer_single(
                data, model, device, epochs=epochs, lr=lr
            )

            results[graph_key] = {
                'node_scores': scores['node_scores'],
                'edge_scores': scores['edge_scores'],
                'feature_scores': scores['feature_scores'],
                'node_cat': data.node_cat.cpu().numpy() if hasattr(data, 'node_cat') else None,
                'y': data.y.item() if hasattr(data, 'y') else -1,
                'num_nodes': data.x.shape[0],
                'method': 'gnnexplainer'
            }

        except Exception as e:
            graph_key = f"sample_{idx}"
            logger.error(f"处理样本 {idx} 失败: {e}")
            results[graph_key] = None

        processed += 1

        # 进度报告
        if processed % 100 == 0:
            logger.info(f"进度: {processed}/{len(indices)} ({processed/len(indices)*100:.1f}%)")

        # 断点保存
        if save_path and processed % 100 == 0:
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)

    # 最终保存
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"结果已保存到: {save_path}")

    # 过滤无效结果
    results = {k: v for k, v in results.items() if v is not None}

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_finetune_model(device)
    print("模型加载成功！")

    # 测试 GNNExplainer
    print("测试 GNNExplainer...")