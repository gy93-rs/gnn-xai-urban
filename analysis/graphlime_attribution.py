"""
GraphLIME 节点级归因分析模块
基于 LIME (Local Interpretable Model-agnostic Explanations) 的图神经网络解释方法

参考论文: "GraphLIME: Local Interpretable Model Explanations for Graph Neural Networks"
"""

import os
import sys
import logging
import pickle
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

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


class GraphLIMEExplainer:
    """
    GraphLIME 解释器。
    在目标节点邻域内，用线性模型拟合 GNN 预测。
    线性模型的系数作为节点特征重要性。
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        num_samples: int = 5000,
        alpha: float = 1.0
    ):
        """
        Args:
            model: 目标 GNN 模型
            device: 设备
            num_samples: 扰动样本数
            alpha: Ridge 回归正则化系数
        """
        self.model = model
        self.device = device
        self.num_samples = num_samples
        self.alpha = alpha

    def perturb_features(
        self,
        x: torch.Tensor,
        num_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        扰动节点特征，生成局部样本。

        Args:
            x: [N, F] 节点特征
            num_samples: 扰动样本数

        Returns:
            perturbed_masks: [num_samples, F] 扰动掩码
            perturbed_preds: [num_samples, num_classes] 扰动后的预测
        """
        num_nodes, num_features = x.shape

        # 对于图分类，我们扰动所有节点的特征
        # 扰动方式：随机掩码特征维度
        perturbed_masks = []
        perturbed_preds = []

        self.model.eval()

        # 批量处理
        batch_size = 100
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            batch_masks = []

            for _ in range(current_batch_size):
                # 随机掩码：每个特征维度有 50% 概率被保留
                mask = (torch.rand(num_features) > 0.5).float()
                batch_masks.append(mask)

            batch_masks = torch.stack(batch_masks)  # [B, F]

            # 应用掩码
            perturbed_x = x.unsqueeze(0) * batch_masks.unsqueeze(1).to(self.device)  # [B, N, F]

            # 预测
            with torch.no_grad():
                for j in range(current_batch_size):
                    data_perturbed = Data(
                        x=perturbed_x[j],
                        edge_index=torch.zeros((2, 0), dtype=torch.long, device=self.device)
                    )
                    # 简化：使用原始边
                    # 实际应用中需要传入 edge_index

            perturbed_masks.extend(batch_masks.cpu().numpy())

        return np.array(perturbed_masks[:num_samples]), None

    def explain_graph(
        self,
        data: Data,
        target_class: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用 GraphLIME 解释图分类。

        对于图分类任务，我们：
        1. 扰动节点特征
        2. 观察预测变化
        3. 用线性模型拟合扰动与预测的关系
        4. 系数的绝对值作为特征重要性

        Args:
            data: PyG Data 对象
            target_class: 目标类别

        Returns:
            node_scores: [N] 节点重要性
            feature_scores: [F] 特征重要性
        """
        self.model.eval()
        data = data.to(self.device)

        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=self.device)

        # 获取原始预测
        with torch.no_grad():
            _, logits = self.model(data)
            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())

        num_nodes = data.x.shape[0]
        num_features = data.x.shape[1]

        # 生成扰动样本
        perturbed_masks = []
        predictions = []

        logger.debug(f"生成 {self.num_samples} 个扰动样本...")

        for _ in range(self.num_samples):
            # 随机掩码节点（更直观）
            node_mask = (torch.rand(num_nodes) > 0.5).float().to(self.device)

            # 应用扰动：掩码节点的所有特征
            perturbed_x = data.x * node_mask.unsqueeze(1)

            # 预测
            with torch.no_grad():
                perturbed_data_obj = Data(
                    x=perturbed_x,
                    edge_index=data.edge_index,
                    batch=data.batch
                )
                _, perturbed_logits = self.model(perturbed_data_obj)
                perturbed_pred = perturbed_logits.softmax(dim=-1)

                # 获取目标类别的预测概率
                pred_prob = perturbed_pred[0, target_class].cpu().numpy()

            # 记录扰动和预测
            perturbed_masks.append(node_mask.cpu().numpy())
            predictions.append(pred_prob)

        # 转换为数组
        X = np.array(perturbed_masks)  # [num_samples, N]
        y = np.array(predictions)  # [num_samples]

        # 用 Ridge 回归拟合
        try:
            from sklearn.linear_model import Ridge

            ridge = Ridge(alpha=self.alpha)
            ridge.fit(X, y)

            # 系数绝对值作为节点重要性
            node_scores = np.abs(ridge.coef_)

        except ImportError:
            logger.warning("sklearn 未安装，使用简化方法")
            # 简化：使用相关性
            if X.shape[0] > 1:
                node_scores = np.abs(np.corrcoef(X.T, y)[-1, :-1])
                node_scores = np.nan_to_num(node_scores, nan=0.0)
            else:
                node_scores = np.ones(num_nodes)

        # 归一化到 [0, 1]
        if node_scores.max() - node_scores.min() > 1e-8:
            node_scores = (node_scores - node_scores.min()) / (node_scores.max() - node_scores.min())
        else:
            node_scores = np.ones(num_nodes) * 0.5

        # 特征重要性（简化）
        feature_scores = np.ones(num_features) / num_features

        return node_scores, feature_scores

    def explain(
        self,
        data: Data,
        target_class: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        解释入口函数。

        Args:
            data: PyG Data 对象
            target_class: 目标类别

        Returns:
            node_scores: [N] 节点重要性
            feature_scores: [F] 特征重要性
        """
        return self.explain_graph(data, target_class)


def compute_graphlime_single(
    data: Data,
    model: nn.Module,
    device: str = "cuda",
    num_samples: int = 5000,
    alpha: float = 1.0,
    target_class: int = None
) -> Dict[str, np.ndarray]:
    """
    对单个图使用 GraphLIME 计算节点重要性。

    Args:
        data: PyG Data 对象
        model: 已加载权重的模型
        device: 设备
        num_samples: 扰动样本数
        alpha: Ridge 正则化系数
        target_class: 目标类别

    Returns:
        包含 node_scores, feature_scores 的字典
    """
    explainer = GraphLIMEExplainer(
        model=model,
        device=device,
        num_samples=num_samples,
        alpha=alpha
    )

    node_scores, feature_scores = explainer.explain(data, target_class)

    return {
        'node_scores': node_scores,
        'edge_scores': None,  # GraphLIME 不直接提供边重要性
        'feature_scores': feature_scores
    }


def compute_graphlime_batch(
    dataset,
    model: nn.Module,
    num_samples: int = 5000,
    alpha: float = 1.0,
    device: str = "cuda",
    save_path: str = None,
    npz_dir: str = None,
    sample_ratio: float = 1.0
) -> Dict[str, dict]:
    """
    使用 GraphLIME 对数据集计算重要性分数。

    Args:
        dataset: MolTestDataset 实例
        model: 已加载权重的模型
        num_samples: 扰动样本数
        alpha: Ridge 正则化系数
        device: 设备
        save_path: 保存路径
        npz_dir: NPZ 目录
        sample_ratio: 抽样比例

    Returns:
        结果字典
    """
    results = {}

    # 加载已有结果
    if save_path and os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            results = pickle.load(f)
        logger.info(f"加载已有结果: {len(results)} 个样本")

    total = len(dataset)
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

            if graph_key in results:
                continue

            # 富化数据
            data = enrich_data_object(data, filename, npz_dir=npz_dir)

            # 计算 GraphLIME 分数
            scores = compute_graphlime_single(
                data, model, device,
                num_samples=num_samples,
                alpha=alpha
            )

            results[graph_key] = {
                'node_scores': scores['node_scores'],
                'edge_scores': scores['edge_scores'],
                'feature_scores': scores['feature_scores'],
                'node_cat': data.node_cat.cpu().numpy() if hasattr(data, 'node_cat') else None,
                'y': data.y.item() if hasattr(data, 'y') else -1,
                'num_nodes': data.x.shape[0],
                'method': 'graphlime'
            }

        except Exception as e:
            logger.error(f"处理样本 {idx} 失败: {e}")
            results[f"sample_{idx}"] = None

        # 进度
        processed = sum(1 for k in results if not k.startswith('sample_') or results[k] is not None)
        if processed % 100 == 0:
            logger.info(f"进度: {processed}/{len(indices)}")

        # 断点保存
        if save_path and processed % 200 == 0:
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)

    # 最终保存
    if save_path:
        results_filtered = {k: v for k, v in results.items() if v is not None}
        with open(save_path, 'wb') as f:
            pickle.dump(results_filtered, f)
        logger.info(f"结果已保存到: {save_path}")

    return {k: v for k, v in results.items() if v is not None}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("GraphLIME 模块已加载")