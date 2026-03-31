"""
节点级归因分析模块
实现 GradCAM 方法识别对 UST 分类贡献最大的地物节点
"""

import os
import sys
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import DataLoader as PyGDataLoader

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 添加原始项目路径（models/dataset 在那里）
ORIGINAL_PROJECT_ROOT = "/media/gy/ssd/shanghai_exp/MolCLR-Urban_alldata1127_512_cls"
if ORIGINAL_PROJECT_ROOT not in sys.path:
    sys.path.insert(0, ORIGINAL_PROJECT_ROOT)

import xai_config as config

logger = logging.getLogger(__name__)


def enrich_data_object(data: Data, filename: str = None, npz_dir: str = None) -> Data:
    """
    在现有 Data 对象上追加可解释性所需的额外字段。

    追加字段：
    - data.node_cat: [N] int，节点类别（从原始 NPZ 文件提取）
    - data.graph_key: str，图标识符

    Args:
        data: PyG Data 对象
        filename: 图文件名（可选）
        npz_dir: NPZ 文件目录（用于重新加载原始节点类别）

    Returns:
        富化后的 Data 对象
    """
    # 节点类别：从原始 NPZ 文件重新提取
    # 注意：dataset_finetune.py 中的 one-hot 编码有 bug（广播导致所有节点相同）
    # 所以必须从原始文件读取 col3 (节点类别)
    if not hasattr(data, 'node_cat') or (hasattr(data, 'node_cat') and data.node_cat.unique().numel() == 1):
        if filename is not None and npz_dir is not None:
            import numpy as np
            npz_path = os.path.join(npz_dir, filename)
            if os.path.exists(npz_path):
                npz_data = np.load(npz_path, allow_pickle=True)
                x1 = npz_data['array1']
                # col3 是原始节点类别，值范围 0-9
                # 转换为 0-based: 0->0, 1->0, 2->1, ..., 9->8
                # 但实际上 col3=0 表示背景/未知，col3=1-9 对应9个类别
                node_cat = x1[:, 3].astype(int)
                # 将 1-9 映射到 0-8，0 保持为 0（或设为 -1 表示未知）
                node_cat = np.clip(node_cat - 1, -1, 8)  # 0->-1, 1->0, ..., 9->8
                data.node_cat = torch.tensor(node_cat, dtype=torch.long)
            else:
                # 降级：从 one-hot 解码（可能不准确）
                data.node_cat = data.x[:, 3:12].argmax(dim=1).long()
        else:
            data.node_cat = data.x[:, 3:12].argmax(dim=1).long()

    if not hasattr(data, 'graph_key'):
        if filename is not None:
            data.graph_key = filename.replace('.npz', '')
        else:
            data.graph_key = "unknown"

    return data


def load_finetune_model(device: str = "cuda") -> nn.Module:
    """
    加载 gcn_finetune.py 中的模型并恢复微调权重。

    Args:
        device: 设备类型

    Returns:
        已加载权重、处于 eval 模式的模型实例
    """
    from models.gcn_finetune import GCN

    # 实例化模型
    model = GCN(
        task=config.MODEL_CONFIG["task"],
        num_layer=config.MODEL_CONFIG["num_layer"],
        emb_dim=config.MODEL_CONFIG["emb_dim"],
        feat_dim=config.MODEL_CONFIG["feat_dim"],
        drop_ratio=config.MODEL_CONFIG["drop_ratio"],
        pool=config.MODEL_CONFIG["pool"]
    )

    # 加载权重
    weights_path = config.MODEL_CONFIG["finetune_weights"]
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")

    state_dict = torch.load(weights_path, map_location=device)

    # 处理 DataParallel 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k.replace("module.", "")] = v
        else:
            new_state_dict[k] = v

    # 加载权重
    try:
        model.load_state_dict(new_state_dict, strict=True)
        logger.info(f"成功加载权重，所有参数匹配")
    except RuntimeError as e:
        logger.warning(f"部分参数不匹配: {e}")
        model.load_state_dict(new_state_dict, strict=False)

    model = model.to(device)
    model.eval()

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型总参数量: {total_params:,}")

    return model


class GradCAMHook:
    """
    通过 register_forward_hook 捕获激活，手动计算梯度。
    兼容旧版 PyTorch。
    """

    def __init__(self, model: nn.Module, target_layer_name: str = "gnns.-1"):
        """
        Args:
            model: GCN 模型
            target_layer_name: 目标层名称，默认最后一层 GNN
        """
        self.model = model
        self.target_layer_name = target_layer_name

        # 存储激活和梯度
        self.activations = None

        # 获取目标层
        self.target_layer = self._get_target_layer()

        # 只注册前向钩子
        self.forward_hook = self.target_layer.register_forward_hook(self._forward_hook)

    def _get_target_layer(self) -> nn.Module:
        """获取目标层模块"""
        # gnns.-1 表示最后一个 GCNConv 层
        if self.target_layer_name == "gnns.-1":
            return self.model.gnns[-1]

        # 支持其他层名
        parts = self.target_layer_name.split('.')
        module = self.model
        for part in parts:
            if part.isdigit() or (part.startswith('-') and part[1:].isdigit()):
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def _forward_hook(self, module, input, output):
        """前向钩子：保存激活值，并启用梯度追踪"""
        # 保留计算图以便反向传播
        self.activations = output
        self.activations.retain_grad()

    def compute_gradcam(self, num_nodes: int) -> np.ndarray:
        """
        计算 GradCAM 节点重要性分数。

        公式：
            α_c = mean_v(∂score_y / ∂h_v^(L)_c)
            importance_v = ReLU(∑_c α_c × h_v^(L)_c)
            再做 per-graph min-max 归一化到 [0, 1]

        Args:
            num_nodes: 节点数量

        Returns:
            node_scores: [N] numpy array，范围 [0, 1]
        """
        if self.activations is None:
            raise RuntimeError("需要先执行 forward")

        if self.activations.grad is None:
            raise RuntimeError("需要先执行 backward")

        # activations: [N, emb_dim]
        # gradients: [N, emb_dim]
        gradients = self.activations.grad
        activations = self.activations.detach()

        # 计算权重 α_c (通道维度的平均梯度)
        alpha = gradients.mean(dim=0)  # [emb_dim]

        # 计算节点重要性
        node_importance = torch.relu(alpha * activations).sum(dim=1)  # [N]

        # 归一化到 [0, 1]
        node_importance = node_importance.cpu().numpy()
        min_val = float(node_importance.min())
        max_val = float(node_importance.max())

        if (max_val - min_val) > 1e-8:
            node_scores = (node_importance - min_val) / (max_val - min_val)
        else:
            # 所有分数相同，设为 0
            node_scores = np.zeros_like(node_importance)
            logger.warning("节点分数退化（全相同），归一化后全为 0")

        return node_scores

    def remove(self):
        """移除钩子"""
        self.forward_hook.remove()


def compute_node_scores_single(
    data: Data,
    model: nn.Module,
    device: str = "cuda",
    target_class: int = None
) -> np.ndarray:
    """
    对单个图计算节点重要性分数。
    使用梯度×激活值的方法（类 GradCAM，但直接作用于节点嵌入）。

    Args:
        data: PyG Data 对象
        model: 已加载权重的模型
        device: 设备
        target_class: 目标类别索引（None 则使用预测类别）

    Returns:
        node_scores: [N] numpy array
    """
    model.eval()  # 使用 eval 模式避免 BatchNorm 的 batch size 问题

    data = data.to(device)

    # 单样本需要设置 batch 属性
    if not hasattr(data, 'batch') or data.batch is None:
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)

    # 方法：使用输入特征的梯度来估计节点重要性
    # 创建输入特征的副本，使其可微分
    x = data.x.clone().detach().requires_grad_(True)
    data.x = x  # 替换为可微分版本

    # Forward
    h, logits = model(data)

    # 获取目标类别
    if target_class is None:
        target_class = logits.argmax(dim=1).item()

    # 使用 torch.autograd.grad 显式计算梯度
    score = logits[0, target_class]
    gradients = torch.autograd.grad(score, x, retain_graph=False, create_graph=False)[0]

    # 使用输入特征梯度的 L2 范数作为节点重要性
    if gradients is not None:
        node_importance = gradients.pow(2).sum(dim=1).sqrt()
    else:
        # 如果梯度为 None，使用零向量
        node_importance = torch.zeros(data.x.shape[0], device=device)

    # 归一化到 [0, 1]
    node_importance = node_importance.cpu().detach().numpy()
    min_val = float(node_importance.min())
    max_val = float(node_importance.max())

    if (max_val - min_val) > 1e-8:
        node_scores = (node_importance - min_val) / (max_val - min_val)
    else:
        # 所有分数相同，设为均匀分布
        node_scores = np.ones_like(node_importance) * 0.5
        logger.warning("节点分数退化（全相同），设为 0.5")

    # 清理
    model.zero_grad()

    return node_scores


def compute_node_scores_batch(
    dataset,
    model: nn.Module,
    batch_size: int = 32,
    device: str = "cuda",
    save_path: str = None,
    npz_dir: str = None
) -> Dict[str, np.ndarray]:
    """
    对数据集全量样本计算节点重要性分数。

    Args:
        dataset: MolTestDataset 实例
        model: 已加载权重的模型
        batch_size: 批量大小（未使用，保留兼容性）
        device: 设备
        save_path: 断点续传保存路径
        npz_dir: NPZ 文件目录（用于重新加载原始节点类别）

    Returns:
        {graph_key: node_scores} 字典
    """
    results = {}

    # 尝试加载已有结果（断点续传）
    if save_path and os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            results = pickle.load(f)
        logger.info(f"加载已有结果: {len(results)} 个样本")

    total = len(dataset)
    processed = 0

    # 直接遍历数据集，每个样本单独处理
    for idx in range(total):
        try:
            # 获取样本（MolTestDataset 返回 (data, filename) 元组）
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

            # 计算节点分数
            node_scores = compute_node_scores_single(data, model, device)

            results[graph_key] = {
                'node_scores': node_scores,
                'node_cat': data.node_cat.cpu().numpy() if hasattr(data, 'node_cat') else None,
                'y': data.y.item() if hasattr(data, 'y') else -1,
                'num_nodes': data.x.shape[0]
            }

        except Exception as e:
            graph_key = f"sample_{idx}"
            logger.error(f"处理样本 {idx} 失败: {e}")
            results[graph_key] = None

        processed += 1

        # 进度报告
        if processed % 500 == 0:
            logger.info(f"进度: {processed}/{total} ({processed/total*100:.1f}%)")

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


class NodeAttributionAnalyzer:
    """节点归因统计分析类"""

    def __init__(self, results: Dict[str, dict], num_classes: int = 17):
        """
        Args:
            results: compute_node_scores_batch 的返回结果
            num_classes: UST 类别数
        """
        self.results = results
        self.num_classes = num_classes

        # 构建汇总 DataFrame
        self._build_summary_df()

    def _build_summary_df(self):
        """构建汇总数据框"""
        records = []
        for graph_key, data in self.results.items():
            if data is None:
                continue

            node_scores = data['node_scores']
            node_cat = data['node_cat']
            y = data['y']

            for i in range(len(node_scores)):
                record = {
                    'graph_key': graph_key,
                    'ust_label': y,
                    'node_idx': i,
                    'node_cat': node_cat[i] if node_cat is not None else -1,
                    'score': node_scores[i]
                }
                records.append(record)

        self.summary_df = pd.DataFrame(records)

    def score_by_node_category(self, ust_label: int) -> pd.DataFrame:
        """
        对指定 UST 类别，按节点类别分组统计。

        Args:
            ust_label: UST 类别标签

        Returns:
            统计 DataFrame
        """
        df = self.summary_df[self.summary_df['ust_label'] == ust_label]

        stats = df.groupby('node_cat')['score'].agg([
            ('count', 'count'),
            ('mean_score', 'mean'),
            ('median_score', 'median'),
            ('std_score', 'std'),
            ('q25_score', lambda x: x.quantile(0.25)),
            ('q75_score', lambda x: x.quantile(0.75))
        ]).reset_index()

        # 添加节点类别名称
        stats['node_cat_name'] = stats['node_cat'].map(
            lambda x: config.NODE_CATEGORY_NAMES.get(x, f"cat_{x}")
        )

        return stats

    def cross_ust_node_importance_matrix(self) -> pd.DataFrame:
        """
        计算节点类别 × UST 类别平均重要性矩阵。

        Returns:
            DataFrame: index=节点类别名, columns=UST标签
        """
        # 计算每对 (node_cat, ust_label) 的平均分数
        matrix = self.summary_df.groupby(['node_cat', 'ust_label'])['score'].mean().unstack()

        # 重命名索引
        matrix.index = [config.NODE_CATEGORY_NAMES.get(x, f"cat_{x}") for x in matrix.index]

        # 重命名列
        matrix.columns = [f"UST-{c}" for c in matrix.columns]

        return matrix

    def export_summary_csv(self, output_path: str):
        """
        导出长格式 CSV。

        Args:
            output_path: 输出路径
        """
        export_df = self.summary_df.copy()
        export_df['node_cat_name'] = export_df['node_cat'].map(
            lambda x: config.NODE_CATEGORY_NAMES.get(x, f"cat_{x}")
        )
        export_df['ust_name'] = export_df['ust_label'].map(
            lambda x: config.UST_NAMES.get(x, f"UST-{x}")
        )

        export_df.to_csv(output_path, index=False)
        logger.info(f"汇总 CSV 已保存: {output_path}")


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    # 测试模型加载
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_finetune_model(device)
    print("模型加载成功！")