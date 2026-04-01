"""
PGExplainer 节点级归因分析模块
参数化图神经网络解释器，训练一个独立的 MLP 网络预测边重要性

参考论文: "Parameterized Explainer for Graph Neural Network"
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
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.utils import k_hop_subgraph


class DataOnlyWrapper(Dataset):
    """包装数据集，只返回 Data 对象而不返回 filename"""
    def __init__(self, original_dataset):
        super(Dataset, self).__init__()
        self.original_dataset = original_dataset

    def __getitem__(self, index):
        result = self.original_dataset[index]
        if isinstance(result, tuple):
            return result[0]  # 只返回 Data 对象
        return result

    def __len__(self):
        return len(self.original_dataset)

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


class PGExplainerModel(nn.Module):
    """
    PGExplainer 的参数化解释网络。
    输入边特征（拼接两端节点嵌入），输出边重要性概率。
    """

    def __init__(self, emb_dim: int = 18, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()

        # 边评分 MLP
        layers = []
        input_dim = emb_dim * 2  # 拼接两端节点嵌入（实际嵌入维度）

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
            ])
        layers.append(nn.Linear(hidden_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, node_emb: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_emb: [N, hidden_dim] 节点嵌入
            edge_index: [2, E] 边索引

        Returns:
            edge_probs: [E] 边重要性概率
        """
        # 获取两端节点嵌入
        src_emb = node_emb[edge_index[0]]  # [E, hidden_dim]
        dst_emb = node_emb[edge_index[1]]  # [E, hidden_dim]

        # 拼接（无向边，考虑两个方向）
        edge_feat = torch.cat([src_emb, dst_emb], dim=-1)  # [E, hidden_dim*2]

        # MLP 输出
        logits = self.mlp(edge_feat).squeeze(-1)  # [E]

        return logits


class PGExplainerTrainer:
    """
    PGExplainer 训练器。
    在整个数据集上训练解释网络，然后用于预测边重要性。
    """

    def __init__(
        self,
        model: nn.Module,
        hidden_dim: int = None,
        device: str = "cuda",
        lr: float = 0.005,
        temp: float = 1.0
    ):
        """
        Args:
            model: 目标 GNN 模型
            hidden_dim: 解释器隐藏维度（默认从模型配置推断）
            device: 设备
            lr: 学习率
            temp: Gumbel-Softmax 温度
        """
        self.model = model
        self.device = device
        self.temp = temp

        # 从模型配置获取嵌入维度和隐藏维度
        emb_dim = config.MODEL_CONFIG.get("emb_dim", 18)
        if hidden_dim is None:
            hidden_dim = config.XAI_CONFIG.get("pgexplainer_hidden", 64)

        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim

        # 创建解释器网络（emb_dim 是节点嵌入维度，hidden_dim 是 MLP 内部维度）
        self.explainer = PGExplainerModel(emb_dim=emb_dim, hidden_dim=hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.explainer.parameters(), lr=lr)

    def get_node_embeddings(self, data: Data) -> torch.Tensor:
        """
        获取 GNN 的节点嵌入（池化前）。
        需要在池化之前获取节点级别的嵌入。
        """
        self.model.eval()
        import torch.nn.functional as F

        x = data.x
        edge_index = data.edge_index
        edge_attr = None

        h = x

        # 遍历所有 GNN 层
        for layer in range(self.model.num_layer):
            h = self.model.gnns[layer](h, edge_index, edge_attr)
            h = self.model.batch_norms[layer](h)
            if layer == self.model.num_layer - 1:
                h = F.dropout(h, self.model.drop_ratio, training=False)
            else:
                h = F.dropout(F.relu(h), self.model.drop_ratio, training=False)

        # h 现在是节点嵌入 [N, emb_dim]
        return h

    def train_epoch(self, dataloader) -> float:
        """
        训练一个 epoch（优化版：向量化计算）。

        损失函数：
        - 预测损失：掩码后预测应与原预测一致
        - 稀疏性损失：掩码边数应尽量少
        """
        self.explainer.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            batch = batch.to(self.device)

            # 确保有 batch 属性
            if not hasattr(batch, 'batch') or batch.batch is None:
                batch.batch = torch.zeros(batch.x.shape[0], dtype=torch.long, device=self.device)

            # 获取节点嵌入
            node_emb = self.get_node_embeddings(batch)

            # 获取原预测
            with torch.no_grad():
                _, original_logits = self.model(batch)
                original_pred = original_logits.softmax(dim=-1)

            # 预测边重要性
            edge_logits = self.explainer(node_emb, batch.edge_index)

            # Gumbel-Softmax 采样得到边掩码
            edge_probs = torch.sigmoid(edge_logits)

            # 向量化计算加权节点嵌入
            num_nodes = batch.x.shape[0]
            edge_index = batch.edge_index

            # 使用 scatter_add 进行向量化聚合
            # 源节点到目标节点的加权聚合
            src_emb = node_emb[edge_index[0]] * edge_probs.unsqueeze(1)  # [E, emb_dim]
            dst_emb = node_emb[edge_index[1]] * edge_probs.unsqueeze(1)  # [E, emb_dim]

            # 聚合到节点
            weighted_node_emb = torch.zeros_like(node_emb)
            weighted_node_emb.scatter_add_(0, edge_index[1].unsqueeze(1).expand(-1, node_emb.shape[1]), src_emb)
            weighted_node_emb.scatter_add_(0, edge_index[0].unsqueeze(1).expand(-1, node_emb.shape[1]), dst_emb)

            # 计算度数（向量化）
            degree = torch.zeros(num_nodes, device=self.device)
            degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1], device=self.device))
            degree.scatter_add_(0, edge_index[1], torch.ones(edge_index.shape[1], device=self.device))
            degree = degree.clamp(min=1).unsqueeze(1)

            weighted_node_emb = weighted_node_emb / degree

            # 损失计算
            # 1. 稀疏性损失
            sparsity_loss = edge_probs.mean()

            # 2. 边熵损失：鼓励边权重趋向0或1
            entropy_loss = -(edge_probs * torch.log(edge_probs + 1e-8) +
                           (1 - edge_probs) * torch.log(1 - edge_probs + 1e-8)).mean()

            # 3. 预测一致性损失（简化版）
            # 使用加权嵌入计算图级表示，与原始预测比较
            graph_emb = weighted_node_emb.mean(dim=0, keepdim=True)  # [1, emb_dim]
            # 通过预测头获取预测
            with torch.no_grad():
                original_graph_emb = node_emb.mean(dim=0, keepdim=True)

            # 嵌入一致性损失
            consistency_loss = 1 - torch.cosine_similarity(
                graph_emb.flatten(),
                original_graph_emb.flatten(),
                dim=0
            )

            # 总损失
            loss = sparsity_loss + 0.1 * entropy_loss + 0.5 * consistency_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def explain(
        self,
        data: Data,
        target_class: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        解释单个图，返回节点和边重要性。

        Args:
            data: PyG Data 对象
            target_class: 目标类别

        Returns:
            node_scores: [N] 节点重要性
            edge_scores: [E] 边重要性
        """
        self.explainer.eval()
        data = data.to(self.device)

        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=self.device)

        # 获取节点嵌入
        node_emb = self.get_node_embeddings(data)

        # 预测边重要性
        with torch.no_grad():
            edge_logits = self.explainer(node_emb, data.edge_index)
            edge_scores = torch.sigmoid(edge_logits).cpu().numpy()

        # 计算节点重要性（向量化聚合边重要性）
        num_nodes = data.x.shape[0]
        edge_index_np = data.edge_index.cpu().numpy()

        # 向量化聚合
        node_scores = np.zeros(num_nodes)
        np.add.at(node_scores, edge_index_np[0], edge_scores)
        np.add.at(node_scores, edge_index_np[1], edge_scores)

        # 归一化
        if node_scores.max() > 0:
            node_scores = node_scores / node_scores.max()

        # 归一化边分数到 [0, 1]
        if edge_scores.max() - edge_scores.min() > 1e-8:
            edge_scores = (edge_scores - edge_scores.min()) / (edge_scores.max() - edge_scores.min())

        return node_scores, edge_scores

    def explain_batch(
        self,
        data_list: List[Data],
        target_classes: List[int] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        批量解释多个图。

        Args:
            data_list: Data 对象列表
            target_classes: 目标类别列表

        Returns:
            [(node_scores, edge_scores), ...] 列表
        """
        self.explainer.eval()

        # 创建批量数据
        batch_data = Batch.from_data_list(data_list).to(self.device)

        # 获取节点嵌入
        node_emb = self.get_node_embeddings(batch_data)

        # 预测边重要性
        with torch.no_grad():
            edge_logits = self.explainer(node_emb, batch_data.edge_index)
            edge_scores_all = torch.sigmoid(edge_logits).cpu().numpy()

        # 分割边分数到每个图
        results = []
        edge_offset = 0
        node_offset = 0

        for i, data in enumerate(data_list):
            num_nodes = data.x.shape[0]
            num_edges = data.edge_index.shape[1]

            # 提取该图的边分数
            edge_scores = edge_scores_all[edge_offset:edge_offset + num_edges]

            # 提取该图的边索引
            edge_index_np = data.edge_index.cpu().numpy()

            # 计算节点重要性
            node_scores = np.zeros(num_nodes)
            np.add.at(node_scores, edge_index_np[0], edge_scores)
            np.add.at(node_scores, edge_index_np[1], edge_scores)

            # 归一化
            if node_scores.max() > 0:
                node_scores = node_scores / node_scores.max()

            if edge_scores.max() - edge_scores.min() > 1e-8:
                edge_scores = (edge_scores - edge_scores.min()) / (edge_scores.max() - edge_scores.min())

            results.append((node_scores, edge_scores))

            edge_offset += num_edges
            node_offset += num_nodes

        return results

    def fit(self, dataset_or_list, epochs: int = 100, batch_size: int = 32, early_stop_patience: int = 10):
        """
        在数据集上训练解释器。

        Args:
            dataset_or_list: 数据集对象或 Data 列表
            epochs: 训练轮数
            batch_size: 批量大小
            early_stop_patience: 早停耐心值（连续N轮loss无改善则停止）
        """
        from torch_geometric.data import DataLoader
        import time

        # 处理输入：如果是列表，转换为 Dataset
        if isinstance(dataset_or_list, list):
            # 创建简单列表数据集
            class ListDataset(Dataset):
                def __init__(self, data_list):
                    super(Dataset, self).__init__()
                    self.data_list = data_list

                def __getitem__(self, idx):
                    return self.data_list[idx]

                def __len__(self):
                    return len(self.data_list)

            wrapped_dataset = ListDataset(dataset_or_list)
        else:
            # 包装数据集，只返回 Data 对象
            wrapped_dataset = DataOnlyWrapper(dataset_or_list)

        dataloader = DataLoader(wrapped_dataset, batch_size=batch_size, shuffle=True)

        logger.info(f"开始训练 PGExplainer，共 {epochs} 轮，batch_size={batch_size}")
        start_time = time.time()

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            loss = self.train_epoch(dataloader)

            # 早停检查
            if loss < best_loss - 0.001:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (epoch + 1) * (epochs - epoch - 1)
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Best: {best_loss:.4f}, ETA: {eta/60:.1f}分钟")

            # 早停
            if patience_counter >= early_stop_patience and epoch >= 10:
                logger.info(f"早停触发：连续 {early_stop_patience} 轮无改善")
                break

        total_time = time.time() - start_time
        logger.info(f"PGExplainer 训练完成，共 {epoch+1} 轮，总耗时: {total_time/60:.1f}分钟")


def compute_pgexplainer_batch(
    dataset,
    model: nn.Module,
    epochs: int = 100,
    hidden_dim: int = 64,
    lr: float = 0.005,
    device: str = "cuda",
    save_path: str = None,
    npz_dir: str = None,
    train_ratio: float = 0.8
) -> Dict[str, dict]:
    """
    使用 PGExplainer 对数据集计算重要性分数。

    Args:
        dataset: MolTestDataset 实例
        model: 已加载权重的模型
        epochs: 训练轮数
        hidden_dim: 解释器隐藏维度
        lr: 学习率
        device: 设备
        save_path: 保存路径
        npz_dir: NPZ 目录
        train_ratio: 训练集比例

    Returns:
        结果字典
    """
    results = {}

    # 加载已有结果
    if save_path and os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            results = pickle.load(f)
        logger.info(f"加载已有结果: {len(results)} 个样本")

    # 创建并训练 PGExplainer
    trainer = PGExplainerTrainer(
        model=model,
        hidden_dim=hidden_dim,
        device=device,
        lr=lr
    )

    # 抽样训练数据集（只用部分数据训练）
    train_ratio = config.XAI_CONFIG.get("pgexplainer_train_ratio", 0.2)
    total_samples = len(dataset)
    train_size = int(total_samples * train_ratio)

    import random
    random.seed(config.XAI_CONFIG.get("random_seed", 42))
    train_indices = random.sample(range(total_samples), min(train_size, total_samples))

    # 创建抽样训练数据集
    train_dataset = [dataset[idx][0] if isinstance(dataset[idx], tuple) else dataset[idx] for idx in train_indices]
    logger.info(f"抽样 {len(train_dataset)}/{total_samples} 个样本用于训练")

    # 训练解释器（带早停）
    trainer.fit(train_dataset, epochs=epochs, early_stop_patience=10)

    # 批量解释所有样本（优化版）
    total = len(dataset)
    batch_size = 50  # 批量解释大小
    logger.info(f"批量解释 {total} 个样本，batch_size={batch_size}")

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_indices = range(batch_start, batch_end)

        # 收集批量数据
        batch_data_list = []
        batch_keys = []
        batch_filenames = []

        for idx in batch_indices:
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
                batch_data_list.append(data)
                batch_keys.append(graph_key)
                batch_filenames.append(filename)

            except Exception as e:
                logger.error(f"准备样本 {idx} 失败: {e}")

        if not batch_data_list:
            continue

        # 批量解释
        try:
            batch_results = trainer.explain_batch(batch_data_list)

            # 存储结果
            for i, (node_scores, edge_scores) in enumerate(batch_results):
                data = batch_data_list[i]
                graph_key = batch_keys[i]

                results[graph_key] = {
                    'node_scores': node_scores,
                    'edge_scores': edge_scores,
                    'feature_scores': None,
                    'node_cat': data.node_cat.cpu().numpy() if hasattr(data, 'node_cat') else None,
                    'y': data.y.item() if hasattr(data, 'y') else -1,
                    'num_nodes': data.x.shape[0],
                    'method': 'pgexplainer'
                }

        except Exception as e:
            logger.error(f"批量解释失败 ({batch_start}-{batch_end}): {e}")
            # 回退到逐个处理
            for i, data in enumerate(batch_data_list):
                try:
                    node_scores, edge_scores = trainer.explain(data)
                    graph_key = batch_keys[i]
                    results[graph_key] = {
                        'node_scores': node_scores,
                        'edge_scores': edge_scores,
                        'feature_scores': None,
                        'node_cat': data.node_cat.cpu().numpy() if hasattr(data, 'node_cat') else None,
                        'y': data.y.item() if hasattr(data, 'y') else -1,
                        'num_nodes': data.x.shape[0],
                        'method': 'pgexplainer'
                    }
                except Exception as e2:
                    logger.error(f"处理样本 {batch_keys[i]} 失败: {e2}")

        # 进度报告
        processed = sum(1 for k in results if not k.startswith('sample_') or results[k] is not None)
        if processed % 500 == 0:
            logger.info(f"进度: {processed}/{total}")

    # 保存
    if save_path:
        results_filtered = {k: v for k, v in results.items() if v is not None}
        with open(save_path, 'wb') as f:
            pickle.dump(results_filtered, f)
        logger.info(f"结果已保存到: {save_path}")

    return {k: v for k, v in results.items() if v is not None}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("PGExplainer 模块已加载")