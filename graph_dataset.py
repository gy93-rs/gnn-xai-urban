"""
图神经网络分类数据集类
支持从文件加载图数据或直接传入图数据列表
"""

import os
import json
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import List, Optional, Tuple, Union
import numpy as np


class GraphClassificationDataset(Dataset):
    """
    用于图神经网络分类任务的数据集类。

    Args:
        graphs: 图数据列表，每个元素是一个字典，包含:
            - node_features: 节点特征矩阵 [num_nodes, num_features]
            - edge_index: 边索引 [2, num_edges]
            - edge_attr: 边特征 [num_edges, num_edge_features] (可选)
            - label: 图级别标签
        transform: 可选的数据增强转换函数
    """

    def __init__(
        self,
        graphs: Optional[List[dict]] = None,
        data_path: Optional[str] = None,
        transform=None
    ):
        self.transform = transform
        self.graphs = []

        if data_path is not None:
            self._load_from_path(data_path)
        elif graphs is not None:
            self._load_from_list(graphs)

    def _load_from_list(self, graphs: List[dict]):
        """从图数据列表加载"""
        for graph_data in graphs:
            pyg_data = self._convert_to_pyg_data(graph_data)
            self.graphs.append(pyg_data)

    def _convert_to_pyg_data(self, graph_data: dict) -> Data:
        """将字典格式的图数据转换为 PyG Data 对象"""
        node_features = torch.tensor(graph_data['node_features'], dtype=torch.float32)
        edge_index = torch.tensor(graph_data['edge_index'], dtype=torch.long)
        label = torch.tensor(graph_data['label'], dtype=torch.long)

        data = Data(
            x=node_features,
            edge_index=edge_index,
            y=label
        )

        # 可选的边特征
        if 'edge_attr' in graph_data:
            data.edge_attr = torch.tensor(graph_data['edge_attr'], dtype=torch.float32)

        # 可选的节点数量（用于批处理时排序）
        if 'num_nodes' in graph_data:
            data.num_nodes = graph_data['num_nodes']

        return data

    def _load_from_path(self, path: str):
        """从文件路径加载数据"""
        if os.path.isdir(path):
            self._load_from_directory(path)
        else:
            self._load_from_file(path)

    def _load_from_file(self, filepath: str):
        """从单个文件加载"""
        ext = os.path.splitext(filepath)[1].lower()

        if ext == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for graph_data in data:
                        pyg_data = self._convert_to_pyg_data(graph_data)
                        self.graphs.append(pyg_data)
                else:
                    self.graphs.append(self._convert_to_pyg_data(data))
        elif ext in ['.pt', '.pth']:
            loaded = torch.load(filepath)
            if isinstance(loaded, list):
                self.graphs = loaded
            else:
                self.graphs.append(loaded)
        elif ext == '.npz':
            npz_data = np.load(filepath, allow_pickle=True)
            for key in npz_data.files:
                graph_data = npz_data[key].item()
                self.graphs.append(self._convert_to_pyg_data(graph_data))

    def _load_from_directory(self, dirpath: str):
        """从目录加载所有图数据文件"""
        for filename in sorted(os.listdir(dirpath)):
            filepath = os.path.join(dirpath, filename)
            if os.path.isfile(filepath):
                self._load_from_file(filepath)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        graph = self.graphs[idx]
        if self.transform is not None:
            graph = self.transform(graph)
        return graph

    def get_num_classes(self) -> int:
        """获取类别数量"""
        labels = [g.y.item() for g in self.graphs]
        return len(set(labels))

    def get_num_node_features(self) -> int:
        """获取节点特征维度"""
        if len(self.graphs) > 0:
            return self.graphs[0].x.shape[1]
        return 0

    def get_num_edge_features(self) -> int:
        """获取边特征维度"""
        if len(self.graphs) > 0 and hasattr(self.graphs[0], 'edge_attr'):
            return self.graphs[0].edge_attr.shape[1]
        return 0

    def save(self, filepath: str):
        """保存数据集到文件"""
        torch.save(self.graphs, filepath)

    @staticmethod
    def collate_fn(batch: List[Data]) -> Data:
        """
        自定义批处理函数，用于 DataLoader
        将多个图数据合并成一个批次
        """
        from torch_geometric.data import Batch
        return Batch.from_data_list(batch)


def test_graph_dataset():
    """
    测试函数：创建虚拟数据并验证 GraphClassificationDataset 类是否正常工作
    """
    print("=" * 60)
    print("开始测试 GraphClassificationDataset")
    print("=" * 60)

    # 1. 创建虚拟图数据
    print("\n[步骤 1] 创建虚拟图数据...")
    dummy_graphs = []
    num_graphs = 5
    num_node_features = 8
    num_classes = 3

    for i in range(num_graphs):
        num_nodes = np.random.randint(5, 12)
        num_edges = np.random.randint(8, 20)

        graph = {
            'node_features': np.random.randn(num_nodes, num_node_features).astype(np.float32),
            'edge_index': np.random.randint(0, num_nodes, (2, num_edges)).astype(np.int64),
            'label': i % num_classes  # 循环分配标签，确保有多类别
        }
        dummy_graphs.append(graph)
        print(f"  图 {i+1}: 节点数={num_nodes}, 边数={num_edges}, 标签={graph['label']}")

    # 2. 实例化 Dataset
    print("\n[步骤 2] 实例化 GraphClassificationDataset...")
    dataset = GraphClassificationDataset(graphs=dummy_graphs)

    # 3. 验证基本属性
    print("\n[步骤 3] 验证数据集基本属性:")
    print(f"  数据集大小: {len(dataset)}")
    print(f"  类别数量: {dataset.get_num_classes()}")
    print(f"  节点特征维度: {dataset.get_num_node_features()}")

    # 4. 验证数据访问
    print("\n[步骤 4] 验证数据访问 (getitem):")
    for i in range(min(3, len(dataset))):
        graph = dataset[i]
        print(f"  图 {i+1}: x.shape={graph.x.shape}, edge_index.shape={graph.edge_index.shape}, y={graph.y.item()}")

    # 5. 测试 DataLoader 批处理
    print("\n[步骤 5] 测试 DataLoader 批处理...")
    from torch_geometric.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    batch_count = 0
    for batch in loader:
        batch_count += 1
        print(f"  批次 {batch_count}:")
        print(f"    包含图数量: {batch.num_graphs}")
        print(f"    总节点数: {batch.x.shape[0]}")
        print(f"    总边数: {batch.edge_index.shape[1]}")
        print(f"    标签: {batch.y.tolist()}")
        if batch_count >= 2:
            break

    # 6. 测试保存和加载
    print("\n[步骤 6] 测试保存和加载功能...")
    save_path = "/tmp/test_graph_dataset.pt"
    dataset.save(save_path)
    print(f"  数据集已保存到: {save_path}")

    loaded_dataset = GraphClassificationDataset(data_path=save_path)
    print(f"  重新加载数据集大小: {len(loaded_dataset)}")

    # 7. 测试结果总结
    print("\n" + "=" * 60)
    all_passed = (
        len(dataset) == num_graphs and
        dataset.get_num_node_features() == num_node_features and
        dataset.get_num_classes() == num_classes
    )

    if all_passed:
        print("✓ 所有测试通过！GraphClassificationDataset 类工作正常。")
    else:
        print("✗ 部分测试未通过，请检查代码。")
    print("=" * 60)

    return dataset


if __name__ == "__main__":
    test_graph_dataset()