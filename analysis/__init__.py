"""
GNN 可解释性分析模块
支持 5 种节点归因方法：GradCAM, GNNExplainer, PGExplainer, GraphMASK, GraphLIME
"""

from .node_attribution import (
    enrich_data_object,
    load_finetune_model,
    GradCAMHook,
    compute_node_scores_single,
    compute_node_scores_batch,
    NodeAttributionAnalyzer
)

from .edge_attribution import (
    get_edge_type,
    annotate_edge_types,
    compute_edge_importance,
    EdgeAttributionAnalyzer
)

from .gnnexplainer_attribution import (
    compute_gnnexplainer_single,
    compute_gnnexplainer_batch
)

from .pgexplainer_attribution import (
    PGExplainerModel,
    PGExplainerTrainer,
    compute_pgexplainer_batch
)

from .graphmask_attribution import (
    GraphMASKPolicy,
    GraphMASKExplainer,
    compute_graphmask_batch
)

from .graphlime_attribution import (
    GraphLIMEExplainer,
    compute_graphlime_single,
    compute_graphlime_batch
)