from functools import partial
from typing import Any, List, Dict

import torch
from torch.nn import Module, Parameter
from torch_geometric.data import Data


class PrunedModel(Module):
    MASK = "_mask"
    MASK_FIXED = "_mask_fixed"
    ORIG = "_orig"
    EDGE_MASK = "adj"

    def __init__(
            self,
            module: Module,
            graph: Data,
            testing=False,
            ignore_keys=None
    ):
        super().__init__()

        self.module = module
        self.graph = graph
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.testing = testing
        self.ignore_keys = ignore_keys

    @property
    def model(self) -> Module:
        return self.module

    def get_params(self) -> List[Parameter]:
        return [param for param_name, param in self.named_parameters(recurse=True)]

    def get_masks(self) -> Dict[str, Parameter]:
        return {
            param_name.removeprefix("module."): param
            for param_name, param in self.named_parameters(recurse=True)
            if param_name.endswith(PrunedModel.MASK)
        }

    def rewind(self, state_dict: Dict[str, Any]) -> None:
        for k, v in state_dict.items():
            module_path, _, param_name = k.rpartition(".")
            # if param_name in self.ignore_keys:
            #     continue
            sub_mod = self.get_submodule(module_path)
            curr_param = getattr(sub_mod, param_name)
            curr_param.data = v

            if param_name.endswith(PrunedModel.MASK):
                curr_param.requires_grad = False
                curr_param.grad = None

            setattr(sub_mod, param_name, curr_param)

    def apply_mask(self, mask_dict: Dict[str, Any]) -> None:
        # Input validation
        param_names = [
                          name
                          for name, _ in self.module.named_parameters()
                          if not PrunedModel._is_injected_parameter(name) and name not in self.ignore_keys
                      ] + [PrunedModel.EDGE_MASK]
        missing_keys = [
            name
            for name in param_names
            if name + PrunedModel.MASK not in mask_dict.keys()
        ]

        if len(missing_keys):
            raise ValueError(f"Masks for {missing_keys} missing from mask_dict!")

        # Injecting module masks
        for param_path in self._get_parameter_paths():
            module_path, _, param_name = param_path.rpartition(".")
            if param_name in self.ignore_keys:
                continue
            sub_mod = self.module.get_submodule(module_path)

            weight = getattr(sub_mod, param_name)
            del sub_mod._parameters[param_name]
            mask = mask_dict[param_path + PrunedModel.MASK]

            sub_mod.register_parameter(
                param_name + PrunedModel.MASK,
                Parameter(mask.data),
            )
            sub_mod.register_parameter(
                param_name + PrunedModel.ORIG,
                Parameter(weight.data),
            )

            sub_mod.register_forward_pre_hook(
                partial(
                    PrunedModel._hook,
                    param_name=param_name,
                    testing=self.testing,
                )
            )

        adj_mask_name = PrunedModel.EDGE_MASK + PrunedModel.MASK
        self.register_parameter(adj_mask_name, Parameter(mask_dict[adj_mask_name].data))
        self.register_buffer(
            PrunedModel.EDGE_MASK + PrunedModel.MASK_FIXED,
            torch.ones_like(mask_dict[adj_mask_name]),
        )
        self.register_forward_pre_hook(PrunedModel._compute_masked_adjacency)

    def forward(self):
        adj_mask = getattr(self, PrunedModel.EDGE_MASK)

        if not isinstance(adj_mask, torch.Tensor):
            raise RuntimeError("invalid adjacency mask!")

        pruned_graph = self.graph.clone()
        pruned_graph.edge_weight = adj_mask
        if hasattr(pruned_graph, "edges"):

            return self.module(
                pruned_graph.x, pruned_graph.edge_index, edge_weight=pruned_graph.edge_weight, edges=pruned_graph.edges
            )
        else:
            return self.module(
                pruned_graph.x, pruned_graph.edge_index, edge_weight=pruned_graph.edge_weight
            )

    @staticmethod
    def _hook(sub_mod: Module, input: Any, param_name: str, testing: bool) -> None:
        setattr(
            sub_mod,
            param_name,
            PrunedModel._compute_masked_param(sub_mod, param_name, testing=testing),
        )

    @staticmethod
    def _compute_masked_param(sub_mod: Module, param_name: str, testing: bool):
        mask = getattr(sub_mod, param_name + PrunedModel.MASK)
        param = getattr(sub_mod, param_name + PrunedModel.ORIG)

        masked_param = param * mask

        if testing:
            masked_param.retain_grad()

        return masked_param

    @staticmethod
    def _compute_masked_adjacency(sub_mod: Module, input: Any):
        mask = getattr(sub_mod, PrunedModel.EDGE_MASK + PrunedModel.MASK)
        param = getattr(sub_mod, PrunedModel.EDGE_MASK + PrunedModel.MASK_FIXED)
        masked_param = torch.relu(param * mask)
        setattr(sub_mod, PrunedModel.EDGE_MASK, masked_param)

    @staticmethod
    def _is_injected_parameter(param_name: str):
        return (
                param_name.endswith(PrunedModel.MASK)
                or param_name.endswith(PrunedModel.ORIG)
                or param_name.endswith(PrunedModel.MASK_FIXED)
        )

    def _get_parameter_paths(self):
        return [
            param_path
            for param_path, _ in self.module.named_parameters(recurse=True)
            if not self._is_injected_parameter(param_path)
        ]


class LinkPredictor(Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, edge_weight, edges):
        x = self.model(x, edge_index, edge_weight=edge_weight)
        edge_feat_i = x[edges[0]]
        edge_feat_j = x[edges[1]]
        return (edge_feat_i * edge_feat_j).sum(dim=-1)
