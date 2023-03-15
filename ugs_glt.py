import functools
import math
from dataclasses import dataclass, field
from random import randint
from typing import Dict, Any, Tuple, Type, Callable, Optional

import torch
import torch_geometric
import tqdm
from sklearn import metrics
from torch.nn import Module, Parameter
from torch.nn.functional import nll_loss
from torch.nn.init import trunc_normal_
from torch.optim import Adam, Optimizer
from torch_geometric.data import Data

from net import PrunedModel

EDGE_MASK = PrunedModel.EDGE_MASK + PrunedModel.MASK
INIT_FUNC = functools.partial(trunc_normal_, mean=1, a=1 - 1e-3, b=1 + 1e-3)


class GLTMask:
    def __init__(self, module: Module, graph: Data, device: torch.device, ignore_keys: Optional[set] = None) -> None:
        self.graph_mask = INIT_FUNC(
            torch.ones((graph.edge_index.shape[1] or graph.num_edges), device=device)
        )
        self.weight_mask = {
            param_name + PrunedModel.MASK: INIT_FUNC(torch.ones_like(param))
            for param_name, param in module.named_parameters()
            if param_name not in ignore_keys
        }

    def sparsity(self) -> Tuple[float, float]:
        norm_graph_mask = float(torch.count_nonzero(self.graph_mask))
        norm_graph = torch.numel(self.graph_mask)
        graph_sparsity = 1 - norm_graph_mask / norm_graph

        norm_weight_mask = 0
        norm_weight = 0

        for v in self.weight_mask.values():
            norm_weight_mask += float(torch.count_nonzero(v))
            norm_weight += torch.numel(v)

        weight_sparsity = 1 - norm_weight_mask / norm_weight
        return graph_sparsity, weight_sparsity

    def to_dict(self, weight_prefix=False) -> Dict[str, Any]:
        pref = "module." if weight_prefix else ""

        return {
            EDGE_MASK: self.graph_mask.detach().clone(),
            **{pref + k: v.detach().clone() for k, v in self.weight_mask.items()},
        }

    def load_and_binarise(
            self,
            model_masks: Dict[str, Parameter],
            p_theta: float,
            p_g: float,
    ) -> None:
        # Validation
        missing_masks = [
            name
            for name in [
                EDGE_MASK,
                *self.weight_mask.keys(),
            ]
            if name not in model_masks.keys()
        ]

        if len(missing_masks):
            raise ValueError(
                f"Model has no masks for the following parameters: {missing_masks}"
            )

        # splitting out m_g and m_theta
        graph_mask = model_masks[EDGE_MASK]
        del model_masks[EDGE_MASK]

        # process graph mask
        self.graph_mask = torch.where(
            self.graph_mask > 0, 1.0, 0.
        )  # needed to support non-binary inits
        all_weights_graph = graph_mask[self.graph_mask == 1]
        num_prune_graph = min(
            math.floor(p_g * len(all_weights_graph)), len(all_weights_graph) - 1
        )
        threshold_graph = all_weights_graph.sort()[0][num_prune_graph]
        self.graph_mask = torch.where(
            graph_mask > threshold_graph, self.graph_mask, 0.
        )

        # process weight masks
        self.weight_mask = {
            k: torch.where(v > 0, 1.0, 0.0) for k, v in self.weight_mask.items()
        }  # needed to support non-binary inits
        all_weights_model = torch.concat(
            [v[self.weight_mask[k] == 1] for k, v in model_masks.items()]
        )
        num_prune_weights = min(
            math.floor(p_theta * len(all_weights_model)), len(all_weights_model) - 1
        )
        threshold_model = all_weights_model.sort()[0][num_prune_weights]
        # pdb.set_trace()
        self.weight_mask = {
            k: torch.where(v > threshold_model, self.weight_mask[k], 0.0)
            for k, v in model_masks.items()
        }


@dataclass
class GLTSearch:
    module: Module
    graph: Data
    lr: float
    reg_graph: float
    reg_model: float
    optim_args: Dict[str, Any]
    task: str
    lr_mask_model: Optional[float] = None
    lr_mask_graph: Optional[float] = None
    optimizer: Type[Optimizer] = Adam
    sparsity: float = 0.99
    prune_rate_model: float = 0.2
    prune_rate_graph: float = 0.05
    max_train_epochs: int = 200
    loss_fn: Callable = nll_loss
    save_all_masks: bool = False
    seed: int = field(default_factory=lambda: randint(1, 9999))
    verbose: bool = False
    ignore_keys: Optional[set] = None

    def __post_init__(self):
        torch.manual_seed(self.seed)

        if not self.lr_mask_graph:
            self.lr_mask_graph = self.lr

        if not self.lr_mask_model:
            self.lr_mask_model = self.lr

        self.optim_args = {"lr": self.lr, **self.optim_args}
        self.ignore_keys = self.ignore_keys if self.ignore_keys is not None else set()

        self.mask = GLTMask(
            self.module,
            self.graph,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            ignore_keys=self.ignore_keys,
        )

    def prune(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        initial_params = {
            "module." + k + PrunedModel.ORIG if k.rpartition(".")[
                                                    -1] not in self.ignore_keys else "module." + k: v.detach().clone()
            for k, v in self.module.state_dict().items()
        }

        ticket = PrunedModel(self.module, self.graph, ignore_keys=self.ignore_keys)
        ticket.apply_mask(self.mask.to_dict())

        test_acc, masks = self.train(ticket, True)
        print("[UNREWOUND] Final test performance:", test_acc)
        self.mask.load_and_binarise(masks, self.prune_rate_model, self.prune_rate_graph)

        ticket.rewind(self.mask.to_dict(weight_prefix=True) | initial_params)
        test_acc, masks = self.train(ticket, False)

        print("[FIXED MASK] Final test performance:", test_acc)
        current_sparsity = self.mask.sparsity()
        print(
            "Graph sparsity:",
            round(current_sparsity[0], 4),
            "Model sparsity:",
            round(current_sparsity[1], 4),
        )

        return initial_params, self.mask.to_dict()

    def train(
            self, ticket: PrunedModel, ugs: bool
    ) -> Tuple[float, Dict[str, Parameter]]:
        best_val = 0.0
        final_test = 0.0
        best_masks = {}
        optimizer = self.optimizer(ticket.parameters(), **self.optim_args)

        with tqdm.trange(self.max_train_epochs, disable=not self.verbose) as t:
            for epoch in t:
                ticket.train()
                optimizer.zero_grad()

                output = ticket()

                if self.task == "node classification":
                    loss = self.loss_fn(
                        output[self.graph.train_mask], self.graph.y[self.graph.train_mask]
                    )
                elif self.task == "link prediction":
                    edge_mask = self.graph.train_mask[self.graph.edges[0]] & self.graph.train_mask[self.graph.edges[1]]
                    loss = self.loss_fn(
                        output[edge_mask], self.graph.edge_labels[edge_mask].float()
                    )
                else:
                    raise ValueError(f"{self.task} must be one of node class. or link pred.")

                if ugs:
                    for mask_name, mask in ticket.get_masks().items():
                        if mask_name.startswith("adj"):
                            loss += self.reg_graph * mask.norm(p=1)
                        else:
                            loss += self.reg_model * mask.norm(p=1)

                loss.backward()
                optimizer.step()

                ticket.eval()
                if self.task == "node classification":
                    preds = ticket().argmax(dim=1)
                    correct_val = (
                            preds[self.graph.val_mask] == self.graph.y[self.graph.val_mask]
                    ).sum()
                    val_acc = int(correct_val) / int(self.graph.val_mask.sum())

                    correct_test = (
                            preds[self.graph.test_mask] == self.graph.y[self.graph.test_mask]
                    ).sum()
                    test_acc = int(correct_test) / int(self.graph.test_mask.sum())

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        final_test = test_acc

                        if ugs:
                            best_masks = ticket.get_masks()

                    t.set_postfix(
                        {"loss": loss.item(), "val_acc": val_acc, "test_acc": test_acc}
                    )

                elif self.task == "link prediction":
                    edge_mask = self.graph.val_mask[self.graph.edges[0]] & self.graph.val_mask[self.graph.edges[1]]
                    preds = ticket()
                    val_preds = preds[edge_mask]
                    val_gt = self.graph.edge_labels[edge_mask].detach().numpy()
                    val_auc = metrics.auc(val_gt, torch.sigmoid(val_preds).detach().numpy())

                    edge_mask = self.graph.test_mask[self.graph.edges[0]] & self.graph.test_mask[self.graph.edges[1]]
                    test_preds = preds[edge_mask]
                    test_gt = self.graph.edge_labels[edge_mask].detach().numpy()
                    test_auc = metrics.auc(test_gt, torch.sigmoid(test_preds).detach().numpy())

                    if val_auc > best_val:
                        best_val_acc = val_auc
                        final_test = test_auc

                        if ugs:
                            best_masks = ticket.get_masks()

                    t.set_postfix(
                        {"loss": loss.item(), "val_auc": val_auc, "test_auc": test_auc}
                    )
                else:
                    raise ValueError(f"{self.task} must be one of node class. or link pred.")

        return final_test, best_masks

# TODO: mask + acc checkpoints
