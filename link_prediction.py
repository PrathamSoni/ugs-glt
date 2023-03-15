from argparse import ArgumentParser

from sklearn import metrics

import torch
import torch_geometric
import tqdm
from torch.nn import Module
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAT, GIN, GCN

from ugs_glt import GLTSearch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def baseline(model: Module, graph: Data, verbose: bool = False):
    initial_params = model.state_dict()
    best_val_auc = 0.0
    final_test_auc = 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-3, weight_decay=8e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    with tqdm.trange(200, disable=not verbose) as t:
        for epoch in t:
            model.train()
            optimizer.zero_grad()

            output = model(graph.x, graph.edge_index, edge_weight=graph.edge_weight, edges=graph.edges)
            edge_mask = graph.train_mask[graph.edges[0]] & graph.train_mask[graph.edges[1]]
            loss = loss_fn(
                output[edge_mask], graph.edge_labels[edge_mask].float()
            )

            loss.backward()
            optimizer.step()

            model.eval()
            preds = model(graph.x, graph.edge_index, edge_weight=graph.edge_weight, edges=graph.edges)
            edge_mask = graph.val_mask[graph.edges[0]] & graph.val_mask[graph.edges[1]]
            val_preds = preds[edge_mask]
            val_gt = graph.edge_labels[edge_mask].detach().numpy()
            val_auc = metrics.auc(val_gt, torch.sigmoid(val_preds).detach().numpy())

            edge_mask = graph.test_mask[graph.edges[0]] & graph.test_mask[graph.edges[1]]
            test_preds = preds[edge_mask]
            test_gt = graph.edge_labels[edge_mask].detach().numpy()
            test_auc = metrics.auc(test_gt, torch.sigmoid(test_preds).detach().numpy())

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                final_test_auc = test_auc

            t.set_postfix(
                {"loss": loss.item(), "val_auc": val_auc, "test_auc": test_auc}
            )

    model.load_state_dict(initial_params)
    print("[BASELINE] Final test accuracy:", final_test_auc)


def generate_edge_data(dataset):
    negative_edges = torch_geometric.utils.negative_sampling(dataset.edge_index)
    edge_labels = [0] * negative_edges.shape[-1] + [1] * dataset.edge_index.shape[-1]
    dataset.edges = torch.cat([dataset.edge_index, negative_edges], dim=-1)
    dataset.edge_labels = torch.tensor(edge_labels, device=device)


class Classifier(Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, edge_weight, edges):
        x = self.model(x, edge_index, edge_weight=edge_weight)
        edge_feat_i = x[edges[0]]
        edge_feat_j = x[edges[1]]
        return (edge_feat_i * edge_feat_j).sum(dim=-1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--model")
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    dataset = Planetoid(root=f"/tmp/{args.dataset}", name=args.dataset)
    print(args.dataset)
    print(args.model)
    data = dataset[0].to(device)
    generate_edge_data(data)

    print(f"Nodes: {data.num_nodes}")
    print(f"Edges: {data.num_edges}")

    hidden_channels = 512
    if args.model.lower() == "gcn":
        gnn = GCN(
            in_channels=dataset.num_node_features,
            output_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=2
        ).to(device)

    elif args.model.lower() == "gin":
        gnn = GIN(
            in_channels=dataset.num_node_features,
            output_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=2
        ).to(device)

    elif args.model.lower() == "gat":
        gnn = GAT(
            in_channels=dataset.num_node_features,
            output_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=2
        ).to(device)

    else:
        raise ValueError("model must be one of gcn, gin, gat")

    gnn = Classifier(gnn)
    # baseline(gnn, data, args.verbose)

    trainer = GLTSearch(
        task="link prediction",
        module=gnn,
        graph=data,
        lr=8e-3,
        reg_graph=0.01,
        reg_model=0.01,
        prune_rate_graph=0.05,
        prune_rate_model=0.8,
        optim_args={"weight_decay": 8e-5},
        seed=1234,
        verbose=args.verbose,
        ignore_keys={"eps"},
        max_train_epochs=200,
        loss_fn=torch.nn.BCEWithLogitsLoss()
    )

    initial_params, mask_dict = trainer.prune()
