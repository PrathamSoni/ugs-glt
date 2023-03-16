from argparse import ArgumentParser

import torch
import tqdm
from torch.nn import Module
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAT, GIN, GCN

from ugs_glt import GLTSearch


def baseline(model: Module, graph: Data, verbose: bool = False):
    initial_params = model.state_dict()
    best_val_acc = 0.0
    final_test_acc = 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-3, weight_decay=8e-5)
    loss_fn = torch.nn.functional.cross_entropy

    with tqdm.trange(200, disable=not verbose) as t:
        for epoch in t:
            model.train()
            optimizer.zero_grad()

            output = model(graph.x, graph.edge_index, edge_weight=graph.edge_weight)
            loss = loss_fn(output[graph.train_mask], graph.y[graph.train_mask])

            loss.backward()
            optimizer.step()

            model.eval()
            preds = model(graph.x, graph.edge_index, edge_weight=graph.edge_weight).argmax(dim=1)
            correct_val = (preds[graph.val_mask] == graph.y[graph.val_mask]).sum()
            val_acc = int(correct_val) / int(graph.val_mask.sum())

            correct_test = (preds[graph.test_mask] == graph.y[graph.test_mask]).sum()
            test_acc = int(correct_test) / int(graph.test_mask.sum())

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc

            t.set_postfix(
                {"loss": loss.item(), "val_acc": val_acc, "test_acc": test_acc}
            )

    model.load_state_dict(initial_params)
    print("[BASELINE] Final test accuracy:", final_test_acc)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-d', "--dataset", default="Cora")
    parser.add_argument('-m', "--model", default="gcn")
    parser.add_argument("--reg_graph", default=.001, type=float)
    parser.add_argument("--reg_model", default=.001, type=float)
    parser.add_argument("--prune_rate_graph", default=.05, type=float)
    parser.add_argument("--prune_rate_model", default=.8, type=float)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Planetoid(root=f"/tmp/{args.dataset}", name=args.dataset)
    print(args.dataset)
    print(args.model)
    data = dataset[0].to(device)

    print(f"Nodes: {data.num_nodes}")
    print(f"Edges: {data.num_edges}")

    if args.model.lower() == "gcn":
        gnn = GCN(
            in_channels=dataset.num_node_features,
            output_channels=dataset.num_classes,
            hidden_channels=512,
            num_layers=2
        ).to(device)

    elif args.model.lower() == "gin":
        gnn = GIN(
            in_channels=dataset.num_node_features,
            output_channels=dataset.num_classes,
            hidden_channels=512,
            num_layers=2
        ).to(device)

    elif args.model.lower() == "gat":
        gnn = GAT(
            in_channels=dataset.num_node_features,
            output_channels=dataset.num_classes,
            hidden_channels=512,
            num_layers=2
        ).to(device)

    else:
        raise ValueError("model must be one of gcn, gin, gat")

    baseline(gnn, data, args.verbose)

    trainer = GLTSearch(
        task="node classification",
        module=gnn,
        graph=data,
        lr=8e-3,
        reg_graph=args.reg_graph,
        reg_model=args.reg_model,
        prune_rate_graph=args.prune_rate_graph,
        prune_rate_model=args.prune_rate_model,
        optim_args={"weight_decay": 8e-5},
        seed=1234,
        verbose=args.verbose,
        ignore_keys={"eps"},
        max_train_epochs=200,
        loss_fn=torch.nn.functional.cross_entropy
    )

    initial_params, mask_dict = trainer.prune()
