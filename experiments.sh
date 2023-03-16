python node_classification.py --dataset Cora --model gcn --prune_rate_graph .1855 --prune_rate_model .5904
python node_classification.py --dataset Cora --model gin --prune_rate_graph .05 --prune_rate_model .2
python node_classification.py --dataset Cora --model gat --prune_rate_graph .4596 --prune_rate_model .9313

python node_classification.py --dataset CiteSeer --model gcn --prune_rate_graph .4867 --prune_rate_model .945
python node_classification.py --dataset CiteSeer --model gin --prune_rate_graph .5819 --prune_rate_model .982
python node_classification.py --dataset CiteSeer --model gat --prune_rate_graph .5123 --prune_rate_model .956

python node_classification.py --dataset PubMed --model gcn --prune_rate_graph .5819 --prune_rate_model .9775
python node_classification.py --dataset PubMed --model gin --prune_rate_graph .4867 --prune_rate_model .9450
python node_classification.py --dataset PubMed --model gat --prune_rate_graph .5819 --prune_rate_model .9775

python link_prediction.py --dataset Cora --model gcn --prune_rate_graph .2649 --prune_rate_model .7379
python link_prediction.py --dataset Cora --model gin --prune_rate_graph .2262 --prune_rate_model .6723
python link_prediction.py --dataset Cora --model gat --prune_rate_graph .4312 --prune_rate_model .9141

python link_prediction.py --dataset CiteSeer --model gcn --prune_rate_graph .3366 --prune_rate_model .8322
python link_prediction.py --dataset CiteSeer --model gin --prune_rate_graph .2649 --prune_rate_model .7379
python link_prediction.py --dataset CiteSeer --model gat --prune_rate_graph .4312 --prune_rate_model .9141

python link_prediction.py --dataset PubMed --model gcn --prune_rate_graph .4013 --prune_rate_model .8926
python link_prediction.py --dataset PubMed --model gin --prune_rate_graph .1426 --prune_rate_model .488
python link_prediction.py --dataset PubMed --model gat --prune_rate_graph .5599 --prune_rate_model .9719
