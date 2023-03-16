python main.py --dataset Cora --model gcn --prune_rate_graph .1855 --prune_rate_model .5904 --task node_classification
python main.py --dataset Cora --model gin --prune_rate_graph .05 --prune_rate_model .2 --task node_classification
python main.py --dataset Cora --model gat --prune_rate_graph .4596 --prune_rate_model .9313 --task node_classification

python main.py --dataset CiteSeer --model gcn --prune_rate_graph .4867 --prune_rate_model .945 --task node_classification
python main.py --dataset CiteSeer --model gin --prune_rate_graph .5819 --prune_rate_model .982 --task node_classification
python main.py --dataset CiteSeer --model gat --prune_rate_graph .5123 --prune_rate_model .956 --task node_classification

python main.py --dataset PubMed --model gcn --prune_rate_graph .5819 --prune_rate_model .9775 --task node_classification
python main.py --dataset PubMed --model gin --prune_rate_graph .4867 --prune_rate_model .9450 --task node_classification
python main.py --dataset PubMed --model gat --prune_rate_graph .5819 --prune_rate_model .9775 --task node_classification

python main.py --dataset Cora --model gcn --prune_rate_graph .2649 --prune_rate_model .7379 --task link_prediction
python main.py --dataset Cora --model gin --prune_rate_graph .2262 --prune_rate_model .6723 --task link_prediction
python main.py --dataset Cora --model gat --prune_rate_graph .4312 --prune_rate_model .9141 --task link_prediction

python main.py --dataset CiteSeer --model gcn --prune_rate_graph .3366 --prune_rate_model .8322 --task link_prediction
python main.py --dataset CiteSeer --model gin --prune_rate_graph .2649 --prune_rate_model .7379 --task link_prediction
python main.py --dataset CiteSeer --model gat --prune_rate_graph .4312 --prune_rate_model .9141 --task link_prediction

python main.py --dataset PubMed --model gcn --prune_rate_graph .4013 --prune_rate_model .8926 --task link_prediction
python main.py --dataset PubMed --model gin --prune_rate_graph .1426 --prune_rate_model .488 --task link_prediction
python main.py --dataset PubMed --model gat --prune_rate_graph .5599 --prune_rate_model .9719 --task link_prediction
