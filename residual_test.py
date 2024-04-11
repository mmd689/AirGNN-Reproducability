import torch
import argparse
import os
import pickle
from datasets import prepare_data
from model import Model
from tqdm import tqdm
from train_eval import evaluate
import numpy as np
from torch import tensor
from datasets import uniqueId, str2bool
import torch_geometric.transforms as T
from deeprobust.graph.data import Dpr2Pyg, Pyg2Dpr
from torch.distributions.multivariate_normal import MultivariateNormal
from datasets import index_to_mask

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--seed', type=int, default=15)
parser.add_argument('--model', type=str, default='AirGNN')
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--lambda_amp', type=float, default=0.1)
parser.add_argument('--lcc', type=str2bool, default=True)
parser.add_argument('--normalize_features', type=str2bool, default=True)
parser.add_argument('--random_splits', type=str2bool, default=False)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=100)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.8, help="dropout")
parser.add_argument('--K', type=int, default=10, help="the number of propagagtion in AirGNN")
parser.add_argument('--model_cache', type=str2bool, default=False)
parser.add_argument('--tune_lambda', type=str2bool, default=False)
parser.add_argument('--layers', type=int, default=64)

args = parser.parse_args()
print('arg : ', args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: in train_eval:", device)


def main():
    dataset, permute_masks = prepare_data(args, lcc=args.lcc)
    model = Model(dataset, args)

    dist = MultivariateNormal(torch.zeros(dataset.num_features), torch.eye(dataset.num_features))
    accs_noisy = []
    accs_non_noisy = []
    for run_k in range(10):
        model.to(device).reset_parameters()
        checkpointPath = "./model/lcc/{}_{}_best_model_run_{}.pth".format(args.dataset, args.model, run_k)
        print("checkpointPath:", checkpointPath)
        checkpoint = torch.load(checkpointPath)
        model.load_state_dict(checkpoint["model_state_dict"])

        dataset_ = dataset[0].clone()
        test_mask = dataset_.test_mask.nonzero().view(-1)
        perm = torch.randperm(test_mask.size()[0])
        test_mask = test_mask[perm]
        noisy = test_mask[: (test_mask.size()[0]) // 10]
        non_noisy = test_mask[(test_mask.size()[0]) // 10:]
        sample = dist.sample(noisy.size())
        dataset_.x[noisy] = sample
        noisy = index_to_mask(noisy, (dataset_.x.size())[0])
        non_noisy = index_to_mask(non_noisy, (dataset_.x.size())[0])

        model.eval()
        logits, score = model(dataset_)
        accs_noisy.append(torch.mean(score[noisy]).item())
        accs_non_noisy.append(torch.mean(score[non_noisy]).item())

    print(accs_noisy)
    print(accs_non_noisy)

    print(np.mean(accs_noisy))
    print(np.mean(accs_non_noisy))


if __name__ == "__main__":
    main()
