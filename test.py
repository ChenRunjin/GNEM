import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
from dataset import collate_fn, MergedMatchingDataset
from torch.utils.data import DataLoader
from EmbedModel import EmbedModel
from GCN import gcn
from logger import set_logger
from utils import _read_csv, accuracy


def fetch_edge(batch):
    edges = []
    types = []
    for ex in batch:
        type = ex["type"]
        center_id = ex["center"][0]
        neighbors = []
        if "neighbors_mask" in ex:
            for i, n in enumerate(ex["neighbors"]):
                if ex["neighbors_mask"][i] == 0:
                    continue
                neighbors.append(n)
        else:
            neighbors = ex["neighbors"]
        if type == 'l':
            edges += [[center_id, n[0]] for n in neighbors]
            types += [0] * len(neighbors)
        elif type == 'r':
            edges += [[n[0], center_id] for n in neighbors]
            types += [1] * len(neighbors)
        else:
            raise NotImplementedError
    return edges, types


def calculate_f1(edges, scores, labels, types, score_type='left'):
    score_dict={}

    for i, edge in enumerate(edges):
        score = scores[i]
        label = labels[i]
        e = tuple(edge)
        if e in score_dict:
            assert score_dict[e][1] == label
            if score_type == 'max':
                score_dict[e] = (max(score_dict[e][0],score),label)
            elif score_type == 'mean':
                score_dict[e] = ((score_dict[e][0] + score) / 2.0, label)
            elif score_type == 'min':
                score_dict[e] = (min(score_dict[e][0], score), label)
            else:
                raise NotImplementedError
        else:
            score_dict[e] = (score,label)
    score_label = score_dict.values()
    scores = np.asarray([i[0] for i in score_label])
    label = np.asarray([i[1] for i in score_label])
    pred = (scores > 0.5).astype('int')


    TP = np.sum((pred == 1) * (label == 1))
    TN = np.sum((pred == 0) * (label == 0))
    FP = np.sum((pred == 1) * (label == 0))
    FN = np.sum((pred == 0) * (label == 1))
    acc = (TP + TN) * 1.0 / (TP + TN + FN + FP)
    if TP == 0:
        p = r = f1 =0.0
    else:
        p = TP * 1.0 / (TP + FP)
        r = TP * 1.0 / (TP + FN)
        f1 = 2 * p * r / (p + r)
    return p, r, f1, acc, score_dict


def test(iter,logger,model,embed_model,crit,test_step=None,tf_logger=None,score_type='mean', prefix='Test'):
    model.eval()
    embed_model.eval()

    edges = []
    scores = []
    labels = []
    types = []
    for j, batch in enumerate(iter):
        with torch.no_grad():
            edge,type = fetch_edge(batch)
            feature, A, label, masks = embed_model(batch)
            masks = masks.view(-1)
            label = label.view(-1)[masks == 1].long()
            pred = model(feature, A)
            pred = pred[masks == 1]
            loss = crit(pred, label)
            pred = F.softmax(pred, dim=1)
            p, r, acc = accuracy(pred, label)
            logger.info(
                '{}\t[{:d}/{:d}]\tLoss {:.3f}\tAccuracy {:.3f}\tPrecison {:.3f}\tRecall {:.3f}'.format(prefix,j+1,len(iter),loss,acc,
                                                                                                                      p, r))
            assert pred.shape[0] == label.shape[0]
            scores += pred[:,1].detach().cpu().numpy().tolist()
            edges += edge
            labels += label.detach().cpu().numpy().tolist()
            types += type

    edges = np.asarray(edges)
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    types = np.asarray(types)

    if not isinstance(score_type,list):
        score_type = [score_type]
    f1s = []
    for t in score_type:
        p, r, f1, acc, score_dict = calculate_f1(edges, scores, labels, types, score_type=t.lower())
        f1s.append(f1)
        logger.info('{}\t{}\tPrecison {:.3f}\tRecall {:.3f}\tF1-score {:.3f}\tAccuracy {:.3f}'.format(prefix, t, p, r, f1, acc))
        if tf_logger:
            tf_logger.add_scalar('{}/{}/Precision'.format(prefix, t), p, test_step)
            tf_logger.add_scalar('{}/{}/Recall'.format(prefix, t), r, test_step)
            tf_logger.add_scalar('{}/{}/f1Score'.format(prefix, t), f1, test_step)
            tf_logger.add_scalar('{}/{}/Accuracy'.format(prefix, t), acc, test_step)
    return f1s


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--score_type', type=str, nargs='+')

    # Test args
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--tableA_path', type=str)
    parser.add_argument('--tableB_path', type=str)
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--val_path', type=str)
    parser.add_argument('--checkpoint_path', type=str)

    # Device
    parser.add_argument('--gpu', type=int, default=[0], nargs='+')

    # Model
    parser.add_argument('--gcn_layer', default=1, type=int)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    tableA = _read_csv(args.tableA_path)
    tableB = _read_csv(args.tableB_path)
    useful_field_num = len(tableA.columns) - 1
    gcn_dim = 768

    test_dataset = MergedMatchingDataset(args.test_path, tableA, tableB, other_path=[args.train_path, args.val_path])


    test_iter = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    embedmodel = EmbedModel(useful_field_num=useful_field_num,device=args.gpu)


    model = gcn(dims=[gcn_dim]*(args.gcn_layer + 1))

    criterion = nn.CrossEntropyLoss().to(embedmodel.device)

    logger = set_logger()

    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path)
        if len(args.gpu) == 1:
            new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint["embed_model"].items()}
            embedmodel.load_state_dict(new_state_dict)
        else:
            embedmodel.load_state_dict(checkpoint["embed_model"])
        model.load_state_dict(checkpoint["model"])
        test_type = [checkpoint["type"]]
        logger.info("Test Type:\t{}".format(checkpoint["type"]))
    else:
        test_type = args.test_type

    embedmodel = embedmodel.to(embedmodel.device)
    model = model.to(embedmodel.device)

    test(iter=test_iter, logger=logger, model=model, embed_model=embedmodel, crit=criterion, score_type=test_type)



