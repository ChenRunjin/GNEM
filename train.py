import os
import os.path as osp
import time
import torch
import torch.nn as nn
import numpy as np
import argparse
from dataset import MatchingDataset, collate_fn, MergedMatchingDataset
from torch.utils.data import DataLoader
from EmbedModel import EmbedModel
from GCN import gcn
from logger import set_logger
from torch.utils.tensorboard import SummaryWriter
from test import test as val
from utils import _read_csv, accuracy
from pytorch_transformers import AdamW, WarmupLinearSchedule


def tally_parameters(model):
    return sum([p.nelement() for p in model.parameters() if p.requires_grad])


def train(iter, dir, logger, tf_logger, model, embed_model, opt, crit, epoch_num, start_epoch=0, scheduler=None, test_iter=None, val_iter=None, log_freq=1, start_f1=None):
    p1=tally_parameters(embed_model)
    p2=tally_parameters(model)
    logger.info("Embed Model Parameter {}".format(p1))
    logger.info("Model Parameter {}".format(p2))
    logger.info("All Parameter {}".format(p1 + p2))

    step = 0
    if start_f1 is None:
        best_f1 = 0.0
    else:
        best_f1 = start_f1
    for i in range(start_epoch, epoch_num):
        model.train()
        embed_model.train()
        for j, batch in enumerate(iter):
            step += 1
            feature, A, label, masks = embed_model(batch)
            pred = model(feature, A)
            masks = masks.view(-1)
            label = label.view(-1)[masks == 1].long()
            pred = pred[masks == 1]
            loss = crit(pred, label)
            p, r, acc = accuracy(pred, label)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(embed_model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if scheduler:
                scheduler.step()
            if (j + 1) % log_freq == 0:
                logger.info(
                    'Train\tEpoch:[{:d}][{:d}/{:d}]\tLoss {:.3f}\tAccuracy {:.3f}\tPrecison {:.3f}\tRecall {:.3f}'.format(
                        i, j + 1, len(iter), loss, acc, p, r))
            if step % log_freq == 0:
                tf_logger.add_scalar('Train/Loss', loss.item(), step)
                tf_logger.add_scalar('Train/Precision', p, step)
                tf_logger.add_scalar('Train/Recall', r, step)
                tf_logger.add_scalar('Train/Accuracy', acc, step)

        if val_iter:
            f1s = val(iter=val_iter, logger=logger, tf_logger=tf_logger, model=model, embed_model=embed_model,prefix='Val',
                      crit=crit, test_step=i + 1, score_type=args.test_score_type)
            if max(f1s) > best_f1:
                best_f1 = max(f1s)
                best_type = args.test_score_type[f1s.index(best_f1)]
                state = {
                    "embed_model": embed_model.state_dict(),
                    "model": model.state_dict(),
                    "epoch": i + 1,
                    "type": best_type,
                    "val_f1":best_f1,
                }
                torch.save(state, os.path.join(dir, "best.pth"))
                logger.info("Val Best F1score\t{}\t{:.4f}".format(best_type, best_f1))
    if test_iter:
        checkpoint = torch.load("best.pth")
        embed_model.load_state_dict(checkpoint["embed_model"])
        model.load_state_dict(checkpoint["model"])
        embed_model = embed_model.to(embed_model.device)
        model = model.to(embed_model.device)
        best_epoch = checkpoint["epoch"]
        best_type = checkpoint["type"]
        valf1 = checkpoint["val_f1"]
        logger.info("load from epoch {:d}  f1score {:.4f}".format(best_epoch, valf1))
        f1s = val(iter=test_iter, logger=logger, model=model, embed_model=embed_model, prefix='Test',
                  crit=crit, score_type=[best_type])
        logger.info("Test F1score\tEpoch\t{:d}\t{}\t{:.4f}".format(best_epoch, best_type, f1s[0]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--exp_dir', default=".", type=str)
    parser.add_argument('--log_freq', default=1, type=int)
    parser.add_argument('--test_score_type', type=str, nargs='+')

    # Optimization args
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--embed_lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dropout', type=float,default=0.0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--pos_neg_ratio', default=1.0, type=float)

    # Training args
    parser.add_argument('--tableA_path', type=str)
    parser.add_argument('--tableB_path', type=str)
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--val_path', type=str)
    parser.add_argument('--checkpoint_path', type=str)

    # Device
    parser.add_argument('--gpu', type=int, default=[0,3], nargs='+')

    # Model
    parser.add_argument('--gcn_layer', default=1, type=int)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    tableA = _read_csv(args.tableA_path)
    tableB = _read_csv(args.tableB_path)
    useful_field_num = len(tableA.columns)-1
    gcn_dim = 768

    val_dataset = MergedMatchingDataset(args.val_path, tableA, tableB, other_path=[args.train_path, args.test_path])
    test_dataset = MergedMatchingDataset(args.test_path, tableA, tableB, other_path=[args.train_path, args.val_path])
    train_dataset = MatchingDataset(args.train_path, tableA, tableB)

    train_iter = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    val_iter = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    test_iter = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    embedmodel = EmbedModel(useful_field_num=useful_field_num,device=args.gpu)

    model = gcn(dims=[gcn_dim]*(args.gcn_layer + 1),  dropout=args.dropout)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in embedmodel.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.embed_lr},
        {'params': [p for n, p in embedmodel.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': args.embed_lr},
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': args.lr}
    ]

    num_train_steps = len(train_iter) * args.epochs
    opt = AdamW(optimizer_grouped_parameters, eps=1e-8)
    scheduler = WarmupLinearSchedule(opt, warmup_steps=0, t_total=num_train_steps)

    model_dir = args.exp_dir
    log_dir = os.path.join(args.exp_dir, "logs")
    tf_log_dir = os.path.join(args.exp_dir, "tf_logs")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(tf_log_dir):
        os.makedirs(tf_log_dir)

    logger = set_logger(os.path.join(log_dir, str(time.time()) + ".log"))
    tf_logger = SummaryWriter(tf_log_dir)

    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location="cuda:{}".format(args.gpu))
        if len(args.gpu) == 1:
            new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint["embed_model"].items()}
            embedmodel.load_state_dict(new_state_dict)
        else:
            embedmodel.load_state_dict(checkpoint["embed_model"])
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"]
        start_f1 = checkpoint["val_f1"]
        logger.info("load checkpoint from {}, start from epoch {:d}, best val f1 {:.4f}".format(args.checkpoint_path, start_epoch, start_f1))
    else:
        start_epoch = 0
        start_f1 = 0.0

    embedmodel = embedmodel.to(embedmodel.device)
    model = model.to(embedmodel.device)
    pos = 2.0 * args.pos_neg_ratio / (1.0 + args.pos_neg_ratio)
    neg = 2.0 / (1.0 + args.pos_neg_ratio)
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([neg, pos])).to(embedmodel.device)

    train(train_iter, model_dir, logger, tf_logger, model, embedmodel, opt, criterion, args.epochs, test_iter=test_iter, val_iter=val_iter,
          scheduler=scheduler, log_freq=args.log_freq, start_epoch=start_epoch, start_f1=start_f1)










