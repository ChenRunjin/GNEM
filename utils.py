import torch
import pandas  as pd
from sklearn.metrics import precision_score, recall_score


def _read_csv(path):
    columns = pd.read_csv(path).columns
    type = {}
    for name in columns:
        if name == 'id':
            continue
        type[name] = str
    data = pd.read_csv(path, dtype=type)
    data = data.fillna(" ")
    return data


def accuracy(pred, label):
    pred = torch.argmax(pred, dim=1).long()
    # pred = (pred > 0.0).long()
    acc = torch.mean((pred == label).float())
    pred = pred.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    p = precision_score(label, pred)
    r = recall_score(label, pred)
    return p,r,acc