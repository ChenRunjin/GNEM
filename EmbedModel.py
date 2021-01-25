import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import BertTokenizer, BertConfig, BertModel



class EmbedModel(nn.Module):
    def __init__(self, useful_field_num, device=0):
        super(EmbedModel, self).__init__()

        if not isinstance(device, list):
            device = [device]
        self.device = torch.device("cuda:{:d}".format(device[0]))

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.config = BertConfig.from_pretrained('bert-base-uncased')
        if torch.cuda.is_available() and len(device) > 1:
            self.model = nn.DataParallel(BertModel.from_pretrained('bert-base-uncased', config=self.config), device_ids=device)
        else:
            self.model = BertModel.from_pretrained('bert-base-uncased', config=self.config)
        for param in self.model.parameters():
            param.requires_grad = True
        self.dim = 768
        self.similarity_network = nn.Sequential(
            nn.Linear(2 * self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, 1)
        )


        self.field_num = useful_field_num

    def get_feature(self, sentences, center_sentence):
        """
        :param sentence
        :return: embedding
        """
        node_num = len(sentences)
        center_tokens = self.tokenizer.tokenize(center_sentence) + ["[SEP]"]
        tokens = [["[CLS]"] + self.tokenizer.tokenize(s) + ["[SEP]"] for s in sentences]
        lengths = [len(t) for t in tokens]
        center_length = len(center_tokens)
        max_len = max(lengths) + center_length


        input_ids = [self.tokenizer.convert_tokens_to_ids(t + center_tokens) for t in tokens]
        segment_ids = [[0] * l + [1] * center_length for l in lengths]
        input_masks = [[1] * (l + center_length) for l in lengths]

        for i in range(node_num):
            padding_len = max_len - len(input_ids[i])
            input_ids[i] += [0] * padding_len
            input_masks[i] += [0] * padding_len
            segment_ids[i] += [0] * padding_len

            assert len(input_ids[i]) == max_len
            assert len(input_masks[i]) == max_len
            assert len(segment_ids[i]) == max_len


        input_ids = torch.Tensor(input_ids).cuda().long()
        segment_ids = torch.Tensor(segment_ids).cuda().long()
        input_masks = torch.Tensor(input_masks).cuda().long()

        _ , pooled_output = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_masks)

        features = pooled_output

        return features


    def single_forward(self, example, max_node):
        attrs = []

        center_attr = " ".join(example["center"][1:])
        for node in example["neighbors"]:
            attr = node[1:]
            attrs.append(" ".join(attr))

        one_hop_nodes = len(attrs)

        features = self.get_feature(attrs, center_attr)

        num_nodes, fdim = features.shape

        N = num_nodes
        A_feat = torch.cat([features.repeat(1, N).view(N * N, -1), features.repeat(N, 1)], dim=1).view(N, -1, 2 * self.dim)
        A = self.similarity_network(A_feat).squeeze(2)
        A = F.softmax(A, dim=1)

        A_ = torch.zeros(max_node, max_node).to(self.device)
        A_[:num_nodes, :num_nodes] = A


        labels = example["labels"].copy()
        if "neighbors_mask" in example:
            mask = example["neighbors_mask"].copy()
        else:
            mask = [1] * len(example["labels"])


        assert len(labels) == one_hop_nodes, "labels len {:d} while only {:d} one_hop_nodes".format(len(labels), one_hop_nodes)

        labels += [-10] * (max_node - one_hop_nodes)
        mask += [0] * (max_node - one_hop_nodes)


        features = torch.cat([features, torch.zeros(max_node - num_nodes, fdim).to(self.device)], dim=0)

        return features, A_, labels, mask


    def forward(self, batch):
        feature = []
        A = []
        label = []
        mask = []

        max_node  = 0

        for ex in batch:
            if len(ex["neighbors"]) >  max_node:
                max_node = len(ex["neighbors"])

        for ex in batch:
            f, _A, l, m = self.single_forward(ex, max_node)
            feature.append(f)
            A.append(_A)
            label.append(l)
            mask.append(m)

        feature = torch.stack(tuple(feature), dim=0).to(self.device)
        A = torch.stack(tuple(A), dim=0).to(self.device)
        label = torch.Tensor(label).to(self.device)
        mask = torch.Tensor(mask).to(self.device)

        return feature, A, label, mask