#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/8/6 上午10:04
# @Author  : PeiP Liu
# @FileName: BertModel.py
# @Software: PyCharm

import torch
import torch.nn as nn
from torch.nn import LayerNorm as BertLayerNorm
import sys
sys.path.append("..")
import torch.nn.functional as F


class BERT_SC(nn.Module):
    def __init__(self, bert_model, idx2label, hidden_size=768, device='cpu'):
        super(BERT_SC, self).__init__()
        self.bert_model = bert_model
        self.hidden_size = hidden_size
        self.num_label = len(idx2label)
        self.device = device

        self.dropout = nn.Dropout(0.5)
        self.bert_sigmod = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

        self.hid2label = nn.Linear(self.hidden_size, self.num_label)
        # init the weight and bias of feature-emission layer
        nn.init.xavier_uniform_(self.hid2label.weight)
        nn.init.constant_(self.hid2label.bias, 0.0)
        self.apply(self.init_bert_weight)

    def init_bert_weight(self, module):
        # cf https://github.com/Louis-udm/NER-BERT-CRF/blob/master/NER_BERT_CRF.py
        # rf https://www.cnblogs.com/BlueBlueSea/p/12875517.html
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_bert_features(self, input_ids, seg_ids, atten_mask):
        # rf https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        outputs = self.bert_model(input_ids, token_type_ids=seg_ids,
                                  attention_mask=atten_mask, output_hidden_states=True, output_attentions=True)
        # last_hidden_states = outputs[0]
        last_hidden_states = outputs.last_hidden_state # (batch_size, seq_length, hidden_size)

        # pooler_outputs = outputs[1]
        # the feature of [CLS], and it represents the feature of whole sentence
        # We can better average or pool the sequence of hidden-states for the whole sequence.
        pooler_outputs = outputs.pooler_output  # (batch_size, hidden_size)

        return pooler_outputs, last_hidden_states  # the feature of [CLS] and all the tokens

    def tfidf_seq(self, bert_seq_features, tfidf_token_masks):
        batch_size, seq_len, feature_dim = bert_seq_features.shape
        seq_reps = torch.zeros((batch_size, feature_dim), dtype=torch.float32).to(self.device)
        for i_seq in range(batch_size):
            i_seq_feature = bert_seq_features[i_seq]
            i_seq_mask = tfidf_token_masks[i_seq]

            assert len(i_seq_feature) == len(i_seq_mask)

            extended_i_seq_mask = i_seq_mask.unsqueeze(1)  # (seq_len, 1)

            masked_seq_feature = i_seq_feature * extended_i_seq_mask  # (seq_len, feature_dim)

            i_seq_rep = masked_seq_feature.sum(0)  # 对所有tf-idf的token-embedding进行了sum-pooling

            seq_reps[i_seq] = i_seq_rep

        return seq_reps

    def forward(self, input_ids, input_mask, seg_ids, tfidf_token_masks, sents_labels, mode):
        bert_cls_features, bert_seq_features = self.get_bert_features(input_ids, seg_ids, input_mask)

        if tfidf_token_masks.count_nonzero().detach().item() == 0:  # 非0数据为0，即没有非0数据，即都是0。此时，我们使用CLS来表示序列信息
            seq_rep = self.bert_sigmod(self.dropout(bert_cls_features))
        else:
            seq_rep = self.tfidf_seq(bert_seq_features, tfidf_token_masks)  # 对于有tf-idf选择的词汇，我们使用tf-idf来表示序列特征
            seq_rep = self.bert_sigmod(self.dropout(seq_rep))

        class_result = self.softmax(self.hid2label(seq_rep))

        if mode == 'train':
            try:
                object_score = F.cross_entropy(class_result, sents_labels, ignore_index=2)
                return object_score
            except:
                print('There is something wrong with the prediction result!')
        else:
            return torch.argmax(class_result, dim=-1).detach().cpu().tolist()


