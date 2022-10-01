#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/8/13 下午2:57
# @Author  : PeiP Liu
# @FileName: Bert_data_utils.py
# @Software: PyCharm

import torch
from torch.utils.data import Dataset


class InputFeature():
    def __init__(self, input_ids, input_mask, seg_ids, tfidf_token_mask, sent_label):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.seg_ids = seg_ids
        self.tfidf_token_mask = tfidf_token_mask
        self.sent_label = sent_label


class DataProcessor():
    def __init__(self, sentences, sentence_labels, tokenizer, max_seq_len, sents_tfidf=None):
        self.sentences = sentences # 列表的列表，一层列表中的每个元素都是一个列表，次级列表中的每个元素都是一个单词；
        self.sentence_labels = sentence_labels # 列表中的每个元素都是一个标签
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.sents_tfidf = sents_tfidf # 如果存在，形式同self.sentences

    def sentence2feature(self, sentence, sent_tfidf=None):

        tokens = ['[CLS]']  # the beginning of a sentence
        tfidf_token_mask = [0]  # when we get the valid tag, padding's tag should be ignored

        for i, i_word in enumerate(sentence):  # the sentence is original text, which not contains pad, bos and eos
            sub_tokens = self.tokenizer.tokenize(i_word)
            if not sub_tokens:
                sub_tokens = ['[UNK]']
            tokens.extend(sub_tokens)

            if sent_tfidf is not None and i_word in sent_tfidf:
                tfidf_token_mask.extend([1]*len(sub_tokens))
            else:
                tfidf_token_mask.extend([0]*len(sub_tokens))

        # truncating before filling
        if len(tokens) > self.max_seq_len-1:
            tokens = tokens[:self.max_seq_len-1]
            tfidf_token_mask = tfidf_token_mask[:self.max_seq_len-1]

        # filling
        tokens = tokens + ['[SEP]']
        tfidf_token_mask.append(0)
        input_mask = len(tokens) * [1]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(input_ids) < self.max_seq_len:
            tfidf_token_mask.append(0)
            input_mask.append(0)
            input_ids.append(0)
            # we also can use the following for padding
            # input_ids = pad_sequences(input_ids, maxlen=self.max_seq_len, dtype='long', value=0,
            # truncating='post', padding='post')

        seg_ids = self.max_seq_len * [0]

        assert len(input_ids) == self.max_seq_len
        assert len(input_mask) == self.max_seq_len
        assert len(tfidf_token_mask) == self.max_seq_len
        assert len(seg_ids) == self.max_seq_len

        return input_ids, input_mask, seg_ids, tfidf_token_mask

    def get_features(self):
        features = []
        if self.sents_tfidf: # 初始化对象时候已经给了信息
            for sentence, sent_tfidf, sent_label in zip(self.sentences, self.sents_tfidf, self.sentence_labels):
                ii, im, si, ttm = self.sentence2feature(sentence, sent_tfidf)
                features.append(InputFeature(ii, im, si, ttm, sent_label))
        else:
            for sentence, sent_label in zip(self.sentences, self.sentence_labels):
                ii, im, si, ttm = self.sentence2feature(sentence)
                features.append(InputFeature(ii, im, si, ttm, sent_label))
        return features


class BertSCData(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        feature = self.features[item]
        # every iteration will return a tuple, which contains the following:
        return feature.input_ids, feature.input_mask, feature.seg_ids, feature.tfidf_token_mask, feature.sent_label

    @classmethod
    def seq_tensor(cls, batch):  # the batch are results of batch_size __getitem__()
        # we also can use it for padding
        # padding = lambda x, max_seq_len: [feature[x]+(max_seq_len-len(feature[x]))*[0] for feature in batch]
        # input_ids = torch.tensor(padding(0, max_seq_len))

        list2tensor = lambda x: torch.tensor([feature[x] for feature in batch], dtype=torch.long)
        input_ids = list2tensor(0)
        input_mask = list2tensor(1)
        seg_ids = list2tensor(2)
        tfidf_token_mask = list2tensor(3)
        sents_labels = list2tensor(4)
        return input_ids, input_mask, seg_ids, tfidf_token_mask, sents_labels
