#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/9/22 下午3:52
# @Author  : PeiP Liu
# @FileName: Bert_retrain.py
# @Software: PyCharm
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

root_addr = '/data/liupeipei/paper/BertSeqC4Vul/'

class Args:
    input_model_addr = root_addr + 'OBertModel/'

    train_addr = root_addr + 'Input/train.json'
    test_addr = root_addr + 'Input/test_a.json'
    bresult = root_addr + 'BResult/'
    rresult = root_addr + 'RResult/'

    REG = True # True表示对输入的原始文本进行正则化
    DSTOP = True # True表示对输入的原始文本删除停用词
    TFIDF = True # True表示使用TFIDF获取的词作为seq的表示
    O4C = False # True表示分类时，利用全部内容去分类；False表示分类时，根据小类去分类

    setting = str(int(REG)) + str(int(DSTOP)) + str(int(TFIDF)) + str(int(O4C))

    output_model_addr = root_addr + 'TBertModel/'+ setting + '__'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size = 4
    max_seq_len = 256
    learning_rate = 5e-4
    weight_decay_finetune = 0.001
    total_train_epoch = 40
    warmup_proportion = 0.002
    gradient_accumulation_steps = 20

