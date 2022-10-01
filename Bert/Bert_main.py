#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/8/6 下午5:12
# @Author  : PeiP Liu
# @FileName: Bert_retrain.py
# @Software: PyCharm
import os
import torch
import time
import datetime
import argparse
from tqdm import trange
import sys
sys.path.append("..")
from Bert.BertModel import BERT_SC
from Bert.Bert_data_utils import DataProcessor, BertSCData
from transformers import AdamW, BertTokenizer, BertModel, BertConfig, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from Bert.evaluate import time_format, test_predict
from Bert.evaluate import bert_evaluate as evaluate


def warmup_linear(x, warmup = 0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


def bert_predict(args, idx2label, train_examples, train_examples_labels, test_examples, test_examples_labels, train_tfidf=None, test_tfidf=None):
    """
    :param args: 参数配置，来自于config原始文件
    :param idx2label: 标签到索引
    :param train_examples: 列表的列表，一层列表中的每个元素都是一个列表，次级列表中的每个元素都是一个单词；
    :param train_examples_labels: 列表中的每个元素都是一个标签数字，经过idx2label处理后的
    :param test_examples: 列表的列表，一层列表中的每个元素都是一个列表，次级列表中的每个元素都是一个单词；
    :param test_examples_labels: 列表中的每个元素都是一个标签数字，-1，没有实际价值
    :param train_tfidf: 训练样本的tf-idf关键词
    :param test_tfidf: 测试样本的tf-idf关键词
    :return:
    """

    output_dir = args.output_model_addr
    device = args.device

    batch_size = args.batch_size
    max_seq_len = args.max_seq_len
    learning_rate = args.learning_rate
    weight_decay_finetune = args.weight_decay_finetune
    total_train_epoch = args.total_train_epoch
    warmup_proportion = args.warmup_proportion
    gradient_accumulation_steps = args.gradient_accumulation_steps

    tokenizer = BertTokenizer.from_pretrained(args.input_model_addr + 'bert-base-uncased', do_lower_case=False)
    config = BertConfig.from_pretrained(args.input_model_addr + 'bert-base-uncased', output_hidden_states=True)
    bert_model = BertModel.from_pretrained(args.input_model_addr + 'bert-base-uncased', config=config)
    # the missing information will be filled by other file and function
    model = BERT_SC(bert_model, idx2label, device=device)

    # you should choose to start the training from scratch or from the previous
    start_epoch = 0
    valid_f1_pre = 0
    if not os.path.exists(output_dir + 'bert_sc.checkpoint.pt'):
        os.mknod(output_dir + 'bert_sc.checkpoint.pt')

    model.to(device=device)
    # we get all of the parameters of model
    params_list = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in params_list if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay_finetune},
        {'params': [p for n, p in params_list if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    # the total_train_steps referring to the loss computing
    total_train_steps = int(len(train_examples)/batch_size/gradient_accumulation_steps)*total_train_epoch
    warmup_steps = int(warmup_proportion*total_train_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_train_steps)

    # i_th step in all steps we have planed(from 0). And here, the ideal batch_size we want is batch_size*grad_acc_steps
    global_step_th = int(len(train_examples)/batch_size/gradient_accumulation_steps * start_epoch)

    train_dp = DataProcessor(train_examples, train_examples_labels, tokenizer, max_seq_len, train_tfidf)
    train_bert_sc_data = BertSCData(train_dp.get_features())

    train_average_loss = []
    train_ave_loss_pre = 1e6
    for epoch in trange(start_epoch, total_train_epoch, desc='Epoch'):
        train_loss = 0
        train_start = time.time()
        model.train()
        model.zero_grad()

        # shuffle=True means that RandomSampler,we can also get the train_dataloader with random by the following:
        # train_sampler = RandomSampler(train_bert_sc_data)
        # train_dataloader = DataLoader(dataset=train_bert_sc_data, sampler=train_sampler, batch_size=batch_size
        # , collate_fn=BertSCData.seq_tensor)
        train_dataloader = DataLoader(dataset=train_bert_sc_data, batch_size=batch_size, shuffle=True, collate_fn=BertSCData.seq_tensor)
        batch_start = time.time()
        for step, batch in enumerate(train_dataloader):
            # we show the time cost ten by ten batches
            if step % 10 == 0 and step != 0:
                print('Ten batches cost time : {}'.format(time_format(time.time()-batch_start)))
                batch_start = time.time()

            # input and output
            batch_data = tuple(cat_data.to(device) for cat_data in batch)
            train_input_ids, train_atten_mask, train_seg_ids, train_tfidf_mask, sents_labels = batch_data
            object_loss = model(train_input_ids, train_atten_mask, train_seg_ids, train_tfidf_mask, sents_labels, 'train')

            # loss regularization
            if gradient_accumulation_steps > 1:
                object_loss = object_loss / gradient_accumulation_steps

            object_loss.backward()
            train_loss = train_loss + object_loss.cpu().item()
            if (step+1) % gradient_accumulation_steps == 0:
                # this is to help prevent the "exploding gradient" problem. We have the L2 paradigm
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0, norm_type=2)
                # update the params
                optimizer.step()
                # updating learning rate schedule
                lr_scheduler.step()
                # clear the gradient
                model.zero_grad()
                # optimizer.zero_grad()
                # # crease the i_th step
                global_step_th = global_step_th + 1
            print("Epoch:{}-{}/{}, Object-loss:{}".format(epoch, step, len(train_dataloader), object_loss))
        ave_loss = train_loss / len(train_dataloader)
        train_average_loss.append(ave_loss)

        print("Epoch: {} is completed, the average loss is: {}, spend: {}".format(epoch, ave_loss, time_format(time.time()-train_start)))
        print("***********************Let us begin the validation of epoch {}******************************".format(epoch))

        valid_acc, valid_p, valid_r, valid_f1 = evaluate(model, train_dataloader, epoch, device, 'Valid') # acc, p, r, f1
        # if the model can achieve SOTA performance, we will save it
        if valid_f1 > valid_f1_pre:
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'valid_acc': valid_acc, 'valid_p': valid_p, 'valid_r': valid_r,
                        'valid_f1': valid_f1, 'max_seq_len': max_seq_len}, output_dir + 'bert_sc.checkpoint.pt')
            valid_f1_pre = valid_f1
        # if ave_loss < train_ave_loss_pre:
        #     torch.save({'epoch': epoch, 'model_state': model.state_dict()}, output_dir + 'bert_sc.checkpoint.pt')

    print("**********************************************\n"
          "********     The training is over.    ********\n"
          "**********************************************")

    # Next, we will test the model on stranger dataset
    # load the pretrained model
    checkpoint = torch.load(output_dir + 'bert_sc.checkpoint.pt', map_location='cpu')
    # parser the model params
    pretrained_model_dict = checkpoint['model_state']
    # get the model param names
    model_state_dict = model.state_dict()
    # get the params interacting between model_state_dict and pretrained_model_dict
    selected_model_state = {k: v for k, v in pretrained_model_dict.items() if k in model_state_dict}
    model_state_dict.update(selected_model_state)
    # load the params into model
    model.load_state_dict(model_state_dict)
    # show the details about loaded model
    # print('Loaded the pretrained BERT_SC model, epoch:', checkpoint['epoch'],
    #       'valid_p:', checkpoint['valid_p'], 'valid_r:', checkpoint['valid_r'], 'valid_f1:',checkpoint['valid_f1'])

    model.to(device)
    test_dp = DataProcessor(test_examples, test_examples_labels, tokenizer, max_seq_len, test_tfidf)
    test_bert_sc_data = BertSCData(test_dp.get_features())
    test_dataloader = DataLoader(dataset=test_bert_sc_data, batch_size=batch_size, collate_fn=BertSCData.seq_tensor)
    test_predictions = test_predict(model, test_dataloader, device)

    return test_predictions
