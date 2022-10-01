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
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from BertModel import BERT_SC
from Bert_data_utils import DataProcessor, BertSCData
from transformers import AdamW, BertTokenizer, BertModel, BertConfig, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

import sys
sys.path.append("..")
from arguments import Args as args
from evaluate import time_format
from evaluate import bert_evaluate as evaluate

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


def warmup_linear(x, warmup = 0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


if __name__ == "__main__":
    output_dir = args.output_model_addr
    device = args.device
    label2idx = args.label2idx
    idx2label = args.idx2label
    train_examples = args.train_seq_list
    train_examples_labels = args.train_seq_label_list
    valid_examples = args.valid_seq_list
    valid_examples_labels = args.valid_seq_label_list
    test_examples = args.test_seq_list
    test_examples_labels = args.test_seq_label_list

    # we may change the value from static configuration to the dynamic
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=args.batch_size, type=int)
    parser.add_argument("--max_seq_len", default=args.max_seq_len, type=int)
    parser.add_argument("--learning_rate", default=args.learning_rate, type=float)
    parser.add_argument("--weight_decay_finetune", default=args.weight_decay_finetune, type=float)
    parser.add_argument("--total_train_epoch", default=args.total_train_epoch, type=int)
    parser.add_argument("--warmup_proportion", default=args.warmup_proportion, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=args.gradient_accumulation_steps, type=int)

    pargs = parser.parse_args()
    batch_size = pargs.batch_size
    max_seq_len = pargs.max_seq_len
    learning_rate = pargs.learning_rate
    weight_decay_finetune = pargs.weight_decay_finetune
    total_train_epoch = pargs.total_train_epoch
    warmup_proportion = pargs.warmup_proportion
    gradient_accumulation_steps = pargs.gradient_accumulation_steps

    tokenizer = BertTokenizer.from_pretrained(args.input_model_addr + '/bert-base-uncased', do_lower_case=False)
    config = BertConfig.from_pretrained(args.input_model_addr + '/bert-base-uncased', output_hidden_states=True)
    bert_model = BertModel.from_pretrained(args.input_model_addr + '/bert-base-uncased', config=config)
    # the missing information will be filled by other file and function
    model = BERT_SC(bert_model, label2idx, device=device)

    # you should choose to start the training from scratch or from the previous
    start_epoch = 0
    valid_p_pre = 0
    valid_r_pre = 0
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

    train_dp = DataProcessor(train_examples, train_examples_labels, tokenizer, max_seq_len)
    train_bert_sc_data = BertSCData(train_dp.get_features())
    valid_dp = DataProcessor(valid_examples, valid_examples_labels, tokenizer, max_seq_len)
    valid_bert_sc_data = BertSCData(valid_dp.get_features())

    train_average_loss = []
    valid_acc_score = []
    valid_f1_score = []
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

        valid_dataloader = DataLoader(dataset=valid_bert_sc_data, batch_size=batch_size, shuffle=True, collate_fn=BertSCData.seq_tensor)
        valid_acc, valid_p, valid_r, valid_f1 = evaluate(model, valid_dataloader, epoch, device, 'Valid') # acc, p, r, f1
        valid_acc_score.append(valid_acc)
        valid_f1_score.append(valid_f1)
        # if the model can achieve SOTA performance, we will save it
        if valid_f1 > valid_f1_pre:
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'valid_acc': valid_acc, 'valid_p': valid_p, 'valid_r': valid_r,
                        'valid_f1': valid_f1, 'max_seq_len': max_seq_len},
                       os.path.join(output_dir + 'bert_sc.checkpoint.pt'))
            valid_f1_pre = valid_f1

    print("**********************************************\n"
          "********     The training is over.    ********\n"
          "**********************************************")

    # then, we will show the training and validation processing by figure.
    # set the plot style from seaborn
    sns.set(style='darkgrid')
    # increase the plot size(line width) and figure size
    sns.set(font_scale=1.5)
    plt.rcParams['figure.figsize'] = [12, 6]

    x_label = np.arange(0,total_train_epoch)
    # plot the learning curve. the params are :values, color, line-title
    line1, = plt.plot(x_label, train_average_loss, color='b', label='train_average_loss')
    line2, = plt.plot(x_label, valid_acc_score, color='r', label='valid_acc_score')
    line3, = plt.plot(x_label, valid_f1_score, color='g', label='valid_f1_score')

    # now we label the plot
    plt.title('Learning curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Acc/F1')
    # legend() shows the line-title
    plt.legend(handles=[line1, line2, line3], labels=['train_average_loss','valid_acc_score', 'valid_f1_score'], loc='best')
    plt.savefig(output_dir + 'BERT_SC.jpg')
    plt.show()

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
    print('Loaded the pretrained BERT_SC model, epoch:', checkpoint['epoch'],
          'valid_acc:', checkpoint['valid_acc'], 'valid_p:', checkpoint['valid_p'], 'valid_r:', checkpoint['valid_r'], 'valid_f1:',checkpoint['valid_f1'])
    model.to(device)
    test_dp = DataProcessor(test_examples, test_examples_labels, tokenizer, max_seq_len)
    test_bert_sc_data = BertSCData(test_dp.get_features())
    test_dataloader = DataLoader(dataset=test_bert_sc_data, batch_size=batch_size, shuffle=True, collate_fn=BertSCData.seq_tensor)
    test_acc, test_p, test_r, test_f1 = evaluate(model, test_dataloader, checkpoint['epoch'], device, 'Test') # acc, p, r, f1

    output_metrics_file = os.path.join(output_dir, 'bert_metrics.text')
    fout_writer = open(output_metrics_file, 'w')
    fout_writer.write("********** Test Eval results **********\n")
    fout_writer.write('The Best Epoch is: ' + str(checkpoint['epoch'])) 
    fout_writer.write('\nTest_ACC: ' + str(test_acc))
    fout_writer.write('\nTest_P: ' + str(test_p))
    fout_writer.write('\nTest_R: ' + str(test_r))
    fout_writer.write('\nTest_F1: '+ str(test_f1))
    fout_writer.close()
