#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/8/11 上午11:18
# @Author  : PeiP Liu
# @FileName: model_evaluation.py
# @Software: PyCharm

import time
import datetime
import torch
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


def time_format(time_diff):
    seconds = int(round(time_diff))
    return str(datetime.timedelta(seconds=seconds))


def bert_evaluate(eva_model, eva_dataloader, eva_epoch_th, eva_device, eva_dataset_name):
    eva_model.eval()
    all_pred_labels = []
    all_true_labels = []
    start = time.time()
    with torch.no_grad():
        for eva_batch in eva_dataloader:
            # we move the data to specific device
            eva_batch_data = tuple(item.to(eva_device) for item in eva_batch)
            # each batch_data contains several kinds of infor
            eva_input_ids, eva_atten_mask, eva_seg_ids, eva_tfidf_mask, eva_sent_labels = eva_batch_data
            # input to model to get the predicted result
            pred_labels_ids = eva_model(eva_input_ids, eva_atten_mask, eva_seg_ids, eva_tfidf_mask, eva_sent_labels, 'eva')

            all_pred_labels.extend(pred_labels_ids)
            all_true_labels.extend(eva_sent_labels.cpu().detach().tolist())

    assert len(all_true_labels) == len(all_pred_labels)

    end = time.time()
    print("This is %s:\n Epoch:%d\n Spending: %s" % \
          (eva_dataset_name,
           eva_epoch_th,
           time_format(end - start)))

    p, r, f1, acc = precision_score(all_true_labels, all_pred_labels, average='weighted'), recall_score(all_true_labels,
                                                                                    all_pred_labels, average='weighted'), f1_score(
        all_true_labels, all_pred_labels, average='weighted'), accuracy_score(all_true_labels, all_pred_labels)

    print("The accuracy is %.2f:\n The precision is %.2f:\n  The Recall is %.2f:\n  The F1 is %.2f:\n" % (
    acc * 100., p * 100., r * 100., f1 * 100.))
    return acc, p, r, f1


def test_predict(eva_model, eva_dataloader, eva_device):
    eva_model.eval()
    all_pred_labels = []
    with torch.no_grad():
        for eva_batch in eva_dataloader:
            # we move the data to specific device
            eva_batch_data = tuple(item.to(eva_device) for item in eva_batch)
            # each batch_data contains several kinds of infor
            eva_input_ids, eva_atten_mask, eva_seg_ids, eva_tfidf_mask, eva_sent_labels = eva_batch_data
            # input to model to get the predicted result
            pred_labels_ids = eva_model(eva_input_ids, eva_atten_mask, eva_seg_ids, eva_tfidf_mask, eva_sent_labels,
                                        'eva')

            all_pred_labels.extend(pred_labels_ids)

    return all_pred_labels