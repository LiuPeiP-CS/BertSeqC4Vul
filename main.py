#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 #
# @Time    : 2022/9/25 8:19
# @Author  : PeiP Liu
# @Email   : peipliu@yeah.net
# @File    : DataReader.py
# @Software: PyCharm

import os
import copy
import operator
from arguments import Args as args
from Bert.Bert_main import bert_predict
from DataPro.data_prepro import *
from DataPro.data_utils import *
from DataPro.tfidf import *


stoplist = ['very', 'ourselves', 'am', 'me', 'just', 'her', 'ours', 'can', 'could', 'will', 'may',
            'because', 'is', 'it', 'only', 'in', 'such', 'too', 'their',
            'if', 'to', 'my', 'himself', 'after', 'why', 'while', 'can', 'each', 'itself', 'his', 'once',
            'herself', 'more', 'our', 'they', 'on', 'ma', 'them', 'its', 'where', 'did', 'll', 'you',
            'as', 'now', 'before', 'those', 'yours', 'from', 'who', 'was', 'm', 'been', 'will',
            'into', 'same', 'how', 'some', 'of', 'out', 'with', 's', 'being', 't', 'she', 'again', 'be',
            'by', 'have', 'yourselves', 'and', 'are', 'o', 'these', 'further', 'most', 'yourself',
            'having', 'here', 'he', 'were', 'but', 'this', 'myself', 'own', 'we', 'so', 'i', 'does', 'both',
            'when', 'between', 'd', 'had', 'the', 'y', 'has', 'than', 'whom',
            'should', 've', 'over', 'themselves', 'then', 'what', 'until', 'won', 'no', 'about',
            'any', 'that', 'for', 'do', 'there', 'an', 'or', 'hers',
            'above', 'a', 'at', 'your', 'theirs', 'other', 're', 'him', 'during', 'which', 'VersX']

if __name__ == '__main__':
    orig_train_data = read(args.train_addr)
    orig_test_data = read(args.test_addr)

    train_data_dict = copy.deepcopy(orig_train_data)
    test_data_dict = copy.deepcopy(orig_test_data)

    if args.REG:
        for train_cve_id, train_cve_detail in train_data_dict.items():
            train_cve_detail['description'] = regular(train_cve_detail['description'])
            train_data_dict[train_cve_id] = train_cve_detail

        for test_cve_id, test_cve_detail in test_data_dict.items():
            test_cve_detail['description'] = regular(test_cve_detail['description'])
            test_data_dict[test_cve_id] = test_cve_detail

    new_train_data_dict = copy.deepcopy(train_data_dict)
    for train_cve_id, train_cve_detail in new_train_data_dict.items():
        if args.DSTOP:
            train_cve_detail['description'] = del_stop(train_cve_detail['description'], stoplist)
        else:
            train_cve_detail['description'] = split_str(train_cve_detail['description'])
        new_train_data_dict[train_cve_id] = train_cve_detail

    new_test_data_dict = copy.deepcopy(test_data_dict)
    for test_cve_id, test_cve_detail in new_test_data_dict.items():
        if args.DSTOP:
            test_cve_detail['description'] = del_stop(test_cve_detail['description'], stoplist)
        else:
            test_cve_detail['description'] = split_str(test_cve_detail['description'])
        new_test_data_dict[test_cve_id] = test_cve_detail

    # print(orig_test_data)
    #   以上是，算法处理前的数据预处理   #

    word_tfidf = tfidf(
        [train_cve_detail['description'] for train_cve_detail in new_train_data_dict.values()] +
        [test_cve_detail['description'] for test_cve_detail in new_test_data_dict.values()]
    )

    # fabn = 0  # TF-IDF分类的异常样本

    test_cve_desc = [test_cve_detail['description'] for test_cve_detail in new_test_data_dict.values()]
    test_labels = len(test_cve_desc)*[-1]
    test_cve_desc_tfidf = [top_k(each_test_cve_desc, word_tfidf) for each_test_cve_desc in test_cve_desc]

    if not args.O4C: # 按子类进行分类
        for s1_type in ['privilege-required', 'attack-vector', 'impact']:
            train_cve_desc, s1_cve_label, s1_id2label = s1_cat_stat(s1_type, new_train_data_dict)
            if args.TFIDF: # 按照tf-idf找到最有效的词汇进行使用
                train_cve_desc_tfidf = []
                for each_train_cve_desc in train_cve_desc:
                    train_cve_desc_tfidf.append(top_k(each_train_cve_desc, word_tfidf))
                test_preds = bert_predict(args, s1_id2label, train_cve_desc, s1_cve_label, test_cve_desc, test_labels, train_cve_desc_tfidf, test_cve_desc_tfidf)

            else:
                test_preds = bert_predict(args, s1_id2label, train_cve_desc, s1_cve_label, test_cve_desc, test_labels)

            assert len(test_preds) == len(new_test_data_dict)

            for iter, (test_cve_id, test_cve_detail) in enumerate(new_test_data_dict.items()):
                test_cve_detail[s1_type] = s1_id2label[test_preds[iter]]
                new_test_data_dict[test_cve_id] = test_cve_detail

                # 测试列表中的第iter个元素，是不是正好对应相应的描述
                assert operator.eq(test_cve_desc[iter], test_cve_detail['description']) == True

        for level in range(2):
            level_types = []
            if level == 0: # impact的二级
                level_types = ['privileged-gained(rce)', 'information-disclosure']
            elif level == 1:
                level_types = ['information-disclosure_other-target(credit)', 'information-disclosure_local(credit)']
            for stype in level_types:
                s23_train_cve_desc, s23_train_cve_label, s23_train_id2label = s23_cat_stat(level, stype, new_train_data_dict)

                s23_train_cve_desc_tfidf = []
                for each_s23_train_cve_desc in s23_train_cve_desc:
                    s23_train_cve_desc_tfidf.append(top_k(each_s23_train_cve_desc, word_tfidf))

                if level == 0:
                    s23_test_cve_desc = []
                    s23_test_cve_desc_tfidf = []
                    for iter, test_cve_detail in enumerate(new_test_data_dict.values()):
                        if test_cve_detail['impact'] == stype: # 对预测出来的，需要进行再分类的，进行选择(包括tf-idf)
                            s23_test_cve_desc.append(test_cve_detail['description'])
                            s23_test_cve_desc_tfidf.append(top_k(test_cve_detail['description'], word_tfidf))
                            # s23_test_cve_desc.append(test_cve_desc[iter])
                            # s23_test_cve_desc_tfidf.append(test_cve_desc_tfidf[iter])
                    s23_test_cve_label = [-1] * len(s23_test_cve_desc_tfidf)

                    if args.TFIDF:
                        s23_test_preds = bert_predict(args, s23_train_id2label, s23_train_cve_desc, s23_train_cve_label, s23_test_cve_desc,
                                                      s23_test_cve_label, s23_train_cve_desc_tfidf, s23_train_cve_desc_tfidf)
                    else:
                        s23_test_preds = bert_predict(args, s23_train_id2label, s23_train_cve_desc, s23_train_cve_label, s23_test_cve_desc,
                                                      s23_test_cve_label)

                    # 将预测结果赋予到impact1上
                    test_pred_label_iter = 0
                    for test_cve_id, test_cve_detail in new_test_data_dict.items():
                        if test_cve_detail['impact'] == stype:  # 对预测出来的，需要进行再分类的，进行选择(包括tf-idf)
                            test_cve_detail['impact1'] = s23_train_id2label[s23_test_preds[test_pred_label_iter]]
                            new_test_data_dict[test_cve_id] = test_cve_detail
                            test_pred_label_iter = test_pred_label_iter + 1

                            assert operator.eq(s23_test_cve_desc[test_pred_label_iter], test_cve_detail['description']) == True

                    assert test_pred_label_iter == len(s23_test_preds)

                elif level == 1:
                    s23_test_cve_desc = []
                    s23_test_cve_desc_tfidf = []
                    for iter, test_cve_detail in enumerate(new_test_data_dict.values()):
                        if 'impact1' in test_cve_detail.keys() and test_cve_detail['impact1'] == stype:
                            s23_test_cve_desc.append(test_cve_detail['description'])
                            s23_test_cve_desc_tfidf.append(top_k(test_cve_detail['description'], word_tfidf))
                            # s23_test_cve_desc.append(test_cve_desc[iter])
                            # s23_test_cve_desc_tfidf.append(test_cve_desc_tfidf[iter])
                    s23_test_cve_label = [-1] * len(s23_test_cve_desc_tfidf)

                    if args.TFIDF:
                        s23_test_preds = bert_predict(args, s23_train_id2label, s23_train_cve_desc, s23_train_cve_label, s23_test_cve_desc,
                                                      s23_test_cve_label, s23_train_cve_desc_tfidf, s23_train_cve_desc_tfidf)
                    else:
                        s23_test_preds = bert_predict(args, s23_train_id2label, s23_train_cve_desc, s23_train_cve_label, s23_test_cve_desc,
                                                      s23_test_cve_label)

                    # 将预测结果赋予到impact1上
                    test_pred_label_iter = 0
                    for test_cve_id, test_cve_detail in new_test_data_dict.items():
                        if 'impact1' in test_cve_detail.keys() and test_cve_detail['impact1'] == stype:  # 对预测出来的，需要进行再分类的，进行选择(包括tf-idf)
                            test_cve_detail['impact2'] = s23_train_id2label[s23_test_preds[test_pred_label_iter]]
                            new_test_data_dict[test_cve_id] = test_cve_detail
                            test_pred_label_iter = test_pred_label_iter + 1

                            assert operator.eq(s23_test_cve_desc[test_pred_label_iter],
                                               test_cve_detail['description']) == True

                    assert test_pred_label_iter == len(s23_test_preds)

    else:
        # 全部同等看待
        for o_type in ['privilege-required', 'attack-vector', 'impact']:
            train_cve_desc, o_cve_label, o_id2label = o_stat_cat(o_type, new_train_data_dict)
            if args.TFIDF: # 按照tf-idf找到最有效的词汇进行使用
                train_cve_desc_tfidf = []
                for each_train_cve_desc in train_cve_desc:
                    train_cve_desc_tfidf.append(top_k(each_train_cve_desc, word_tfidf))
                test_preds = bert_predict(args, o_id2label, train_cve_desc, o_cve_label, test_cve_desc, test_labels, train_cve_desc_tfidf, test_cve_desc_tfidf)

            else:
                test_preds = bert_predict(args, o_id2label, train_cve_desc, o_cve_label, test_cve_desc, test_labels)

            assert len(test_preds) == len(new_test_data_dict)

            for iter, (test_cve_id, test_cve_detail) in enumerate(new_test_data_dict.items()):
                test_cve_detail[o_type] = o_id2label[test_preds[iter]]
                new_test_data_dict[test_cve_id] = test_cve_detail

    # 将cve description重新设置为原始信息
    for test_cve_id, new_test_cve_detail in new_test_data_dict.items():
        new_test_cve_desc = new_test_cve_detail['description']
        orig_test_cve_desc = orig_test_data[test_cve_id]['description']
        for each_word in new_test_cve_desc:
            if each_word not in orig_test_cve_desc:
                print(test_cve_id)

        new_test_cve_detail['description'] = orig_test_data[test_cve_id]['description']
        new_test_data_dict[test_cve_id] = new_test_cve_detail

    pred_test_addr = args.bresult + args.setting + '.json'
    with open(pred_test_addr, "w") as f:
        json.dump(new_test_data_dict, f, indent=4)

