#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 #
# @Time    : 2022/9/25 8:29
# @Author  : PeiP Liu
# @Email   : peipliu@yeah.net
# @File    : data_utils.py
# @Software: PyCharm
from collections import defaultdict


def build_label_dict(cat_count):
    label2index = {}
    for label_id, type_name in enumerate(cat_count.keys()):
        label2index[type_name] = label_id
    index2label = {index: label for label, index in label2index.items()}
    print('The label dictionary is {}'.format(label2index))

    return label2index, index2label


def o_stat_cat(keynote, all_data):
    # 获取某keynotes(如privilege-required)下面每个类的所有文档词汇
    """
    :param keynote: which signal we use for get the cats, ['privilege-required', 'attack-vector', 'impact']
    :param all_data: the dataset_dict is from function read_json()
    :return: all the cats for the special keynote
    """
    cats = defaultdict(int) # 类型
    for cve_detail in all_data.values():
        cats[cve_detail[keynote]]+=1
    print('The sub-types statistics of '+ keynote + ' is:')
    print(cats)

    label2id, id2label = build_label_dict(cats)

    cve_desc = []
    cve_label = []
    for cve_detail in all_data.values():
        cve_desc.append(cve_detail['description'])
        cve_label.append(label2id[cve_detail[keynote]])

    return cve_desc, cve_label, id2label


def s1_cat_stat(keynote, all_data):
    # 针对impact的三层结构，本次按照第一层级处理
    # 获取某keynotes(如privilege-required)下面每个类的所有文档词汇
    """
    :param keynote: which signal we use for get the cats, ['privilege-required', 'attack-vector', 'impact']
    :param all_data: the dataset_dict is from function read_json()
    :return: all the cats for the special keynote
    """
    cats = defaultdict(int) # 类型
    for cve_detail in all_data.values():
        if keynote == 'impact':
            if 'privileged-gained(rce)' in cve_detail[keynote]:
                cats['privileged-gained(rce)'] += 1
            elif 'information-disclosure' in cve_detail[keynote]:
                cats['information-disclosure'] += 1
            else:
                cats[cve_detail[keynote]]+=1
        else:
            cats[cve_detail[keynote]] += 1
    print('The sub-types statistics of '+ keynote + ' is:')
    print(cats)

    label2id, id2label = build_label_dict(cats)

    cve_desc = []
    cve_label = []
    for cve_detail in all_data.values():
        cve_desc.append(cve_detail['description'])
        if keynote == 'impact':
            if 'privileged-gained(rce)' in cve_detail[keynote]:
                cve_label.append(label2id['privileged-gained(rce)'])
            elif 'information-disclosure' in cve_detail[keynote]:
                cve_label.append(label2id['information-disclosure'])
            else:
                cve_label.append(label2id[cve_detail[keynote]])
        else:
            cve_label.append(label2id[cve_detail[keynote]])

    return cve_desc, cve_label, id2label


def s23_cat_stat(level, keynote, all_data):
    """
    :param level: 正在处理第X个层级
    :param keynote: 该层级基于ltype进行再分类
    :param all_data: 所有的原始数据
    :return:
    """
    if level == 0:
        types = defaultdict(int)  # 类型
        for cve_detail in all_data.values():
            if keynote in cve_detail['impact']: # 筛选掉其他的第一级别
                if keynote == 'privileged-gained(rce)': # 其没有下一级
                    types[cve_detail['impact']] += 1
                    # print("This is a test for lstat_cat")
                elif keynote == 'information-disclosure': # 仍然存在下一级
                    if 'information-disclosure_other-target(credit)' in cve_detail['impact']: # 对下一级进行整合
                        types['information-disclosure_other-target(credit)'] += 1
                    elif 'information-disclosure_local(credit)' in cve_detail['impact']: # 对下一级进行整合
                        types['information-disclosure_local(credit)'] += 1
                    else:
                        types[cve_detail['impact']] += 1

        label2id, id2label = build_label_dict(types)
        cve_desc = []
        cve_label = []

        for cve_detail in all_data.values():
            if keynote in cve_detail['impact']: # 筛选掉其他的第一级别
                if keynote == 'privileged-gained(rce)': # 其没有下一级
                    cve_desc.append(cve_detail['description'])
                    cve_label.append(label2id[cve_detail['impact']])
                elif keynote == 'information-disclosure': # 仍然存在下一级
                    cve_desc.append(cve_detail['description'])
                    if 'information-disclosure_other-target(credit)' in cve_detail['impact']: # 对下一级进行整合
                        cve_label.append(label2id['information-disclosure_other-target(credit)'])
                    elif 'information-disclosure_local(credit)' in cve_detail['impact']: # 对下一级进行整合
                        cve_label.append(label2id['information-disclosure_local(credit)'])
                    else:
                        cve_label.append(label2id[cve_detail['impact']])

        return cve_desc, cve_label, id2label

    elif level == 1:
        types = defaultdict(int)  # 类型
        for cve_detail in all_data.values():
            if keynote in cve_detail['impact']: # 筛选掉其他的第一级别
                types[cve_detail['impact']] += 1

        label2id, id2label = build_label_dict(types)
        cve_desc = []
        cve_label = []

        for cve_detail in all_data.values():
            if keynote in cve_detail['impact']: # 筛选掉其他的第一级别
                cve_desc.append(cve_detail['description'])
                cve_label.append(label2id[cve_detail['impact']])

        return cve_desc, cve_label, id2label



