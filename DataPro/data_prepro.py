#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 #
# @Time    : 2022/9/25 9:54
# @Author  : PeiP Liu
# @Email   : peipliu@yeah.net
# @File    : data_prepro.py
# @Software: PyCharm
import re
import json
from collections import OrderedDict


def read(file_addr):
    dataset_dict = OrderedDict()
    with open(file_addr, 'r', encoding='utf-8') as fr:
        content = fr.readlines()
        for line in content:
            if line.strip() is not '':
                dict = json.loads(line)
                dataset_dict[dict['cve-number']] = dict
    return dataset_dict  # {cve-number: cve-number, description, privilege-required, attack-vector, impact}


def regular(str_input):
    # this is only for the pre-trained language model
    """
    :param str_input: the input is a string from the vul description
    :return: we replace the irregular substr with some words
    """
    new_string = re.sub(r"([0-9A-Za-z]+[\.|-])+[0-9A-Za-z]*", "VersX", str_input)
    return new_string


def split_str(input_str):
    return input_str.replace(';', ' ').replace(',', ' ').replace('\"', ' ').replace("\'", " ").replace('(', ' ').replace(')', ' ').strip().split()


def del_stop(input_str, stoplist):
    word_list = split_str(input_str)
    # print(word_list)
    re_put = []
    for word in word_list:
        if word not in stoplist:  # 去除停用词
            # word_list.remove(word)
            re_put.append(word)

    # re_put = word_list*1
    # print(re_put)
    return re_put