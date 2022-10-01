#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 #
# @Time    : 2022/9/25 10:00
# @Author  : PeiP Liu
# @Email   : peipliu@yeah.net
# @File    : tfidf.py
# @Software: PyCharm
import math
import operator
from collections import Counter
from collections import defaultdict


def tfidf(all_list_words):
    # all_list_words = exc_list_words + cat_list_words

    # compute the frequency of each word
    doc_frequency = defaultdict(int)
    for word_list in all_list_words:
        for word in word_list:
            doc_frequency[word] += 1

    # compute the TF of each word
    word_tf = {}
    for word in doc_frequency:
        word_tf[word] = doc_frequency[word] / sum(doc_frequency.values())

    # 计算每个词的IDF值
    doc_num = len(all_list_words)  # 文档总数。我们这里，以一个正样本文档去和所有的负样本文档进行集合，得到所有数据，以便来区别找出正样本的关键信息
    word_idf = {}  # 存储每个词的idf值
    word_doc = defaultdict(int)  # 存储包含该词的文档数
    for word in doc_frequency:
        for doc in all_list_words:
            if word in doc:
                word_doc[word] += 1
    for word in doc_frequency:
        word_idf[word] = math.log(doc_num / (word_doc[word] + 1))

    word_tfidf = {}
    for word in doc_frequency:
        word_tfidf[word] = word_idf[word] * word_tf[word]

    return word_tfidf


def top_k(words_list, word_tfidf, k=5):
    """
    :param words_list: 一个列表，列表中是原始词汇
    :param word_tfidf: 来自于all_tfidf(all_list_words)
    :param k: 选择top-k作为本描述的代表
    :return:
    """
    keywords = []  # 该文档的top-k词
    # 计算每个词的TF*IDF的值
    word_tf_idf = {}
    for word in words_list:
        word_tf_idf[word] = word_tfidf[word]

    # 对字典按值由大到小排序
    dict_feature_select = sorted(word_tf_idf.items(), key=operator.itemgetter(1), reverse=True)[:k]  # 每个文档的top-k个词
    for each_item in dict_feature_select:
        keywords.append(each_item[0])

    return keywords
