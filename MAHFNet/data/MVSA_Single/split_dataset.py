# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 19:54:24 2021

@author: ROG
"""

import numpy as np
import random
import os
import shutil
import math

label_file = './all_information_MVSA_Single.txt'

fh = open(label_file, 'r', encoding='utf-8')
imgs = []
label_name = {}
label_text = {}

_0 = []
_1 = []
_2 = []

for line in fh:
    line = line.rstrip()
    words = line.split()  # 以空格进行split
    # print(words)
    name = words[0]
    label = words[1]
    text = ''
    for p in words[2:]:
        text += p
        text += ' '  # 先加上空格还原最初的文本，放到bert中会被处理掉
    text = text.rstrip()  # 把末尾的空格去掉
    name = name.split("/")[-1]
    imgs.append((name, label, text))
    label_name[name] = label
    label_text[name] = text
    if label == '0':
        _0.append(name)

    if label == '1':
        _1.append(name)

    if label == '2':
        _2.append(name)

# train_set = _0[:math.floor(len(_0)*0.8)] +\
#             _1[:math.floor(len(_1)*0.8)] +\
#             _2[:math.floor(len(_2)*0.8)]
#
# test_set = _0[math.floor(len(_0)*0.8):math.floor(len(_0)*0.9)] +\
#            _1[math.floor(len(_1)*0.8):math.floor(len(_1)*0.9)] +\
#            _2[math.floor(len(_2)*0.8):math.floor(len(_2)*0.9)]
#
# valid_set = _0[math.floor(len(_0)*0.9):] + \
#            _1[math.floor(len(_1)*0.9):] + \
#            _2[math.floor(len(_2)*0.9):]

train_set = _0[:math.floor(len(_0)*0.8)] +\
            _1[:math.floor(len(_1)*0.8)] +\
            _2[:math.floor(len(_2)*0.8)]

test_set = _0[math.floor(len(_0)*0.8):math.floor(len(_0)*0.9)] +\
           _1[math.floor(len(_1)*0.8):math.floor(len(_1)*0.9)] +\
           _2[math.floor(len(_2)*0.8):math.floor(len(_2)*0.9)]

valid_set = _0[math.floor(len(_0)*0.9):] + \
           _1[math.floor(len(_1)*0.9):] + \
           _2[math.floor(len(_2)*0.9):]

random.shuffle(train_set)
random.shuffle(test_set)
random.shuffle(valid_set)

data_path = './MVSA_Single/MVSA_Single_all/'

with open('./train_0.9.txt', 'a', encoding='utf-8') as f:
    for name in train_set:
        f.write(data_path + name + '      ' + str(label_name[name]) + '      ' + label_text[name] + '\n')

with open('./test_0.1.txt', 'a', encoding='utf-8') as f:
    for name in test_set:
        f.write(data_path + name + '      ' + str(label_name[name]) + '      ' + label_text[name] + '\n')

with open('./valid_0.1.txt', 'a', encoding='utf-8') as f:
    for name in valid_set:
        f.write(data_path + name + '      ' + str(label_name[name]) + '      ' + label_text[name] + '\n')








