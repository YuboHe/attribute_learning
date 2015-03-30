__author__ = 'wangyufei'

import PIL
from PIL import Image
import numpy as np
import os
from os import path
import random



def create_pair(first, second, similarity):
    temp1 = first.split(' ')
    temp2 = second.split(' ')
    first_paired = temp1[0] + ' ' + str(similarity) + ' ' + temp1[1]
    second_paired = temp2[0] + ' ' + str(similarity) + ' ' + temp2[1]
    pair = (first_paired, second_paired)
    return pair



def create_dataset():
    load_txt_pre = '/home/feiyu1990/local/attributes/features/attribute_location/'
    attribute_n = 30
    load_txt_path = load_txt_pre + str(attribute_n) + '.txt'
    f = open(load_txt_path)
    class_attribute = [[] for x in xrange(200)]

    for line in f:
        temp1 = line.split('/')[8]
        temp2 = int(temp1.split('.')[0]) - 1
        class_attribute[temp2].append(line)

    f.close()
    similar_pair = []
    for n in xrange(200):
        names = class_attribute[n]
        for i in xrange(len(names)):
            for j in xrange(i + 1, len(names)):
                similar_pair.append(create_pair(names[i], names[j], 1))

    all_attirbute = []
    for n in xrange(200):
        for i in xrange(len(class_attribute[n])):
            all_attirbute.append((class_attribute[n][i], n))


    disimilar_pair = []
    while len(disimilar_pair) < len(similar_pair):
        random.shuffle(all_attirbute)
        i = -2
        while len(disimilar_pair) < len(similar_pair) and i + 2 < len(all_attirbute) - 1:
            i = i + 2
            if (all_attirbute[i][1] != all_attirbute[i + 1][1]):
                disimilar_pair.append(create_pair(all_attirbute[i][0], all_attirbute[i + 1][0], 0))

    all_pair = similar_pair + disimilar_pair
    random.shuffle(all_pair)
    test_pair = all_pair[:int(len(all_pair)/100)]
    train_pair = all_pair[int(len(all_pair)/100):]
    print len(test_pair)
    print len(train_pair)

    save_txt1 = '/home/feiyu1990/local/attributes/features/train_attribute.txt'
    save_txt2 = '/home/feiyu1990/local/attributes/features/train_attribute_p.txt'
    f_write1 = open(save_txt1, "wb")
    f_write2 = open(save_txt2, "wb")
    for pair in train_pair:
        f_write1.write(pair[0])
        f_write2.write(pair[1])
    f_write1.close()
    f_write2.close()

    save_txt1 = '/home/feiyu1990/local/attributes/features/test_attribute.txt'
    save_txt2 = '/home/feiyu1990/local/attributes/features/test_attribute_p.txt'
    f_write1 = open(save_txt1, "wb")
    f_write2 = open(save_txt2, "wb")
    for pair in test_pair:
        f_write1.write(pair[0])
        f_write2.write(pair[1])
    f_write1.close()
    f_write2.close()

if __name__ == '__main__':
    create_dataset()
