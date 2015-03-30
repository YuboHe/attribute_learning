__author__ = 'wangyufei'
import PIL
from PIL import Image
import numpy as np
import os
from os import path
import random
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

def read_image_list():
    path = 'CUB_200_2011/images.txt'
    with open(path, 'r') as f:
        image_list = []
        for line in f:
            x = line.split()
            x = x[1]
            image_list.append(x)
    print 'length of image_list: ', len(image_list)
    return image_list

def attribute_to_crop():
    imageset_root = 'CUB_200_2011/images_segmented/'
    save_CNN_root = '/home/feiyu1990/local/attributes/features/predicted_attributes_large1/'
    attribute_l = np.load('/home/feiyu1990/local/attributes/features/attribute_predicted.npy')
    #indice = np.load('/home/feiyu1990/local/attributes/features/indice.npy')
    image_list = read_image_list()
    i = 0

    for image_name in image_list:
        im = Image.open(imageset_root+image_name)
        temp = image_name.split('/')[1]
        temp1 = temp.split('.')[0]
        for h in xrange(13):
            for w in xrange(13):
                l = attribute_l[i]
                #if indice[i] == True:
                if not os.path.exists(save_CNN_root + str(l)):
                    os.makedirs(save_CNN_root + str(l))
                (width, height) = im.size
                im_crop = im.crop((max(int(width*(w-1)/13), 0), max(int(height*(h-1)/13), 0), min(int(width*(2+w)/13), width), min(int(height*(h+1)/13), height)))
                save_img_path = save_CNN_root + str(l)+'/' + image_name.split('/')[0] + temp1
                if (os.path.isfile(save_img_path+'.jpg')):
                    j = 1
                    while (os.path.isfile(save_img_path+str(j)+'.jpg')):
                        j = j + 1
                    save_img_path = save_img_path+str(j)

                im_crop.save(save_img_path + '.jpg')
                i = i + 1

def attribute_to_txt():
    imageset_root = '/home/feiyu1990/local/attributes/dataset/CUB_200_2011/images_segmented/'
    save_txt_pre = '/home/feiyu1990/local/attributes/features/attribute_location/'
    attribute_l = np.load('/home/feiyu1990/local/attributes/features/attribute_predicted.npy')
    #indice = np.load('/home/feiyu1990/local/attributes/features/indice.npy')
    image_list = read_image_list()
    for attribute_n in xrange(100):
        i = 0
        save_txt_path = save_txt_pre + str(attribute_n) + '.txt'
        f_write = open(save_txt_path, "wb")
        for image_name in image_list:
            for h in xrange(13):
                for w in xrange(13):
                    if attribute_l[i] == attribute_n:
                        print 'w = ', w, ' h = ', h, ' attribute_l[i] = ', attribute_l[i], ' attribute_n = ', attribute_n
                        string = imageset_root + image_name + ' ' + str(h * 13 + w) + '\n'
                        f_write.write(string)
                        print attribute_n, image_name
                    i = i + 1
        f_write.close()

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
