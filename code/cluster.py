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


'''read image list to array:image_list'''
def read_image_list():
    path = '/home/feiyu1990/local/attributes/dataset/CUB_200_2011/images.txt'
    with open(path, 'r') as f:
        image_list = []
        for line in f:
            x = line.split()
            x = x[1]
            image_list.append(x)
    print 'length of image_list: ', len(image_list)
    return image_list


'''for each of the attribute(100), create a txt file in which each line is
   an attribute location(1~169) + the name of the corresponding image'''
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


'''from each of the attribute, create the cropping and save it in save_CNN_root/i'''
def attribute_to_crop():
    imageset_root = '/home/feiyu1990/local/attributes/dataset/CUB_200_2011/images_segmented/'
    save_CNN_root = '/home/feiyu1990/local/attributes/features/predicted_attributes_large1/'
    if not os.path.exists(save_CNN_root):
        os.mkdir(save_CNN_root)
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
                im_crop = im.crop((max(int(width*(w-1)/13), 0), max(int(height*(h-1)/13), 0), min(int(width*(1+w)/13), width), min(int(height*(h+1)/13), height)))
                save_img_path = save_CNN_root + str(l)+'/' + image_name.split('/')[0] + temp1
                if (os.path.isfile(save_img_path+'.jpg')):
                    j = 1
                    while (os.path.isfile(save_img_path+str(j)+'.jpg')):
                        j = j + 1
                    save_img_path = save_img_path+str(j)

                im_crop.save(save_img_path + '.jpg')
                i = i + 1




if __name__ == '__main__':
    attribute_to_crop()
