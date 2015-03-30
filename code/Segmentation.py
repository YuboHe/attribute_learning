__author__ = 'wangyufei'
import PIL
from PIL import Image
import numpy as np
import os
from os import path

def read_bounding_box():
    path = 'CUB_200_2011/bounding_boxes.txt'
    with open(path, 'r') as f:
        bounding_box = []
        for line in f:
            x = [int(float(i)) for i in line.split()]
            x = x[1:]
            #print x
            bounding_box.append(x)
    #bounding_box = np.asarray(bounding_box)
    return bounding_box

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
def segment():
    image_list = read_image_list()
    bounding_boxes = read_bounding_box()
    image_path = 'CUB_200_2011/images/'
    output_path = 'CUB_200_2011/images_segmented/'
    for i in xrange(len(image_list)):
        bounding_box = bounding_boxes[i]
        print image_list[i]
        im = Image.open(image_path+image_list[i])
        if not os.path.exists(output_path + image_list[i].split('/')[0]):
            print output_path + image_list[i].split('/')[0]
            os.mkdir(output_path + image_list[i].split('/')[0])
        im_crop = im.crop((bounding_box[0], bounding_box[1], bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]))
        im_crop.save(output_path+image_list[i])

    pass
if __name__ == '__main__':
    segment()