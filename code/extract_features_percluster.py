__author__ = 'wangyufei'
import numpy as np
import sys
import random
import os
import optparse
from os.path import isfile, join
caffe_root = '/home/feiyu1990/local/caffe_siamese_attribute/'
sys.path.insert(1,caffe_root + 'python')
import caffe
from os import listdir
import cPickle
model_deploy = '/home/feiyu1990/local/caffe_siamese_attribute/examples/attribute/imagenet_deploy2.prototxt'
model_path = '/home/feiyu1990/local/caffe_siamese_attribute/examples/attribute/iter_109000.caffemodel'
#model_path = '/home/feiyu1990/local/caffe_0203/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

def load_network():
    print 'IMAGENET NETWORK'
    background_model = model_path
    net = caffe.Classifier(model_deploy,background_model,
                           mean=np.load(caffe_root+'python/caffe/imagenet/ilsvrc_2012_mean.npy'),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=[227,227])
    caffe.set_phase_test()
    caffe.set_mode_gpu()
    #net.set_mean('data',caffe_root+'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    #net.set_channel_swap('data',(2,1,0))
    #net.set_input_scale('data',255)
    return net

def read_image_list():
    path = '/home/feiyu1990/local/attributes/features/attribute_location/30.txt'
    with open(path, 'r') as f:
        image_list = []
        for line in f:
            x = line.split(' ')
            temp1 = x[0]
            temp2 = int(x[1].replace('\n', ''))
            image_list.append((x[0], x[1]))
    print 'length of image_list: ', len(image_list)
    return image_list

def extract_features(net, feature_names=['loc6', 'feat']):
    image_list = read_image_list()
    #imageset_root = 'CUB_200_2011/images_segmented/'

    print 'FEATURE EXTRACTION:'
    for feature_name in feature_names:
        mean = np.load(caffe_root+'/python/caffe/imagenet/ilsvrc_2012_mean.npy')
        save_CNN_root = '/home/feiyu1990/local/attributes/features/'+feature_name+'_'
        print '  -> extracting features %s to directory: %s ...' % (feature_name, save_CNN_root)
        [temp, n, a, b] = net.blobs[feature_name].data.shape
        loc = np.zeros((1,1,1,1))
        classes = []
        features = np.zeros((0, n, a, b))
        if not os.path.exists(save_CNN_root):
            os.makedirs(save_CNN_root)

        for image_pair in image_list:
            print image_pair
            temp1 = image_pair[0].split('/')[8]
            label = int(temp1.split('.')[0]) - 1
            input_image = caffe.io.load_image(image_pair[0])
            loc[0][0][0][0]  = image_pair[1]
            caffe_input = np.asarray([net.preprocess('data', input_image)])
            net.forward(data = caffe_input, location = loc)
            feature = net.blobs[feature_name].data.copy()
            print features.shape, feature.shape, loc[0][0][0][0]
            features = np.concatenate((features, feature), axis = 0)
            classes.append(label)
        print features.shape
        np.save(save_CNN_root+'iter_109000nC_features_30.npy', features)
        #np.save(save_CNN_root + 'label_30.npy', classes)

if __name__ == '__main__':
    net = load_network()
    extract_features(net)

