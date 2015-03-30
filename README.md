# code
- segmentation.py: original images -> cropped images (with only birds)
- cluster.py: after kmeans, clustered feature -> image patches from each of the attribute
- create_siamese_dataset.py: attribute location+images -> random training/test pairs
- extract_features.py: pretrained network -> certain layer features
- extract_features_percluster.py: trained network -> certain layer features from certarin attribute label

# features

*images:*
- predicted_attributes_large1/ : the image patches assigned for each of the attirbute label

*ipython notebook:*
- pca_check_30attribute.ipynb: check the features(feat_) all assigned for attribute#30 extracted fromm siamese network trained on attr#30 image pathes. Check if they are separated. 

*.npy:*
- caffe_ref_net/attribute_predicted.npy: imagenet pretrained network -> kmeans(k=100) -> assignment of conv5 features of every image to 100 attributes
- caffe_ref_net/conv4_features.npy: imagenet pretrained network -> conv4 features of each image (11788x384x13x13)
- caffe_ref_net/conv4_flattened.npy: flattened version of conv4_features.npy (384x1992172)
- caffe_ref_net/conv5_features.npy: imagenet pretrained network -> conv5 features of each image (11788x256x13x13)
- siamese_attr30/feat_reduction_features_30.npy: *feat* extracted from iter_1000 (siamese network trained on attr#30 image pathes)
- siamese_attr30/loc6_features_30.npy: *loc6* extracted from iter_1000
- siamese_attr30/feat_iter_109000_features_30.npy: *feat* extracted from iter_109000 (siamese network trained on attr#30 image pathes)
- siamese_attr30/loc6_iter_109000_features_30.npy: *loc6* extracted from iter_109000
- siamese_attr30/loc6_label_30.npy: label of image patches from attr#30. Order is the same with attribute_location/30.txt


*.txt:*
- train_attribute(_p).txt: information for training set of attr#30.
- test_attribute(_p).txt: information for test set of attr#30.
- attribute_location/*.txt : 100 text files, each of them consists of every image+location that falls into certain attribute label


# caffe
*path: local/caffe_siamese_attribute/examples/attribute*
- siamese_imagenet_train_val.prototxt: siamese architecture. relu5(p)->loc6(p)->contrastive loss
- siamese_imagenet_train_val2.prototxt: (now using) siamese architecture. relu5(p)->loc6(p)->feat(2-d)->contrastive loss
- siamese_imagenet_solver.prototxt/train_mnist_siamese.sh: corresponding stuff
- siamese_deploy.prototxt: siamese architecture with loss output
- siamese_imagenet_deploy.prototxt: imagenet architecture(input:data&loc) with loc6 output
- imagenet_deploy2.prototxt: imagenet architecture(input:data&loc) with feat output
- iter_*.caffemodel/solverstate: trained from siamese_imagenet_train_val2.prototxt
