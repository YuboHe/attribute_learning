# code
- segmentation.py: original images -> cropped images (with only birds)
- cluster.py: after kmeans, clustered feature -> image patches from each of the attribute
- create_siamese_dataset.py: attribute location+images -> random training/test pairs
- extract_features.py: pretrained network -> certain layer features
- extract_features_percluster.py: trained network -> certain layer features from certarin attribute label

# features

*folders:*
- attribute_location/ : 100 text files, each of them consists of every image+location that falls into certain attribute label
- predicted_attributes_large1/ : the image patches assigned for each of the attirbute label

*ipython notebook:*
- pca_check_30attribute.ipynb: check the features(feat_) all assigned for attribute#30 extracted fromm siamese network trained on attr#30 image pathes. Check if they are separated. 

*.npy:*
- attribute_predicted.npy:
- conv4_features.npy:
- conv4_flattened.npy:
- conv5_features.npy:
- 

- feat_reduction_features_30.npy: *feat* extracted from iter_1000 (siamese network trained on attr#30 image pathes)
- loc6_features_30.npy: *loc6* extracted from iter_1000
- feat_iter_109000_features_30.npy: *feat* extracted from iter_109000 (siamese network trained on attr#30 image pathes)
- loc6_iter_109000_features_30.npy: *loc6* extracted from iter_109000
- loc6_label_30.npy: label of image patches from attr#30. Order is the same with attribute_location/30.txt


*.txt:*
- train_attribute(_p).txt: information for training set of attr#30.
- test_attribute(_p).txt: information for test set of attr#30.


# caffe
