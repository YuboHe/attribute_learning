


# code/
- segmentation.py: original images -> cropped images (with only birds)
- cluster.py: after kmeans, clustered feature -> image patches from each of the attribute
- create_siamese_dataset.py: attribute location+images -> random training/test pairs
- extract_features.py: pretrained network -> certain layer features
- extract_features_percluster.py: trained network -> certain layer features from certarin attribute label

# features/

-- attribute_location/ : 100 text files, each of them consists of every image+location that falls into certain attribute label
- 
