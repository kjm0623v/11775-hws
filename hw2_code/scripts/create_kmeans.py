#!/bin/python
import numpy as np
import os
from tqdm import tqdm
import _pickle as cPickle
from sklearn.cluster.k_means_ import KMeans
import sys
# Generate k-means features for videos; each video is represented by a single vector

if __name__ == '__main__':

    cluster_num = 225
    kmeans_model = 'kmeans.{}.model'.format(cluster_num)
    file_list = 'list/all.video'

    # load the kmeans model
    kmeans = cPickle.load(open(kmeans_model,"rb"))
    
    fread = open(file_list, "r")
    
    for line in tqdm(fread.readlines()):
        surf_path = "surf/" + line.replace('\n','') + ".npy"
        out_path = "kmeans/" + line.replace('\n','') + ".npy"
        #fwrite = open(out_path, 'w')
        
        if os.path.exists(surf_path) == False:
            bow = np.ones(cluster_num) / cluster_num
        else:
            bow = np.zeros(cluster_num)
            array = np.load(surf_path)
            classes = kmeans.predict(array)
            
            for c in classes:
                bow[c] += 1
            bow = bow / len(classes)
            
        #print(sum(bow))
        np.save(out_path, bow)
           
    print("K-means features generated successfully!")
