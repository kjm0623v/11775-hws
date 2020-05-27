#!/bin/python 

import numpy
import os
from sklearn.cluster.k_means_ import KMeans
import _pickle as cPickle
import sys

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':

    surf_csv_file = 'select.surf.csv'
    cluster_num = 225
    output_file = 'kmeans.{}.model'.format(cluster_num)
    
    data = numpy.genfromtxt(surf_csv_file, delimiter=";")
    print(data.shape)
    model = KMeans(n_clusters=cluster_num, n_jobs=5)
    model.fit(data)
    cPickle.dump(model, open(output_file, 'wb'))

    print("K-means trained successfully!")
