#!/bin/python

import numpy as np
import os
from sklearn.svm.classes import SVC
import cPickle
import sys

# Apply the SVM model to the testing videos; Output the score for each video

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print "Usage: {0} model_file feat_dir feat_dim output_file".format(sys.argv[0])
        print "model_file -- path of the trained svm file"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features; provided just for debugging"
        print "output_file -- path to save the prediction score"
        exit(1)

    model_file = sys.argv[1]#para
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]
    event_name = model_file.split('.')[-2]

    fwrite = open(output_file, 'w')

    # data loading
    val_list = open('../all_val.lst', 'r')
    test_list = open('../all_test_fake.lst', 'r')
    val_x = []
    for file in val_list.readlines():
        filename = file[:-1].split(' ')[0]
        data_path = feat_dir + filename + '.feats'
        data = np.genfromtxt(data_path, delimiter=';')

        val_x.append(data)
    val_x = 1*np.array(val_x)

    test_x = []
    for file in test_list.readlines():
        filename = file[:-1].split(' ')[0]
        data_path = feat_dir + filename + '.feats'
        data = np.genfromtxt(data_path, delimiter=';')

        test_x.append(data)
    test_x = 1*np.array(test_x)

    # Prediction
    model = cPickle.load(open(model_file, 'rb'))
    pred = model.predict(val_x)
    decision = model.decision_function(val_x)

    #
    # Writing results
    for p in decision:
        fwrite.write(str(p) + '\n')
    fwrite.close()

    fwrite = open('{}/test_{}'.format(output_file.split('/')[0], output_file.split('/')[1]), 'w')
    pred = model.predict(test_x)
    decision = model.decision_function(test_x)

    # Writing results
    for p in decision:
        fwrite.write(str(p) + '\n')
    fwrite.close()
