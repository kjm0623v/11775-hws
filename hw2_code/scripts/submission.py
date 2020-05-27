import os
import numpy as np
import pandas as pd
import _pickle as cPickle

# data loading
test_list = open('../../all_test.video', 'r')
feat_dir = '../resnet50/'
#feat_dir = '../kmeans/'
x = []
filenames = []
for file in test_list.readlines():
    filename = file[:-1]
    filenames.append(filename)
    data_path = feat_dir + filename + '.npy'
    data = np.load(data_path)
    x.append(data)            
x = 1*np.array(x)
    
# Prediction
events = ['P001','P002','P003']
preds = []
for event in events:
    model_path = '../resnet50_pred/svm.' + event + '.model'
    #model_path = '../surf_pred/svm.' + event + '.model'
    print(model_path)
    model = cPickle.load(open(model_path, 'rb'))
    pred = model.decision_function(x)
    preds.append(pred)
    print(pred)
    print("=======")
    
preds = np.array(preds)
final = np.zeros(preds.shape[1])
for i in range(preds.shape[1]):
    a = preds[:, i]
    if max(a) > -0.7:
        final[i] = np.argmax(a) + 1
    else:
        final[i] = 3
final = final.astype(int)
