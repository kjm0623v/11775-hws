import numpy as np
from glob import glob

filelist = glob('../places/*.npy')
#print(filelist)
for file in filelist:
    data = np.load(file)
    if data.shape[0] != 4096:
        print(file)
        