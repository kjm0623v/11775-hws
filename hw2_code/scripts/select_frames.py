#!/bin/python
# Randomly select 

import numpy
import os
import sys
from tqdm import tqdm

if __name__ == '__main__':
    file_list = 'list/train.video'
    output_file = 'select.surf.csv'
    ratio = 0.35

    fread = open(file_list,"r")
    fwrite = open(output_file,"w")

    # random selection is done by randomizing the rows of the whole matrix, and then selecting the first 
    # num_of_frame * ratio rows
    numpy.random.seed(2020)

    print("[*] Select random 30% of frames")
    for line in tqdm(fread.readlines()):
        surf_path = "surf/" + line.replace('\n','') + ".npy"
        if os.path.exists(surf_path) == False:
            continue
        array = numpy.load(surf_path)
        numpy.random.shuffle(array)
        select_size = int(array.shape[0] * ratio)
        feat_dim = array.shape[1]

        for n in range(select_size):
            line = str(array[n][0])
            for m in range(1, feat_dim):
                line += ';' + str(array[n][m])
            fwrite.write(line + '\n')
    fwrite.close()

