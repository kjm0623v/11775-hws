import numpy as np
from glob import glob

res_files = glob('resnet50/*.npy')
for res_file in res_files:
    filename = res_file.split('.')[0].split('/')[-1]
    places_file = 'places/' + filename + '.npy'
    output_file = 'hybrid/' + filename + '.npy'
    
    res_data = np.load(res_file)
    places_data = np.load(places_file)
    hybrid_data = np.concatenate([res_data, places_data], axis=0)
    np.save(output_file, hybrid_data)
    