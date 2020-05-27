import numpy as np

for i in range(1, 4):
    test_list = open('../all_val.lst', 'r')
    trg_label = 'P00' + str(i)
    output_file = 'list/P00' + str(i) + '_val_label'
    fwrite = open(output_file, 'w')
    
    for line in test_list.readlines():
        label = line[:-1].split(' ')[1]

        if label == trg_label:
            fwrite.write('1' + '\n')
        else:
            fwrite.write('0' + '\n')
            
    fwrite.close()
    