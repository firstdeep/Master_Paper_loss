import os
from shutil import copyfile, move

# flip along z-axis
# image000 --> image max

src_dir = '/home/hyoseok/research/medical/aaa/dataset/'
tgt_dir = '/home/hyoseok/research/medical/aaa/dataset/'

dir_list = os.listdir(src_dir + '/raw')

for entry in dir_list:
    file_list = os.listdir(src_dir + '/raw/' + entry)
    file_list_inv = os.listdir(src_dir + '/raw/' + entry)
    file_list.sort()
    file_list_inv.sort(reverse=True)
    num_files = len(file_list)

    if not os.path.exists(src_dir + '/raw_zinv/' + entry):
        os.makedirs(src_dir + '/raw_zinv/' + entry)


    for i in range(num_files):
        name1 = file_list[i]
        name2 = file_list_inv[i]
        print('%s --> %s'%(name1, name2))

        src_name = src_dir + '/raw/' + entry + '/' + name1
        tgt_name = src_dir + '/raw_zinv/' + entry + '/' + name2

        print(src_name)
        print(tgt_name)

        copyfile(src_name, tgt_name)



    # for f in file_list:
