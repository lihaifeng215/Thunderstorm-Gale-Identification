import numpy as np
import os
'''
1. 利用冰雹记录构建冰雹样本集
2. 输入：冰雹发生的时间，相对坐标
3. 输出：构建的冰雹样本集。
'''
data = np.loadtxt("file1/bingbao1.txt",delimiter=",")
def bingbao2radar(bingbao_time):
    str_time = str(int(bingbao_time))
    number = int(str_time[-2:]) % 6
    if number <= 3:
        time = int(str_time[-2:]) // 6 * 6
    else:
        time = int(str_time[-2:]) // 6 * 6 + 6
    time = str(int(time)).zfill(2)
    str_time = str_time[:-2] + time
    return int(str_time)

size = 3
dataset_all = []
# find radar of bingbao time
for i in data:
    dataset = []
    str_time = str(int(bingbao2radar(i[0])))
    year = str_time[:4]
    directory = "radar" + year
    path = os.path.join(r'/home/ices/Documents',directory)
    for hight in [1500,2500,3500,4500,5500,6500,7500,8500,9500]:
        file = 'cappi_ref_' + str_time + '_' + str(hight) + '_0.ref'
        data = np.fromfile(os.path.join(path, file),dtype=np.uint8).reshape(700,900)
        data[(data<0) | (data>80)] = 0
        dataset.append(data[int(i[1])-size:int(i[1])+size+1,int(i[2])-size:int(i[2])+size+1])
    dataset_all.append(dataset)

dataset_all = np.array(dataset_all)
print(dataset_all.shape)
data_r = np.max(dataset_all,axis=1)
print(data_r.shape)
data_p = data_r.reshape(22,-1)
label_p = np.ones(shape=(data_r.shape[0],),dtype=int)
data_p = np.column_stack((data_p, label_p))
print("data_p.shape:",data_p.shape)
np.savetxt("positive_dataset.txt",data_p,delimiter=',',fmt='%d')

negitive_data = np.random.randint(0,80,(size * 2 + 1) ** 2 * 100).reshape(100,7,7)
label_n = np.zeros(shape=(negitive_data.shape[0],),dtype=int)
data_n = negitive_data.reshape(100,-1)
data_n = np.column_stack((data_n, label_n))
print("data_n.shape:",data_n.shape)
np.savetxt("negitive_dataset.txt",data_n,delimiter=',',fmt='%d')

data = np.row_stack((data_p,data_n))
np.random.shuffle(data)
print("data.shape:",data.shape)
np.savetxt("file1/dataset.txt",data,delimiter=',',fmt='%d')
