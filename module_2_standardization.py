import numpy as np
from sklearn.preprocessing import MinMaxScaler
'''
1. 标准化包括：
    a. 将雷达回波（dbz）的取值范围规定在（0，80），将回波数值为默认值（125）或为负值（<0）的数值设置为0
    b. 将雷达回波的数值归一化的（0，1）,便于后期模型的输入
2. 输入：上一步提取的雷达回波区域图像
3. 输出：标准化后的雷达区域图像
'''
# data = np.loadtxt("file/positive_sample_17y_Radar_denoised_13x13.csv", delimiter=',')
data = np.loadtxt("file/negative_sample_05_17y_Radar_denoised_13x13.csv", delimiter=',')
data[:,5:] = data[((data < 0) | (data > 80))] = 0
data[:,5:] = MinMaxScaler().fit_transform(data[:,5:])
# np.savetxt("file/positive_sample_17y_Radar_denoised_13x13_standardization.csv",data,fmt='%f',delimiter=',')
np.savetxt("file/negative_sample_05_17y_Radar_denoised_13x13_standardization.csv",data,fmt='%f',delimiter=',')

