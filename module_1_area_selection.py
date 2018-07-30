import numpy as np
import os
'''
1. 根据自动站的时间和相对坐标，提取对应时间大小为（size * 2 + 1）* （size * 2 + 1）的9个高度层的雷达子图像，这个过程构成区域选择模块。
2. 输入：自动站时间及相对坐标文件，雷达图像目录，高度层数，区域子图size
3. 输出：每个自动站位置的区域9个高度层的区域子图。
'''
class Radar_features:
    def __init__(self):
        pass

    '''
    get sub images of all layers
    '''
    def get_subImage_all(self, path, time, row, col, hightList, size):
        # path
        subImage_all = []
        for hight in hightList:
            data = np.fromfile(path + "cappi_ref_" + str(time) + '_' + str(hight) + "_0" + '.ref',dtype=np.int8).reshape(700, 900)
            # 当前高度层回波
            subImage_now = data[row - size:row + size + 1, col - size:col + size + 1].reshape((size * 2 + 1) ** 2, )
            # 所有高度层回波
            subImage_all.extend(subImage_now)
        subImage_all = np.array(subImage_all).reshape(len(hightList), size * 2 + 1, size * 2 + 1)
        subImage_all[((subImage_all < 0) | (subImage_all > 80))] = 0
        return subImage_all

    """
    get reflectivity
    """
    def get_R(self, subImage_all):
        print("subImage_all.shape", subImage_all.shape)
        return np.array(subImage_all).reshape(subImage_all.shape[0], -1)

    '''
    get time before 6 minutes
    '''
    def get_time_b6m(self, time):
        import datetime
        now = datetime.datetime.strptime(str(time), '%Y%m%d%H%M')
        b6m = now - datetime.timedelta(minutes=6)
        time_b6m = b6m.strftime('%Y%m%d%H%M')
        return time_b6m

    '''
    将自动站时间转换为对应的雷达时间
    input : int AWS_time
    output : int Radar_time
    '''
    def AWS_to_Radar_time(self, AWS_time):
        time_dict = {'00': 0, '05': 1, '06': 0, '10': 2, '12':0,  '15': 3,'18':0, '20': -2,'24':0, '25': -1, '30': 0, '35': 1,'36':0, '40': 2,'42':0, "45": 3,
                     '48':0,"50": -2,'54':0,'55': -1}
        return AWS_time + time_dict[str(AWS_time)[-2:]]

class Radar_features_append(Radar_features):
    def __init__(self):
        pass

    '''
    find_subImage_all
    '''
    def find_subImage_all(self, AWS_sample, hightList, directory, size):
        print("AWS_sample.shape:", AWS_sample.shape)
        dataset = []
        for i in range(AWS_sample.shape[0]):
            print("i:", i)
            # path = directory + str(int(AWS_sample[i][0]))[:8] + '\\'
            # 使用后缀有ref的雷达数据
            path = os.path.join(directory,str(int(AWS_sample[i][0]))[:8] + 'ref\\')
            print("path:",path)
            radar_sample1 = self.get_subImage_all(path, self.AWS_to_Radar_time(int(AWS_sample[i][0])), int(AWS_sample[i][2]), int(AWS_sample[i][3]), hightList, size)
            print(self.AWS_to_Radar_time(int(AWS_sample[i][0])),int(AWS_sample[i][2]),int(AWS_sample[i][3]))
            dataset.append(radar_sample1)
        sub_Image_all = np.array(dataset)
        print("find_subImage_all.shape:", sub_Image_all.shape)
        return sub_Image_all

    '''
    find_subImage_all_b6m
    '''
    def find_subImage_all_b6m(self, AWS_sample, hightList, directory, size):
        print("AWS_sample.shape:", AWS_sample.shape)
        dataset = []
        for i in range(AWS_sample.shape[0]):
            print(i)
            # path = directory + str(int(AWS_sample[i][0]))[:8] + '\\'
            # 使用后缀有ref的雷达数据
            path = directory + str(self.get_time_b6m(self.AWS_to_Radar_time(int(AWS_sample[i][0]))))[:8] + 'ref\\'
            print("path:",path)
            radar_sample1 = self.get_subImage_all(path, self.get_time_b6m(self.AWS_to_Radar_time(int(AWS_sample[i][0]))), int(AWS_sample[i][2]), int(AWS_sample[i][3]), hightList, size)
            print(self.get_time_b6m(self.AWS_to_Radar_time(int(AWS_sample[i][0]))),int(AWS_sample[i][2]),int(AWS_sample[i][3]))
            dataset.append(radar_sample1)
        sub_Image_all_b6m = np.array(dataset)
        print("find_subImage_all_b6m.shape:", sub_Image_all_b6m.shape)
        return sub_Image_all_b6m

    '''
    append R
    '''
    def append_R(self, origin_data, subImage_all):
        print("origin_data.shape:", origin_data.shape)
        dataset = self.get_R(subImage_all)
        append_r = np.column_stack((origin_data, dataset))
        print("append_r.shape:", append_r.shape)
        return append_r

    '''
    remove the noise record
    '''
    def remove_noise(self, dataset):
        label = []
        for i in range(dataset.shape[0]):
            # 过滤条件
            if np.median(dataset[i, 5:]) < 1:
                label.append(i)
        # print(label)
        print("len(noise label):", len(label))
        label = np.array(label)
        print("noise label :", len(label) / dataset.shape[0])
        denoised_dataset = np.delete(dataset, label, axis=0)
        print("denoised_dataset.shape:", denoised_dataset.shape)
        return denoised_dataset

if __name__ == '__main__':
    hightList = [1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500]
    directory = r"F:\2017radar"
    size = 6
    # 待合并的自动站数据
    # AWS_sample = np.loadtxt("file/positive_sample_17y_filtered.csv", delimiter=',')
    AWS_sample = np.loadtxt("file/negative_sample_05_17y_filtered.csv", delimiter=',')
    print("AWS_sample.shape:", AWS_sample.shape)

    # # 合并雷达数据
    Radar = Radar_features_append()
    subImage_all_dbz = Radar.find_subImage_all(AWS_sample, hightList, directory, size)
    data = Radar.append_R(AWS_sample, subImage_all_dbz)
    print("data.shape:", data.shape)

    # 去除回波不满足条件的数据
    data_denoised = Radar.remove_noise(data)
    print("data_denoised.shape:", data_denoised.shape)
    # np.savetxt("file/positive_sample_17y_Radar_denoised_13x13.csv", data_denoised, delimiter=',', fmt='%f')
    # np.savetxt("file/negative_sample_05_17y_Radar_denoised_13x13.csv", data_denoised, delimiter=',', fmt='%f')
